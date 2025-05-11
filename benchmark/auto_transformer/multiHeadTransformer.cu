#include "apis_cu.h"
#include "device_launch_parameters.h"
#include <math_constants.h>
#include <cmath>

// Return d_k = d_model / h   (kept as a host‑side helper for brevity)
static inline int dk(int d_model, int h) { return d_model / h; }

// -----------------------------------------------------------------------------
// Kernel 1 : compute raw attention scores  Q_i  *  K_i^T  (one matrix per head)
//            scores  shape  (h, N, N)
// Each thread computes a single (row, col) element for a given head.
// N = seq_len * batch_size
// -----------------------------------------------------------------------------
__global__ void kernel_qk_dot(const float* __restrict__ Q,
                              const float* __restrict__ K,
                              float* __restrict__ scores,
                              int N, int d_model, int h)
{
    const int head = blockIdx.z;                      // [0, h)
    const int row  = blockIdx.y * blockDim.y + threadIdx.y; // query index   [0, N)
    const int col  = blockIdx.x * blockDim.x + threadIdx.x; // key index     [0, N)
    const int d_k  = d_model / h;

    if (row >= N || col >= N) return;

    const float* q_ptr = Q + row * d_model + head * d_k;
    const float* k_ptr = K + col * d_model + head * d_k;

    float dot = 0.f;
    #pragma unroll
    for (int i = 0; i < 1024; ++i) {         // compile‑time upper bound – trimmed below
        if (i >= d_k) break;
        dot += q_ptr[i] * k_ptr[i];
    }

    // scale by 1/sqrt(d_k)
    dot *= rsqrtf(static_cast<float>(d_k));

    // write to   scores[head, row, col]   with row‑major contiguous layout per head
    scores[(head * N + row) * N + col] = dot;
}

// -----------------------------------------------------------------------------
// Kernel 2 : row‑wise softmax over the (N) keys for every (head, query)
//            Operates in‑place on the scores buffer
// -----------------------------------------------------------------------------
__global__ void kernel_row_softmax(float* scores, int N)
{
    const int head = blockIdx.x;   // [0, h)
    const int row  = blockIdx.y;   // [0, N)

    // pointer to this row
    float* row_ptr = scores + (head * N + row) * N;

    // ---- compute max --------------------------------------------------------
    float max_val = row_ptr[0];
    for (int j = 1; j < N; ++j)
        max_val = fmaxf(max_val, row_ptr[j]);

    // ---- exponentiate & accumulate sum -------------------------------------
    float sum_val = 0.f;
    for (int j = 0; j < N; ++j) {
        float e = expf(row_ptr[j] - max_val);
        row_ptr[j] = e;
        sum_val += e;
    }

    // ---- normalise ---------------------------------------------------------
    float inv_sum = 1.f / sum_val;
    for (int j = 0; j < N; ++j)
        row_ptr[j] *= inv_sum;
}

// -----------------------------------------------------------------------------
// Kernel 3 :  A_i  *  V_i   -> output
//   For each (head, query) we compute a length‑d_k output vector.
//   The block handles one (head,row).  Each thread covers one d_k dimension.
// -----------------------------------------------------------------------------
__global__ void kernel_apply_av(const float* __restrict__ scores,
                                const float* __restrict__ V,
                                float* __restrict__ output,
                                int N, int d_model, int h)
{
    const int head = blockIdx.x;  // [0, h)
    const int row  = blockIdx.y;  // [0, N)
    const int dim  = threadIdx.x; // [0, d_k)

    const int d_k = d_model / h;
    if (dim >= d_k) return;

    const float* score_row = scores + (head * N + row) * N;         // length N

    float acc = 0.f;
    for (int j = 0; j < N; ++j) {
        float w  = score_row[j];
        float vj = V[j * d_model + head * d_k + dim];
        acc += w * vj;
    }

    output[row * d_model + head * d_k + dim] = acc;
}

// -----------------------------------------------------------------------------
// Public entry – signature provided by the grading harness
// -----------------------------------------------------------------------------
void solve(const float* Q, const float* K, const float* V,
                      float* output, int N, int d_model, int h)
{
    const int d_k = dk(d_model, h);

    // ---- scratch space for attention scores --------------------------------
    float* d_scores = nullptr;
    size_t scores_bytes = static_cast<size_t>(h) * N * N * sizeof(float);
    cudaMalloc(&d_scores, scores_bytes);

    // ---------------------------------------------------------------------
    // 1) QK^T   --->   d_scores
    // ---------------------------------------------------------------------
    const dim3 block1(16, 16, 1);
    const dim3 grid1((N + block1.x - 1) / block1.x,
                     (N + block1.y - 1) / block1.y,
                     h);

    kernel_qk_dot<<<grid1, block1>>>(Q, K, d_scores, N, d_model, h);
    // CUDA_CHECK(cudaPeekAtLastError());

    // ---------------------------------------------------------------------
    // 2) softmax row‑wise   (in‑place on d_scores)
    //     – one thread‑block per (head,row); single‑thread blocks for clarity.
    // ---------------------------------------------------------------------
    const dim3 grid2(h, N, 1);
    kernel_row_softmax<<<grid2, 1>>>(d_scores, N);
    // CUDA_CHECK(cudaPeekAtLastError());

    // ---------------------------------------------------------------------
    // 3) multiply with V   ->   output
    //    Each block handles one (head,row) and launches d_k threads.
    // ---------------------------------------------------------------------
    const dim3 grid3(h, N, 1);
    const dim3 block3(d_k, 1, 1);   // d_k <= 1024 assumed; adjust if needed

    kernel_apply_av<<<grid3, block3>>>(d_scores, V, output, N, d_model, h);
    // CUDA_CHECK(cudaPeekAtLastError());

    // ---- cleanup -----------------------------------------------------------
    // CUDA_CHECK(cudaFree(d_scores));
    cudaFree(d_scores);
}