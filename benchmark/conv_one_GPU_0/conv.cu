#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>
#include <string>

#include "apis_cu.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;

__global__ void conv(float* M1, float* kernel, float* M_out, int width, int height, int in_channels, int out_channels, int kernel_size, int stride, int padding = 3) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;

	int out_width = (width - kernel_size + 2 * padding) / stride + 1;
	int out_height = (height - kernel_size + 2 * padding) / stride + 1;
	if (x < out_width && y < out_height && z < out_channels) {
		// 计算输入特征图中卷积核覆盖的区域的起始坐标
		int x_start = x * stride - padding;
		int y_start = y * stride - padding;

		float sum = 0;
		for (int c = 0; c < in_channels; c++) {
			for (int i = 0; i < kernel_size; i++) {
				for (int j = 0; j < kernel_size; j++) {
					int input_x = x_start + j;
					int input_y = y_start + i;

					// 边界处理：如果输入坐标在边界外，则使用边界填充的值
					if (input_x >= 0 && input_x < width && input_y >= 0 && input_y < height) {
						float val = M1[input_x + input_y * width + c * width * height];
						float k_val = kernel[j + i * kernel_size + c * kernel_size * kernel_size + z * kernel_size * kernel_size * in_channels];
						sum += val * k_val;
					}
				}
			}
		}
		M_out[x + y * out_width + z * out_height * out_width] = sum;
	}
}


int main(int argc, char **argv){
    int idX = atoi(argv[1]);
    int idY = atoi(argv[2]);
    int width = 3;
    int height = 224;
    int in_channels = 224;
    int out_channels = 56;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    int out_width = (width - kernel_size + 2 * padding) / stride + 1;
    int out_height = (height - kernel_size + 2 * padding) / stride + 1;
    printf("out_width: %d, out_height: %d\n", out_width, out_height);

    float *d_M1, *d_kernel, *d_M_out;
    cudaMalloc((void **)&d_M1, width * height * in_channels * sizeof(float));
    cudaMalloc((void **)&d_kernel, kernel_size * kernel_size * in_channels * out_channels * sizeof(float));
    cudaMalloc((void **)&d_M_out, out_width * out_height * out_channels * sizeof(float));
    float *h_kernel = (float *)malloc(kernel_size * kernel_size * in_channels * out_channels * sizeof(float));
    float *h_M1 = (float *)malloc(width * height * in_channels * sizeof(float));
    for(int i = 0; i < kernel_size * kernel_size * in_channels * out_channels; i++){
        h_kernel[i] = rand() % 10;
    }
    for(int i=0;i < width * height * in_channels;i++){
        h_M1[i] = (rand() % 255); 
    }
    cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * in_channels * out_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M1, h_M1, width * height * in_channels * sizeof(float), cudaMemcpyHostToDevice);
    // receiveMessage(idX, idY, 5, 5, d_M1, width * height * in_channels*sizeof(float));

    dim3 threadsPerBlock(16, 16, 4);
    dim3 numBlocks((out_width + threadsPerBlock.x - 1) / threadsPerBlock.x, (out_height + threadsPerBlock.y - 1) / threadsPerBlock.y, (out_channels + threadsPerBlock.z - 1) / threadsPerBlock.z);

    
    conv<<<numBlocks, threadsPerBlock>>>(d_M1, d_kernel, d_M_out, width, height, in_channels, out_channels, kernel_size, stride, padding);
    // cudaDeviceSynchronize();
    // sendMessage( 5, 5, idX, idY, d_M_out, out_width * out_height * out_channels*sizeof(float));
    cudaFree(d_M1);
    cudaFree(d_kernel);
    cudaFree(d_M_out);
    return 0;
}