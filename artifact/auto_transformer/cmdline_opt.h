#include "third_party/CLI11/include/CLI/CLI.hpp"

/**
 * @defgroup cmdline
 * @brief Command line parser.
 * @{
 */
/**
 * @brief Options from command line.
 */
class CmdLineOptions {
   public:
    /**
     * @brief Constructor.
     */
    CmdLineOptions()
        : m_srcX(), m_srcY(), m_topology_width(), d_model(512), num_heads(8), N(1), src_vocab(1024), trg_vocab(1024), batch_size(10), seq_len(100), dropout(0.1) {}

    /**
     * @brief Read options from command line.
     * @param argc Number of argument.
     * @param argv String of argument.
     */
    int parse(int argc, const char* argv[]) {
        CLI::App app{"transformer"};
        app.add_option("--srcX", m_srcX, "Source X coordinate")
            ->required();
        app.add_option("--srcY", m_srcY, "Source Y coordinate")
            ->required();
        app.add_option("-t,--topology_width", m_topology_width, "Topology width")
            ->required();
        app.add_option("--d_model", d_model, "d_model")
            ->check(CLI::PositiveNumber);
        app.add_option("--num_heads", num_heads, "num_heads")
            ->check(CLI::PositiveNumber);
        app.add_option("-n", N, "N")
            ->check(CLI::PositiveNumber);
        app.add_option("--src_vocab", src_vocab, "src_vocab")
            ->check(CLI::PositiveNumber);
        app.add_option("--trg_vocab", trg_vocab, "trg_vocab")
            ->check(CLI::PositiveNumber);
        app.add_option("--batch_size", batch_size, "batch_size")
            ->check(CLI::PositiveNumber);
        app.add_option("--seq_len", seq_len, "seq_len")
            ->check(CLI::PositiveNumber);
        app.add_option("--dropout", dropout, "dropout")
            ->check(CLI::PositiveNumber);

        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError& e) {
            int ret = app.exit(e);
            exit(ret);
        }

        return 0;
    }

   public:
    /**
     * @brief Source X coordinate.
     */
    int m_srcX;
    /**
     * @brief Source Y coordinate.
     */
    int m_srcY;

    /**
     * @brief m_topology_width
    **/
    int m_topology_width;

    /**
     * @brief d_model
    **/
    int d_model;

    /**
     * @brief num_heads
    **/
    int num_heads;

    /**
     * @brief N
    **/
    int N;

    /**
     * @brief src_vocab
    **/
    int src_vocab;

    /**
     * @brief trg_vocab
    **/
    int trg_vocab;

    /**
     * @brief batch_size
    **/
    int batch_size;

    /**
     * @brief seq_len
    **/
    int seq_len;

    /**
     * @brief dropout
    **/
    double dropout;
};
/**
 * @}
 */
