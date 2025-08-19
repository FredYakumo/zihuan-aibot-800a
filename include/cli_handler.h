#ifndef CLI_HANDLER_H
#define CLI_HANDLER_H

#include <optional>

namespace neural_network { enum class Device; }

/**
 * Command line handler for one-shot maintenance / runtime flags.
 * Responsibilities:
 * 1. Process table initialization commands (init_*_table / init_all_tables)
 * 2. Parse device selection flags (--use-coreml / --use-mps / --use-cuda / --use-tensorrt / --use-cpu)
 *
 * Usage pattern in main:
 *   if (CLIHandler::handle_table_init(argc, argv)) return 0; // performed action and exit
 *   auto device = CLIHandler::parse_device(argc, argv).value_or(neural_network::Device::CPU);
 */
class CLIHandler {
public:
    // Executes table init command if present; returns true if program should exit afterwards.
    static bool handle_table_init(int argc, char *argv[]);

    // Returns chosen device if any device flag present, otherwise std::nullopt
    static std::optional<neural_network::Device> parse_device(int argc, char *argv[]);

    // Determines the device to use (falls back to CPU if none specified) and logs the choice.
    // This centralizes the previous logging switch from main() so callers can simply:
    //   auto device = CLIHandler::determine_device(argc, argv);
    static neural_network::Device determine_device(int argc, char *argv[]);
};

#endif // CLI_HANDLER_H
