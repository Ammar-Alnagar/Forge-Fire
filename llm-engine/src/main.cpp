#include <iostream>
#include <string>
#include <vector>
#include "loaders/onnx_loader.hpp"

void print_usage() {
    std::cout << "Usage: infer --model <path_to_onnx_model> [options]\n";
}

int main(int argc, char* argv[]) {
    std::string model_path;
    std::vector<std::string> args(argv + 1, argv + argc);

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--model" && i + 1 < args.size()) {
            model_path = args[++i];
        }
        // Other arguments like --prompt, --max-tokens will be added later
    }

    if (model_path.empty()) {
        print_usage();
        return 1;
    }

    std::cout << "Loading model from: " << model_path << std::endl;

    try {
        auto initializers = load_onnx_initializers(model_path);
        std::cout << "Successfully loaded " << initializers.size() << " initializers:" << std::endl;
        for (const auto& pair : initializers) {
            std::cout << " - " << pair.first << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}