#include "lane_detector.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

struct Args {
    std::string engine_path;
    std::optional<std::string> video_path;
    std::optional<std::string> save_path;
    int camera_index = 0;
    int camera_width = 1280;
    int camera_height = 720;
    int camera_fps = 60;
    std::string camera_fourcc = "UYVY";
    float threshold = 0.3f;
    double crop_top_fraction = 0.35;
    double target_x_bias_px = 40.0;
    int input_width = 256;
    int input_height = 144;
    int num_sample_rows = 16;
    bool headless = false;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&](const char* name) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
            return std::string(argv[++i]);
        };
        if (key == "--engine") args.engine_path = next("--engine");
        else if (key == "--video") args.video_path = next("--video");
        else if (key == "--save") args.save_path = next("--save");
        else if (key == "--camera") args.camera_index = std::stoi(next("--camera"));
        else if (key == "--camera-width") args.camera_width = std::stoi(next("--camera-width"));
        else if (key == "--camera-height") args.camera_height = std::stoi(next("--camera-height"));
        else if (key == "--camera-fps") args.camera_fps = std::stoi(next("--camera-fps"));
        else if (key == "--camera-fourcc") args.camera_fourcc = next("--camera-fourcc");
        else if (key == "--threshold") args.threshold = std::stof(next("--threshold"));
        else if (key == "--crop-top-fraction") args.crop_top_fraction = std::stod(next("--crop-top-fraction"));
        else if (key == "--target-x-bias-px") args.target_x_bias_px = std::stod(next("--target-x-bias-px"));
        else if (key == "--input-width") args.input_width = std::stoi(next("--input-width"));
        else if (key == "--input-height") args.input_height = std::stoi(next("--input-height"));
        else if (key == "--num-sample-rows") args.num_sample_rows = std::stoi(next("--num-sample-rows"));
        else if (key == "--headless") args.headless = true;
        else throw std::runtime_error("Unknown argument: " + key);
    }
    if (args.engine_path.empty()) throw std::runtime_error("Pass --engine <path-to-engine>");
    return args;
}

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        LaneCenterlineDetectorCpp detector(
            args.engine_path,
            args.threshold,
            cv::Size(args.input_width, args.input_height),
            args.crop_top_fraction,
            args.target_x_bias_px,
            args.num_sample_rows,
            true
        );

        cv::VideoCapture cap;
        if (args.video_path) {
            cap.open(*args.video_path);
        } else {
            cap.open(args.camera_index);
            if (!args.camera_fourcc.empty()) {
                cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc(
                    args.camera_fourcc[0], args.camera_fourcc[1], args.camera_fourcc[2], args.camera_fourcc[3]));
            }
            cap.set(cv::CAP_PROP_FRAME_WIDTH, args.camera_width);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, args.camera_height);
            cap.set(cv::CAP_PROP_FPS, args.camera_fps);
        }

        if (!cap.isOpened()) throw std::runtime_error("Failed to open input source.");

        cv::VideoWriter writer;
        if (args.save_path) {
            const double fps = cap.get(cv::CAP_PROP_FPS) > 1e-3 ? cap.get(cv::CAP_PROP_FPS) : 30.0;
            const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            const int cut = static_cast<int>(height * args.crop_top_fraction);
            writer.open(*args.save_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height - cut));
        }

        std::cout << "[INFO] Running C++ lane runtime\n";
        cv::Mat frame;
        int frame_count = 0;
        const auto t0 = std::chrono::steady_clock::now();
        while (cap.read(frame)) {
            const auto result = detector.process_frame(frame, false, true);
            if (writer.isOpened()) writer.write(result.overlay);
            if (!args.headless) {
                cv::imshow("lane_runtime", result.overlay);
                if ((cv::waitKey(1) & 0xFF) == 'q') break;
            }
            ++frame_count;
            if (frame_count % 10 == 0) {
                const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                const double fps = frame_count / std::max(elapsed, 1e-6);
                std::cout << "fps=" << fps
                          << " predict=" << result.timings.predict_ms << "ms"
                          << " post=" << result.timings.postprocess_ms << "ms"
                          << " center=" << result.timings.centerline_ms << "ms"
                          << " total=" << result.timings.total_ms << "ms\n";
            }
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "[ERROR] " << exc.what() << '\n';
        return 1;
    }
}
