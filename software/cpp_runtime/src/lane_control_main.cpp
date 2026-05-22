#include "lane_detector.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <termios.h>
#include <unistd.h>

namespace {
constexpr canid_t CAN_ID_ARM = 0x200;
constexpr canid_t CAN_ID_CTRL = 0x201;

constexpr int STEER_MIN = 1050;
constexpr int STEER_MAX = 1950;
constexpr int STEER_NEU = 1500;
constexpr int STEER_FALLBACK_LEFT = 1100;
constexpr int STEER_FALLBACK_RIGHT = 1900;

constexpr int THR_MIN = 1000;
constexpr int THR_MAX = 2000;
constexpr int THR_NEU = 1500;

template <typename T>
T clamp_value(T v, T lo, T hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

struct Args {
    std::string engine_path;
    int camera_index = 0;
    int camera_width = 1280;
    int camera_height = 720;
    int camera_fps = 120;
    int camera_buffer_size = 1;
    std::string camera_fourcc = "UYVY";
    float threshold = 0.3f;
    double crop_top_fraction = 0.35;
    double target_x_bias_px = 40.0;
    int input_width = 256;
    int input_height = 144;
    int num_sample_rows = 16;

    std::string can_channel = "can0";
    int send_hz = 20;

    double steering_gain = 0.9;
    double steering_kd = 0.3;
    double steering_deadband_px = 10.0;
    double steering_smoothing = 0.55;
    int steering_limit_us = 220;
    int min_centers = 8;
    int max_bad_frames = 3;
    double single_lane_gain_boost = 1.6;
    int trim_step_us = 10;
    int throttle_step_us = 5;
    int max_steer_step_us = 60;
    int auto_throttle_boost_us = 1545;
    int auto_throttle_dip_us = 1535;
    int auto_throttle_hold_us = 1540;
    int auto_throttle_boost_cycles = 1;
    int auto_throttle_dip_cycles = 1;
    int auto_throttle_repeat_cycles = 0;
    double print_every = 0.5;
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
        else if (key == "--camera") args.camera_index = std::stoi(next("--camera"));
        else if (key == "--camera-width") args.camera_width = std::stoi(next("--camera-width"));
        else if (key == "--camera-height") args.camera_height = std::stoi(next("--camera-height"));
        else if (key == "--camera-fps") args.camera_fps = std::stoi(next("--camera-fps"));
        else if (key == "--camera-buffer-size") args.camera_buffer_size = std::stoi(next("--camera-buffer-size"));
        else if (key == "--camera-fourcc") args.camera_fourcc = next("--camera-fourcc");
        else if (key == "--threshold") args.threshold = std::stof(next("--threshold"));
        else if (key == "--crop-top-fraction") args.crop_top_fraction = std::stod(next("--crop-top-fraction"));
        else if (key == "--target-x-bias-px") args.target_x_bias_px = std::stod(next("--target-x-bias-px"));
        else if (key == "--input-width") args.input_width = std::stoi(next("--input-width"));
        else if (key == "--input-height") args.input_height = std::stoi(next("--input-height"));
        else if (key == "--num-sample-rows") args.num_sample_rows = std::stoi(next("--num-sample-rows"));
        else if (key == "--can-channel") args.can_channel = next("--can-channel");
        else if (key == "--send-hz") args.send_hz = std::stoi(next("--send-hz"));
        else if (key == "--steering-gain") args.steering_gain = std::stod(next("--steering-gain"));
        else if (key == "--steering-kd") args.steering_kd = std::stod(next("--steering-kd"));
        else if (key == "--steering-deadband-px") args.steering_deadband_px = std::stod(next("--steering-deadband-px"));
        else if (key == "--steering-smoothing") args.steering_smoothing = std::stod(next("--steering-smoothing"));
        else if (key == "--steering-limit-us") args.steering_limit_us = std::stoi(next("--steering-limit-us"));
        else if (key == "--min-centers") args.min_centers = std::stoi(next("--min-centers"));
        else if (key == "--max-bad-frames") args.max_bad_frames = std::stoi(next("--max-bad-frames"));
        else if (key == "--single-lane-gain-boost") args.single_lane_gain_boost = std::stod(next("--single-lane-gain-boost"));
        else if (key == "--trim-step-us") args.trim_step_us = std::stoi(next("--trim-step-us"));
        else if (key == "--throttle-step-us") args.throttle_step_us = std::stoi(next("--throttle-step-us"));
        else if (key == "--max-steer-step-us") args.max_steer_step_us = std::stoi(next("--max-steer-step-us"));
        else if (key == "--auto-throttle-boost-us") args.auto_throttle_boost_us = std::stoi(next("--auto-throttle-boost-us"));
        else if (key == "--auto-throttle-dip-us") args.auto_throttle_dip_us = std::stoi(next("--auto-throttle-dip-us"));
        else if (key == "--auto-throttle-hold-us") args.auto_throttle_hold_us = std::stoi(next("--auto-throttle-hold-us"));
        else if (key == "--auto-throttle-boost-cycles") args.auto_throttle_boost_cycles = std::stoi(next("--auto-throttle-boost-cycles"));
        else if (key == "--auto-throttle-dip-cycles") args.auto_throttle_dip_cycles = std::stoi(next("--auto-throttle-dip-cycles"));
        else if (key == "--auto-throttle-repeat-cycles") args.auto_throttle_repeat_cycles = std::stoi(next("--auto-throttle-repeat-cycles"));
        else if (key == "--print-every") args.print_every = std::stod(next("--print-every"));
        else if (key == "--headless") args.headless = true;
        else throw std::runtime_error("Unknown argument: " + key);
    }

    if (args.engine_path.empty()) throw std::runtime_error("Pass --engine <path-to-engine>");
    return args;
}

class TerminalKeyboard {
public:
    TerminalKeyboard() {
        if (tcgetattr(STDIN_FILENO, &old_) != 0) {
            throw std::runtime_error("Failed to read terminal settings.");
        }
        termios raw = old_;
        raw.c_lflag &= static_cast<unsigned int>(~(ICANON | ECHO));
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) {
            throw std::runtime_error("Failed to set terminal raw mode.");
        }

        const int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        if (flags < 0 || fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK) < 0) {
            throw std::runtime_error("Failed to make stdin non-blocking.");
        }
        old_flags_ = flags;
    }

    ~TerminalKeyboard() {
        tcsetattr(STDIN_FILENO, TCSANOW, &old_);
        if (old_flags_ >= 0) {
            fcntl(STDIN_FILENO, F_SETFL, old_flags_);
        }
    }

    std::optional<char> poll_key() {
        char ch = 0;
        const ssize_t n = ::read(STDIN_FILENO, &ch, 1);
        if (n == 1) return ch;
        return std::nullopt;
    }

private:
    termios old_{};
    int old_flags_ = -1;
};

class CanBus {
public:
    explicit CanBus(const std::string& ifname) {
        socket_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
        if (socket_fd_ < 0) throw std::runtime_error("Failed to create CAN socket.");

        ifreq ifr{};
        std::strncpy(ifr.ifr_name, ifname.c_str(), IFNAMSIZ - 1);
        if (ioctl(socket_fd_, SIOCGIFINDEX, &ifr) < 0) {
            ::close(socket_fd_);
            throw std::runtime_error("Failed to resolve CAN interface: " + ifname);
        }

        sockaddr_can addr{};
        addr.can_family = AF_CAN;
        addr.can_ifindex = ifr.ifr_ifindex;
        if (bind(socket_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            ::close(socket_fd_);
            throw std::runtime_error("Failed to bind CAN socket.");
        }
    }

    ~CanBus() {
        if (socket_fd_ >= 0) ::close(socket_fd_);
    }

    void send_arm(bool armed) {
        can_frame frame{};
        frame.can_id = CAN_ID_ARM;
        frame.can_dlc = 1;
        frame.data[0] = armed ? 1 : 0;
        write_frame(frame);
    }

    void send_ctrl(int steer_us, int thr_us) {
        can_frame frame{};
        frame.can_id = CAN_ID_CTRL;
        frame.can_dlc = 4;
        frame.data[0] = static_cast<unsigned char>(steer_us & 0xFF);
        frame.data[1] = static_cast<unsigned char>((steer_us >> 8) & 0xFF);
        frame.data[2] = static_cast<unsigned char>(thr_us & 0xFF);
        frame.data[3] = static_cast<unsigned char>((thr_us >> 8) & 0xFF);
        write_frame(frame);
    }

private:
    int socket_fd_ = -1;

    void write_frame(const can_frame& frame) {
        const ssize_t n = ::write(socket_fd_, &frame, sizeof(frame));
        if (n != sizeof(frame)) throw std::runtime_error("Failed to send CAN frame.");
    }
};

class AutoSteerControllerCpp {
public:
    explicit AutoSteerControllerCpp(const Args& args)
        : steering_gain_(args.steering_gain),
          steering_kd_(args.steering_kd),
          steering_deadband_px_(args.steering_deadband_px),
          steering_smoothing_(args.steering_smoothing),
          steering_limit_us_(args.steering_limit_us),
          min_centers_(args.min_centers),
          max_bad_frames_(args.max_bad_frames),
          single_lane_gain_boost_(args.single_lane_gain_boost),
          throttle_step_us_(args.throttle_step_us),
          max_steer_step_us_(args.max_steer_step_us),
          auto_throttle_boost_us_(args.auto_throttle_boost_us),
          auto_throttle_dip_us_(args.auto_throttle_dip_us),
          auto_throttle_hold_us_(args.auto_throttle_hold_us),
          auto_throttle_boost_cycles_(std::max(args.auto_throttle_boost_cycles, 1)),
          auto_throttle_dip_cycles_(std::max(args.auto_throttle_dip_cycles, 1)),
          auto_throttle_repeat_cycles_(std::max(args.auto_throttle_repeat_cycles, 0)) {}

    void toggle_arm() {
        armed_ = !armed_;
        if (!armed_) {
            prev_delta_us_ = 0.0;
            prev_steering_error_px_ = 0.0;
            last_steer_us_ = STEER_NEU;
            throttle_us_ = THR_NEU;
            auto_throttle_enabled_ = false;
            auto_throttle_stage_ = 0;
            auto_throttle_cycles_in_stage_ = 0;
            hold_cycles_since_restart_ = 0;
        }
    }

    bool is_armed() const { return armed_; }

    void nudge_trim(int delta_us) {
        manual_trim_us_ = clamp_value(manual_trim_us_ + delta_us, -steering_limit_us_, steering_limit_us_);
    }

    void clear_controls() {
        manual_trim_us_ = 0;
        throttle_us_ = THR_NEU;
        auto_throttle_enabled_ = false;
        auto_throttle_stage_ = 0;
        auto_throttle_cycles_in_stage_ = 0;
        hold_cycles_since_restart_ = 0;
    }

    void throttle_up() {
        auto_throttle_enabled_ = false;
        auto_throttle_stage_ = 0;
        auto_throttle_cycles_in_stage_ = 0;
        hold_cycles_since_restart_ = 0;
        throttle_us_ = clamp_value(throttle_us_ + throttle_step_us_, THR_MIN, THR_MAX);
    }

    void throttle_down() {
        auto_throttle_enabled_ = false;
        auto_throttle_stage_ = 0;
        auto_throttle_cycles_in_stage_ = 0;
        hold_cycles_since_restart_ = 0;
        throttle_us_ = clamp_value(throttle_us_ - throttle_step_us_, THR_MIN, THR_MAX);
    }

    void toggle_auto_throttle() {
        auto_throttle_enabled_ = !auto_throttle_enabled_;
        auto_throttle_stage_ = 0;
        auto_throttle_cycles_in_stage_ = 0;
        hold_cycles_since_restart_ = 0;
        if (!auto_throttle_enabled_) {
            throttle_us_ = THR_NEU;
        }
    }

    bool auto_throttle_enabled() const { return auto_throttle_enabled_; }

    struct Output {
        int steer_us = STEER_NEU;
        int throttle_us = THR_NEU;
        bool armed = false;
        int trim_us = 0;
        int bad_frames = 0;
    };

    Output compute(const DetectionResult& result) {
        const bool lane_ok =
            result.steering_error.has_value() &&
            static_cast<int>(result.centers.size()) >= min_centers_ &&
            (result.lane_state.both_visible || result.lane_state.single_visible);

        if (lane_ok) bad_frames_ = 0;
        else ++bad_frames_;

        double target_delta_us = 0.0;
        std::string force_full_lock;
        if (!armed_ || bad_frames_ >= max_bad_frames_) {
            target_delta_us = 0.0;
            prev_steering_error_px_ = 0.0;
        } else if (result.lane_state.single_visible &&
                   !result.lane_state.both_visible &&
                   (result.lane_state.single_lane_turn_direction == "left" ||
                    result.lane_state.single_lane_turn_direction == "right")) {
            target_delta_us = 0.0;
            force_full_lock = result.lane_state.single_lane_turn_direction;
            prev_steering_error_px_ = result.steering_error.value_or(0);
        } else if (result.steering_error && std::abs(*result.steering_error) > steering_deadband_px_) {
            const double half_width = std::max(result.cropped_frame.cols / 2.0, 1.0);
            const double normalized_error = *result.steering_error / half_width;
            const double d_error = *result.steering_error - prev_steering_error_px_;
            const double normalized_d_error = d_error / half_width;
            prev_steering_error_px_ = *result.steering_error;

            const double p_term = steering_gain_ * normalized_error * steering_limit_us_;
            const double d_term = steering_kd_ * normalized_d_error * steering_limit_us_;
            target_delta_us = p_term + d_term;

            if (result.lane_state.single_visible && !result.lane_state.both_visible) {
                target_delta_us *= single_lane_gain_boost_;
            }
        } else if (result.steering_error) {
            prev_steering_error_px_ = *result.steering_error;
        }

        target_delta_us = clamp_value(target_delta_us, -static_cast<double>(steering_limit_us_), static_cast<double>(steering_limit_us_));
        const double delta_us = steering_smoothing_ * prev_delta_us_ + (1.0 - steering_smoothing_) * target_delta_us;
        prev_delta_us_ = force_full_lock.empty() ? delta_us : 0.0;

        int target_steer_us = STEER_NEU;
        if (force_full_lock == "left") {
            target_steer_us = STEER_FALLBACK_LEFT;
        } else if (force_full_lock == "right") {
            target_steer_us = STEER_FALLBACK_RIGHT;
        } else {
            target_steer_us = clamp_value(static_cast<int>(std::lround(STEER_NEU + delta_us + manual_trim_us_)), STEER_MIN, STEER_MAX);
        }
        int steer_delta = target_steer_us - last_steer_us_;
        if (steer_delta > max_steer_step_us_) target_steer_us = last_steer_us_ + max_steer_step_us_;
        else if (steer_delta < -max_steer_step_us_) target_steer_us = last_steer_us_ - max_steer_step_us_;

        last_steer_us_ = clamp_value(target_steer_us, STEER_MIN, STEER_MAX);

        Output out;
        out.steer_us = last_steer_us_;
        out.throttle_us = throttle_us_;
        out.armed = armed_;
        out.trim_us = manual_trim_us_;
        out.bad_frames = bad_frames_;
        return out;
    }

    int current_throttle_for_send() {
        if (!armed_) return THR_NEU;
        if (!auto_throttle_enabled_) return throttle_us_;

        switch (auto_throttle_stage_) {
            case 0:
                throttle_us_ = auto_throttle_boost_us_;
                ++auto_throttle_cycles_in_stage_;
                if (auto_throttle_cycles_in_stage_ >= auto_throttle_boost_cycles_) {
                    auto_throttle_stage_ = 1;
                    auto_throttle_cycles_in_stage_ = 0;
                    hold_cycles_since_restart_ = 0;
                }
                break;
            case 1:
                throttle_us_ = auto_throttle_dip_us_;
                ++auto_throttle_cycles_in_stage_;
                if (auto_throttle_cycles_in_stage_ >= auto_throttle_dip_cycles_) {
                    auto_throttle_stage_ = 2;
                    auto_throttle_cycles_in_stage_ = 0;
                    hold_cycles_since_restart_ = 0;
                }
                break;
            case 2:
            default:
                throttle_us_ = auto_throttle_hold_us_;
                if (auto_throttle_repeat_cycles_ > 0) {
                    ++hold_cycles_since_restart_;
                    if (hold_cycles_since_restart_ >= auto_throttle_repeat_cycles_) {
                        auto_throttle_stage_ = 0;
                        auto_throttle_cycles_in_stage_ = 0;
                        hold_cycles_since_restart_ = 0;
                    }
                }
                break;
        }
        return throttle_us_;
    }

    int displayed_throttle() const {
        if (!armed_) return THR_NEU;
        return throttle_us_;
    }

private:
    double steering_gain_;
    double steering_kd_;
    double steering_deadband_px_;
    double steering_smoothing_;
    int steering_limit_us_;
    int min_centers_;
    int max_bad_frames_;
    double single_lane_gain_boost_;
    int throttle_step_us_;
    int max_steer_step_us_;
    int auto_throttle_boost_us_;
    int auto_throttle_dip_us_;
    int auto_throttle_hold_us_;
    int auto_throttle_boost_cycles_;
    int auto_throttle_dip_cycles_;
    int auto_throttle_repeat_cycles_;

    bool armed_ = false;
    bool auto_throttle_enabled_ = false;
    int manual_trim_us_ = 0;
    int throttle_us_ = THR_NEU;
    int bad_frames_ = 0;
    int last_steer_us_ = STEER_NEU;
    int auto_throttle_stage_ = 0;
    int auto_throttle_cycles_in_stage_ = 0;
    int hold_cycles_since_restart_ = 0;
    double prev_delta_us_ = 0.0;
    double prev_steering_error_px_ = 0.0;
};

cv::Mat draw_assist_view(const DetectionResult& result, int steer_us, int thr_us, bool armed, int trim_us, bool auto_throttle_enabled) {
    cv::Mat overlay = result.overlay.empty() ? result.cropped_frame.clone() : result.overlay.clone();
    cv::putText(overlay, std::string("ARMED=") + (armed ? "1" : "0"), {15, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {255, 255, 255}, 2, cv::LINE_AA);
    cv::putText(overlay, "err_px=" + std::to_string(result.steering_error.value_or(0)), {15, 60}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {255, 255, 255}, 2, cv::LINE_AA);
    cv::putText(overlay, "steer_us=" + std::to_string(steer_us), {15, 90}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {255, 255, 255}, 2, cv::LINE_AA);
    cv::putText(overlay, "thr_us=" + std::to_string(thr_us), {15, 120}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {255, 255, 255}, 2, cv::LINE_AA);
    cv::putText(overlay, "trim_us=" + std::to_string(trim_us), {15, 150}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {255, 255, 255}, 2, cv::LINE_AA);
    cv::putText(overlay, std::string("auto_thr=") + (auto_throttle_enabled ? "1" : "0"), {15, 180}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {255, 255, 255}, 2, cv::LINE_AA);
    return overlay;
}
}  // namespace

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

        AutoSteerControllerCpp controller(args);
        TerminalKeyboard keyboard;
        CanBus bus(args.can_channel);

        cv::VideoCapture cap(args.camera_index);
        if (!cap.isOpened()) throw std::runtime_error("Failed to open camera.");

        if (!args.camera_fourcc.empty() && args.camera_fourcc.size() == 4) {
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc(
                args.camera_fourcc[0], args.camera_fourcc[1], args.camera_fourcc[2], args.camera_fourcc[3]));
        }
        cap.set(cv::CAP_PROP_BUFFERSIZE, args.camera_buffer_size);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, args.camera_width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, args.camera_height);
        cap.set(cv::CAP_PROP_FPS, args.camera_fps);

        std::cout << "[INFO] C++ lane CAN control running\n";
        std::cout << "[INFO] Controls: r arm/disarm | t auto throttle on/off | w/s manual throttle override | a/d trim | space neutralize | q quit\n";
        std::cout << "[INFO] Camera requested=" << args.camera_width << "x" << args.camera_height << "@" << args.camera_fps
                  << " actual=" << static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH))
                  << "x" << static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))
                  << "@" << cap.get(cv::CAP_PROP_FPS) << "\n";

        bus.send_arm(false);
        bus.send_ctrl(STEER_NEU, THR_NEU);

        cv::Mat frame;
        const auto start_t = std::chrono::steady_clock::now();
        auto last_print_t = start_t;
        auto last_send_t = start_t;
        bool running = true;
        int frame_count = 0;

        while (running && cap.read(frame)) {
            const auto loop_t0 = std::chrono::steady_clock::now();
            while (true) {
                auto key = keyboard.poll_key();
                if (!key.has_value()) break;
                switch (*key) {
                    case 'q':
                    case 'Q':
                    case 27:
                        running = false;
                        break;
                    case 'r':
                    case 'R':
                        controller.toggle_arm();
                        bus.send_arm(controller.is_armed());
                        break;
                    case 'w':
                    case 'W':
                        controller.throttle_up();
                        break;
                    case 's':
                    case 'S':
                        controller.throttle_down();
                        break;
                    case 't':
                    case 'T':
                        controller.toggle_auto_throttle();
                        break;
                    case 'a':
                    case 'A':
                        controller.nudge_trim(-args.trim_step_us);
                        break;
                    case 'd':
                    case 'D':
                        controller.nudge_trim(args.trim_step_us);
                        break;
                    case ' ':
                        controller.clear_controls();
                        break;
                    default:
                        break;
                }
            }
            if (!running) break;

            const auto result = detector.process_frame(frame, false, !args.headless);
            const auto out = controller.compute(result);

            const auto now = std::chrono::steady_clock::now();
            const double send_period_s = 1.0 / std::max(args.send_hz, 1);
            int throttle_to_send = controller.displayed_throttle();
            if (std::chrono::duration<double>(now - last_send_t).count() >= send_period_s) {
                throttle_to_send = controller.current_throttle_for_send();
                bus.send_ctrl(out.armed ? out.steer_us : STEER_NEU, out.armed ? throttle_to_send : THR_NEU);
                last_send_t = now;
            }

            if (!args.headless) {
                cv::Mat view = draw_assist_view(
                    result,
                    out.steer_us,
                    throttle_to_send,
                    out.armed,
                    out.trim_us,
                    controller.auto_throttle_enabled()
                );
                cv::imshow("lane_runtime_can", view);
                if ((cv::waitKey(1) & 0xFF) == 'q') break;
            }

            ++frame_count;
            if (std::chrono::duration<double>(now - last_print_t).count() >= args.print_every) {
                const double elapsed_s = std::chrono::duration<double>(now - start_t).count();
                const double fps = frame_count / std::max(elapsed_s, 1e-6);
                const double loop_ms = std::chrono::duration<double, std::milli>(now - loop_t0).count();
                std::cout
                    << "armed=" << (out.armed ? 1 : 0)
                    << " | err_px=" << result.steering_error.value_or(0)
                    << " | centers=" << result.centers.size()
                    << " | steer_us=" << out.steer_us
                    << " | thr_us=" << throttle_to_send
                    << " | trim_us=" << out.trim_us
                    << " | bad=" << out.bad_frames
                    << " | both=" << (result.lane_state.both_visible ? 1 : 0)
                    << " | auto_thr=" << (controller.auto_throttle_enabled() ? 1 : 0)
                    << " | fps=" << fps
                    << " | predict=" << result.timings.predict_ms << "ms"
                    << " | post=" << result.timings.postprocess_ms << "ms"
                    << " | center=" << result.timings.centerline_ms << "ms"
                    << " | total=" << result.timings.total_ms << "ms"
                    << " | loop=" << loop_ms << "ms\n";
                last_print_t = now;
            }
        }

        bus.send_ctrl(STEER_NEU, THR_NEU);
        bus.send_arm(false);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "[ERROR] " << exc.what() << '\n';
        return 1;
    }
}
