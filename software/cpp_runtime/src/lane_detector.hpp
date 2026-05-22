#pragma once

#include "tensorrt_runner.hpp"

#include <opencv2/opencv.hpp>

#include <optional>
#include <string>
#include <tuple>
#include <vector>

struct TimingInfo {
    double crop_ms = 0.0;
    double predict_ms = 0.0;
    double postprocess_ms = 0.0;
    double centerline_ms = 0.0;
    double scale_and_mask_ms = 0.0;
    double total_ms = 0.0;
};

struct LaneState {
    int left_count = 0;
    int right_count = 0;
    bool both_visible = false;
    bool single_visible = false;
    std::string single_lane_turn_direction;
    bool inner_lane_missing_turn = false;
};

struct DetectionResult {
    cv::Mat cropped_frame;
    cv::Mat mask;
    cv::Mat raw_mask_vis;
    cv::Mat overlay;
    std::vector<cv::Point> centers;
    std::vector<cv::Point> left_points;
    std::vector<cv::Point> right_points;
    std::optional<cv::Point> target_point;
    std::optional<int> steering_error;
    TimingInfo timings;
    LaneState lane_state;
};

class LaneCenterlineDetectorCpp {
public:
    LaneCenterlineDetectorCpp(const std::string& engine_path,
                              float threshold = 0.3f,
                              cv::Size input_size = {256, 144},
                              double crop_top_fraction = 0.35,
                              double target_x_bias_px = 40.0,
                              int num_sample_rows = 16,
                              bool verbose = true);

    void reset_tracking();
    DetectionResult process_frame(const cv::Mat& frame, bool draw_debug = false, bool generate_visuals = true);

private:
    struct Blob {
        int left = 0;
        int right = 0;
        int center = 0;
        int width = 0;
    };

    float threshold_;
    cv::Size input_size_;
    double crop_top_fraction_;
    double target_x_bias_px_;
    bool verbose_;
    int num_sample_rows_;

    int min_component_area_ = 120;
    int min_blob_width_ = 3;
    double search_margin_frac_ = 0.18;
    double max_row_jump_frac_ = 0.18;
    double default_half_lane_width_frac_ = 0.30;
    double lane_width_smoothing_ = 0.20;
    double center_smoothing_ = 0.15;
    double min_lane_width_frac_ = 0.15;
    double max_lane_width_frac_ = 0.80;
    int point_smooth_window_ = 5;
    double target_alpha_ = 0.20;
    double target_index_far_frac_ = 0.90;
    double target_index_near_frac_ = 0.45;
    double single_lane_target_index_far_frac_ = 0.60;
    double single_lane_target_index_near_frac_ = 0.35;
    double lookahead_turn_trigger_frac_ = 0.50;
    int single_lane_curve_trigger_px_ = 4;
    double single_lane_target_bias_gain_ = 4.0;
    double single_lane_target_bias_max_frac_ = 0.75;
    std::vector<int> sample_rows_;

    cv::Mat kernel_open_;
    cv::Mat kernel_close_;

    std::optional<int> prev_target_x_;
    std::optional<int> prev_target_y_;
    std::optional<int> prev_frame_left_x_;
    std::optional<int> prev_frame_right_x_;

    TensorRTRunner trt_;

    cv::Mat apply_clahe_bgr(const cv::Mat& image) const;
    std::vector<float> prepare_input_tensor(const cv::Mat& frame_crop) const;
    std::pair<cv::Mat, cv::Mat> predict_mask(const cv::Mat& frame_crop);
    cv::Mat largest_components(const cv::Mat& mask) const;
    cv::Mat postprocess_mask(const cv::Mat& mask) const;
    std::vector<Blob> get_row_blobs(const cv::Mat& mask_row) const;
    std::optional<Blob> choose_blob_near_target(const std::vector<Blob>& blobs, int target_x, int margin) const;
    std::optional<Blob> choose_extreme_blob(const std::vector<Blob>& blobs, const std::string& side, int center_x) const;
    std::string assign_single_blob_side(const Blob& blob,
                                        const std::optional<int>& prev_left_x,
                                        const std::optional<int>& prev_right_x,
                                        int img_center_x) const;
    bool is_candidate_plausible(const std::optional<int>& candidate_x,
                                const std::optional<int>& prev_x,
                                int max_jump) const;
    std::pair<std::optional<int>, std::optional<int>> find_lane_positions_in_row(
        const cv::Mat& mask_row,
        const std::optional<int>& prev_left_x,
        const std::optional<int>& prev_right_x,
        const std::optional<int>& prev_center_x,
        int half_lane_width) const;
    std::tuple<std::vector<cv::Point>, std::vector<cv::Point>, std::vector<cv::Point>, int> extract_centerline(
        const cv::Mat& mask,
        const std::optional<int>& initial_left_x,
        const std::optional<int>& initial_right_x,
        const std::optional<int>& initial_center_x) const;
    std::vector<cv::Point> smooth_points(const std::vector<cv::Point>& points, bool use_poly) const;
    int single_lane_target_bias(const std::vector<cv::Point>& left_points_small,
                                const std::vector<cv::Point>& right_points_small,
                                int mask_width) const;
    std::string single_lane_turn_direction(const std::vector<cv::Point>& left_points_small,
                                           const std::vector<cv::Point>& right_points_small) const;
    int adaptive_target_index(const std::vector<cv::Point>& centers_small,
                              const std::vector<cv::Point>& left_points_small,
                              const std::vector<cv::Point>& right_points_small) const;

    static std::vector<cv::Point> scale_points(const std::vector<cv::Point>& points, double sx, double sy);
    static cv::Mat draw_polyline(const cv::Mat& image, const std::vector<cv::Point>& points, const cv::Scalar& color, int thickness);
};
