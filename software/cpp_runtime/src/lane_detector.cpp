#include "lane_detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>

namespace {
using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
}

LaneCenterlineDetectorCpp::LaneCenterlineDetectorCpp(const std::string& engine_path,
                                                     float threshold,
                                                     cv::Size input_size,
                                                     double crop_top_fraction,
                                                     double target_x_bias_px,
                                                     int num_sample_rows,
                                                     bool verbose)
    : threshold_(threshold),
      input_size_(input_size),
      crop_top_fraction_(crop_top_fraction),
      target_x_bias_px_(target_x_bias_px),
      verbose_(verbose),
      num_sample_rows_(num_sample_rows),
      kernel_open_(cv::Mat::ones(3, 3, CV_8U)),
      kernel_close_(cv::Mat::ones(5, 5, CV_8U)),
      trt_(engine_path, verbose) {
    sample_rows_.reserve(num_sample_rows_);
    for (int i = 0; i < num_sample_rows_; ++i) {
        const double alpha = num_sample_rows_ == 1 ? 0.0 : static_cast<double>(i) / (num_sample_rows_ - 1);
        const double frac = 0.80 + alpha * (0.45 - 0.80);
        sample_rows_.push_back(std::clamp(static_cast<int>(std::round(input_size_.height * frac)), 0, input_size_.height - 1));
    }
}

void LaneCenterlineDetectorCpp::reset_tracking() {
    prev_target_x_.reset();
    prev_target_y_.reset();
    prev_frame_left_x_.reset();
    prev_frame_right_x_.reset();
}

std::vector<float> LaneCenterlineDetectorCpp::prepare_input_tensor(const cv::Mat& frame_crop) const {
    cv::Mat image;
    cv::resize(frame_crop, image, input_size_, 0.0, 0.0, cv::INTER_LINEAR);
    image = apply_clahe_bgr(image);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32F, 1.0 / 255.0);

    std::vector<float> tensor(3 * input_size_.width * input_size_.height);
    const int stride = input_size_.width * input_size_.height;
    for (int y = 0; y < input_size_.height; ++y) {
        for (int x = 0; x < input_size_.width; ++x) {
            const auto px = image.at<cv::Vec3f>(y, x);
            tensor[y * input_size_.width + x] = px[0];
            tensor[stride + y * input_size_.width + x] = px[1];
            tensor[2 * stride + y * input_size_.width + x] = px[2];
        }
    }
    return tensor;
}

cv::Mat LaneCenterlineDetectorCpp::apply_clahe_bgr(const cv::Mat& image) const {
    cv::Mat lab;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> channels;
    cv::split(lab, channels);

    auto clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(channels[0], channels[0]);

    cv::merge(channels, lab);
    cv::Mat out;
    cv::cvtColor(lab, out, cv::COLOR_Lab2BGR);
    return out;
}

std::pair<cv::Mat, cv::Mat> LaneCenterlineDetectorCpp::predict_mask(const cv::Mat& frame_crop) {
    const auto output = trt_.infer(prepare_input_tensor(frame_crop));
    cv::Mat raw_mask(input_size_.height, input_size_.width, CV_8U);
    cv::Mat mask(input_size_.height, input_size_.width, CV_8U);
    for (int y = 0; y < input_size_.height; ++y) {
        for (int x = 0; x < input_size_.width; ++x) {
            const int idx = y * input_size_.width + x;
            const float p = 1.0f / (1.0f + std::exp(-output[idx]));
            raw_mask.at<unsigned char>(y, x) = static_cast<unsigned char>(std::round(p * 255.0f));
            mask.at<unsigned char>(y, x) = p > threshold_ ? 255 : 0;
        }
    }
    return {raw_mask, mask};
}

cv::Mat LaneCenterlineDetectorCpp::largest_components(const cv::Mat& mask) const {
    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    std::vector<std::pair<int, int>> areas;
    for (int i = 1; i < n; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= min_component_area_) areas.emplace_back(area, i);
    }
    std::sort(areas.begin(), areas.end(), std::greater<>());
    cv::Mat cleaned = cv::Mat::zeros(mask.size(), CV_8U);
    for (size_t i = 0; i < std::min<size_t>(4, areas.size()); ++i) {
        cleaned.setTo(255, labels == areas[i].second);
    }
    return cleaned;
}

cv::Mat LaneCenterlineDetectorCpp::postprocess_mask(const cv::Mat& mask) const {
    cv::Mat tmp;
    cv::morphologyEx(mask, tmp, cv::MORPH_OPEN, kernel_open_);
    cv::morphologyEx(tmp, tmp, cv::MORPH_CLOSE, kernel_close_);
    return largest_components(tmp);
}

std::vector<LaneCenterlineDetectorCpp::Blob> LaneCenterlineDetectorCpp::get_row_blobs(const cv::Mat& mask_row) const {
    std::vector<Blob> blobs;
    int start = -1;
    int prev = -1;
    for (int x = 0; x < mask_row.cols; ++x) {
        if (mask_row.at<unsigned char>(0, x) == 0) continue;
        if (start < 0) start = prev = x;
        else if (x == prev + 1) prev = x;
        else {
            const int width = prev - start + 1;
            if (width >= min_blob_width_) blobs.push_back({start, prev, (start + prev) / 2, width});
            start = prev = x;
        }
    }
    if (start >= 0) {
        const int width = prev - start + 1;
        if (width >= min_blob_width_) blobs.push_back({start, prev, (start + prev) / 2, width});
    }
    return blobs;
}

std::optional<LaneCenterlineDetectorCpp::Blob> LaneCenterlineDetectorCpp::choose_blob_near_target(
    const std::vector<Blob>& blobs,
    int target_x,
    int margin) const {
    std::optional<Blob> best;
    int best_dist = 0;
    int best_width = 0;

    for (const auto& blob : blobs) {
        const int dist = std::abs(blob.center - target_x);
        if (dist > margin) continue;
        if (!best || dist < best_dist || (dist == best_dist && blob.width > best_width)) {
            best = blob;
            best_dist = dist;
            best_width = blob.width;
        }
    }

    return best;
}

std::optional<LaneCenterlineDetectorCpp::Blob> LaneCenterlineDetectorCpp::choose_extreme_blob(
    const std::vector<Blob>& blobs,
    const std::string& side,
    int center_x) const {
    std::optional<Blob> best;
    for (const auto& blob : blobs) {
        if (side == "left") {
            if (blob.center >= center_x) continue;
            if (!best || blob.center > best->center) best = blob;
        } else {
            if (blob.center < center_x) continue;
            if (!best || blob.center < best->center) best = blob;
        }
    }
    return best;
}

std::string LaneCenterlineDetectorCpp::assign_single_blob_side(
    const Blob& blob,
    const std::optional<int>& prev_left_x,
    const std::optional<int>& prev_right_x,
    int img_center_x) const {
    if (prev_left_x && prev_right_x) {
        return std::abs(blob.center - *prev_left_x) <= std::abs(blob.center - *prev_right_x) ? "left" : "right";
    }
    if (prev_left_x) return "left";
    if (prev_right_x) return "right";
    return blob.center < img_center_x ? "left" : "right";
}

bool LaneCenterlineDetectorCpp::is_candidate_plausible(
    const std::optional<int>& candidate_x,
    const std::optional<int>& prev_x,
    int max_jump) const {
    if (!candidate_x || !prev_x) return true;
    return std::abs(*candidate_x - *prev_x) <= max_jump;
}

std::pair<std::optional<int>, std::optional<int>> LaneCenterlineDetectorCpp::find_lane_positions_in_row(
    const cv::Mat& mask_row,
    const std::optional<int>& prev_left_x,
    const std::optional<int>& prev_right_x,
    const std::optional<int>& prev_center_x,
    int half_lane_width) const {
    const int w = mask_row.cols;
    const int img_center_x = w / 2;
    const int search_margin = static_cast<int>(w * search_margin_frac_);
    const int max_row_jump = static_cast<int>(w * max_row_jump_frac_);
    const auto blobs = get_row_blobs(mask_row);
    if (blobs.empty()) return {std::nullopt, std::nullopt};

    std::optional<int> left_x;
    std::optional<int> right_x;

    if (blobs.size() == 1) {
        const auto side = assign_single_blob_side(blobs.front(), prev_left_x, prev_right_x, img_center_x);
        if (side == "left") left_x = blobs.front().center;
        else right_x = blobs.front().center;
        return {left_x, right_x};
    }

    std::optional<Blob> left_blob;
    std::optional<Blob> right_blob;

    if (prev_left_x) left_blob = choose_blob_near_target(blobs, *prev_left_x, search_margin);
    if (prev_right_x) right_blob = choose_blob_near_target(blobs, *prev_right_x, search_margin);

    if (left_blob && right_blob && left_blob->center == right_blob->center) {
        const int split_center = prev_center_x.value_or(img_center_x);
        if (left_blob->center < split_center) right_blob.reset();
        else left_blob.reset();
    }

    const int split_center = prev_center_x.value_or(img_center_x);
    if (!left_blob) left_blob = choose_extreme_blob(blobs, "left", split_center);
    if (!right_blob) right_blob = choose_extreme_blob(blobs, "right", split_center);

    left_x = left_blob ? std::optional<int>(left_blob->center) : std::nullopt;
    right_x = right_blob ? std::optional<int>(right_blob->center) : std::nullopt;

    if (!is_candidate_plausible(left_x, prev_left_x, max_row_jump)) left_x.reset();
    if (!is_candidate_plausible(right_x, prev_right_x, max_row_jump)) right_x.reset();

    if (!left_x && prev_left_x) {
        left_blob = choose_blob_near_target(blobs, *prev_left_x, max_row_jump);
        if (left_blob) left_x = left_blob->center;
    }
    if (!right_x && prev_right_x) {
        right_blob = choose_blob_near_target(blobs, *prev_right_x, max_row_jump);
        if (right_blob) right_x = right_blob->center;
    }

    if (left_x && right_x && *right_x <= *left_x) {
        if (prev_center_x) {
            const int expected_left = *prev_center_x - half_lane_width;
            const int expected_right = *prev_center_x + half_lane_width;
            if (std::abs(*left_x - expected_left) < std::abs(*right_x - expected_right)) right_x.reset();
            else left_x.reset();
        } else {
            left_x.reset();
            right_x.reset();
        }
    }

    if (left_x && right_x) {
        const int lane_width = *right_x - *left_x;
        const int min_lane_width = static_cast<int>(w * min_lane_width_frac_);
        const int max_lane_width = static_cast<int>(w * max_lane_width_frac_);
        if (lane_width < min_lane_width || lane_width > max_lane_width) {
            if (prev_center_x) {
                const int expected_left = *prev_center_x - half_lane_width;
                const int expected_right = *prev_center_x + half_lane_width;
                if (std::abs(*left_x - expected_left) <= std::abs(*right_x - expected_right)) right_x.reset();
                else left_x.reset();
            } else {
                left_x.reset();
                right_x.reset();
            }
        }
    }

    return {left_x, right_x};
}

std::vector<cv::Point> LaneCenterlineDetectorCpp::smooth_points(const std::vector<cv::Point>& points, bool use_poly) const {
    if (static_cast<int>(points.size()) < 3 || point_smooth_window_ < 3) return points;

    if (use_poly && points.size() >= 3) {
        cv::Mat a(static_cast<int>(points.size()), 3, CV_64F);
        cv::Mat b(static_cast<int>(points.size()), 1, CV_64F);
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            const double y = static_cast<double>(points[i].y);
            a.at<double>(i, 0) = y * y;
            a.at<double>(i, 1) = y;
            a.at<double>(i, 2) = 1.0;
            b.at<double>(i, 0) = static_cast<double>(points[i].x);
        }

        cv::Mat coeffs;
        if (cv::solve(a, b, coeffs, cv::DECOMP_SVD)) {
            std::vector<cv::Point> out;
            out.reserve(points.size());
            for (const auto& p : points) {
                const double y = static_cast<double>(p.y);
                const double x = coeffs.at<double>(0, 0) * y * y + coeffs.at<double>(1, 0) * y + coeffs.at<double>(2, 0);
                out.emplace_back(static_cast<int>(std::round(x)), p.y);
            }
            return out;
        }
    }

    const int pad = point_smooth_window_ / 2;
    std::vector<cv::Point> out;
    out.reserve(points.size());
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        double sx = 0.0;
        for (int k = -pad; k <= pad; ++k) {
            const int idx = std::clamp(i + k, 0, static_cast<int>(points.size()) - 1);
            sx += points[idx].x;
        }
        out.emplace_back(static_cast<int>(std::round(sx / point_smooth_window_)), points[i].y);
    }
    return out;
}

int LaneCenterlineDetectorCpp::single_lane_target_bias(
    const std::vector<cv::Point>& left_points_small,
    const std::vector<cv::Point>& right_points_small,
    int mask_width) const {
    if (left_points_small.size() >= 2 && right_points_small.empty()) {
        const int bottom_x = left_points_small.front().x;
        const int top_x = left_points_small.back().x;
        const int curve_dx = top_x - bottom_x;
        if (curve_dx > single_lane_curve_trigger_px_) {
            const int raw_bias = static_cast<int>(std::round(curve_dx * single_lane_target_bias_gain_));
            const int max_bias = static_cast<int>(std::round(mask_width * single_lane_target_bias_max_frac_));
            return std::clamp(raw_bias, 0, max_bias);
        }
    }

    if (right_points_small.size() >= 2 && left_points_small.empty()) {
        const int bottom_x = right_points_small.front().x;
        const int top_x = right_points_small.back().x;
        const int curve_dx = top_x - bottom_x;
        if (curve_dx < -single_lane_curve_trigger_px_) {
            const int raw_bias = static_cast<int>(std::round((-curve_dx) * single_lane_target_bias_gain_));
            const int max_bias = static_cast<int>(std::round(mask_width * single_lane_target_bias_max_frac_));
            return -std::clamp(raw_bias, 0, max_bias);
        }
    }

    return 0;
}

std::string LaneCenterlineDetectorCpp::single_lane_turn_direction(
    const std::vector<cv::Point>& left_points_small,
    const std::vector<cv::Point>& right_points_small) const {
    if (!right_points_small.empty() && left_points_small.empty()) return "left";
    if (!left_points_small.empty() && right_points_small.empty()) return "right";
    return "";
}

int LaneCenterlineDetectorCpp::adaptive_target_index(const std::vector<cv::Point>& centers_small,
                                                     const std::vector<cv::Point>& left_points_small,
                                                     const std::vector<cv::Point>& right_points_small) const {
    if (centers_small.empty()) return 0;

    int min_x = centers_small.front().x;
    int max_x = centers_small.front().x;
    for (const auto& p : centers_small) {
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
    }

    const double center_span = static_cast<double>(max_x - min_x);
    const double turn_ratio = center_span / std::max(static_cast<double>(input_size_.width), 1.0);
    const double turn_strength = std::min(1.0, turn_ratio / std::max(lookahead_turn_trigger_frac_, 1e-6));
    const bool single_visible = left_points_small.empty() ^ right_points_small.empty();
    const double far_frac = single_visible ? single_lane_target_index_far_frac_ : target_index_far_frac_;
    const double near_frac = single_visible ? single_lane_target_index_near_frac_ : target_index_near_frac_;
    const double frac = far_frac * (1.0 - turn_strength) + near_frac * turn_strength;

    return std::clamp(static_cast<int>(centers_small.size() * frac), 0, static_cast<int>(centers_small.size()) - 1);
}

std::tuple<std::vector<cv::Point>, std::vector<cv::Point>, std::vector<cv::Point>, int>
LaneCenterlineDetectorCpp::extract_centerline(const cv::Mat& mask,
                                              const std::optional<int>& initial_left_x,
                                              const std::optional<int>& initial_right_x,
                                              const std::optional<int>& initial_center_x) const {
    const int w = mask.cols;
    int half_lane_width = static_cast<int>(w * default_half_lane_width_frac_);
    auto prev_left_x = initial_left_x;
    auto prev_right_x = initial_right_x;
    auto prev_center_x = initial_center_x ? initial_center_x : std::optional<int>(w / 2);

    std::vector<cv::Point> centers, left_points, right_points;
    for (int y : sample_rows_) {
        auto [left_x, right_x] = find_lane_positions_in_row(mask.row(y), prev_left_x, prev_right_x, prev_center_x, half_lane_width);
        int center_x = prev_center_x.value_or(w / 2);
        if (left_x && right_x && *right_x > *left_x) {
            const int lane_width = *right_x - *left_x;
            if (lane_width >= static_cast<int>(w * min_lane_width_frac_) &&
                lane_width <= static_cast<int>(w * max_lane_width_frac_)) {
                half_lane_width = static_cast<int>((1.0 - lane_width_smoothing_) * half_lane_width + lane_width_smoothing_ * (lane_width / 2));
                center_x = (*left_x + *right_x) / 2;
            }
        } else if (left_x) {
            center_x = *left_x + half_lane_width;
        } else if (right_x) {
            center_x = *right_x - half_lane_width;
        }

        if (prev_center_x) {
            if (!left_x && right_x) {
                const int right_dx = prev_right_x ? *right_x - *prev_right_x : 0;
                if (right_dx < -single_lane_curve_trigger_px_) center_x = std::min(center_x, *prev_center_x);
                else if (right_dx > single_lane_curve_trigger_px_) center_x = std::max(center_x, *prev_center_x);
            } else if (!right_x && left_x) {
                const int left_dx = prev_left_x ? *left_x - *prev_left_x : 0;
                if (left_dx > single_lane_curve_trigger_px_) center_x = std::max(center_x, *prev_center_x);
                else if (left_dx < -single_lane_curve_trigger_px_) center_x = std::min(center_x, *prev_center_x);
            }
        }

        if (prev_center_x) {
            center_x = static_cast<int>((1.0 - center_smoothing_) * *prev_center_x + center_smoothing_ * center_x);
        }
        center_x = std::clamp(center_x, 0, w - 1);

        centers.emplace_back(center_x, y);
        if (left_x) {
            left_points.emplace_back(*left_x, y);
            prev_left_x = left_x;
        }
        if (right_x) {
            right_points.emplace_back(*right_x, y);
            prev_right_x = right_x;
        }
        prev_center_x = center_x;
    }

    return {smooth_points(centers, true), smooth_points(left_points, false), smooth_points(right_points, false), half_lane_width};
}

std::vector<cv::Point> LaneCenterlineDetectorCpp::scale_points(const std::vector<cv::Point>& points, double sx, double sy) {
    std::vector<cv::Point> out;
    out.reserve(points.size());
    for (const auto& p : points) {
        out.emplace_back(static_cast<int>(std::round(p.x * sx)), static_cast<int>(std::round(p.y * sy)));
    }
    return out;
}

cv::Mat LaneCenterlineDetectorCpp::draw_polyline(const cv::Mat& image,
                                                 const std::vector<cv::Point>& points,
                                                 const cv::Scalar& color,
                                                 int thickness) {
    cv::Mat out = image.clone();
    if (points.size() >= 2) cv::polylines(out, points, false, color, thickness);
    else if (points.size() == 1) cv::circle(out, points.front(), 4, color, cv::FILLED);
    return out;
}

DetectionResult LaneCenterlineDetectorCpp::process_frame(const cv::Mat& frame, bool draw_debug, bool generate_visuals) {
    (void)draw_debug;
    DetectionResult result;
    const auto t0 = Clock::now();

    const int cut = static_cast<int>(frame.rows * crop_top_fraction_);
    result.cropped_frame = frame.rowRange(cut, frame.rows).clone();
    result.timings.crop_ms = elapsed_ms(t0);

    const auto t1 = Clock::now();
    auto [raw_mask, mask_small] = predict_mask(result.cropped_frame);
    result.timings.predict_ms = elapsed_ms(t1);

    const auto t2 = Clock::now();
    mask_small = postprocess_mask(mask_small);
    result.timings.postprocess_ms = elapsed_ms(t2);

    const auto t3 = Clock::now();
    std::optional<int> prev_center_small;
    if (prev_frame_left_x_ && prev_frame_right_x_) prev_center_small = (*prev_frame_left_x_ + *prev_frame_right_x_) / 2;
    auto [centers_small, left_small, right_small, half_lane_width_small] =
        extract_centerline(mask_small, prev_frame_left_x_, prev_frame_right_x_, prev_center_small);
    result.timings.centerline_ms = elapsed_ms(t3);

    const auto t4 = Clock::now();
    const double sx = static_cast<double>(result.cropped_frame.cols) / mask_small.cols;
    const double sy = static_cast<double>(result.cropped_frame.rows) / mask_small.rows;
    result.centers = scale_points(centers_small, sx, sy);
    result.left_points = scale_points(left_small, sx, sy);
    result.right_points = scale_points(right_small, sx, sy);
    cv::resize(mask_small, result.mask, result.cropped_frame.size(), 0.0, 0.0, cv::INTER_NEAREST);
    cv::resize(raw_mask, result.raw_mask_vis, result.cropped_frame.size(), 0.0, 0.0, cv::INTER_LINEAR);

    const int img_center_x = result.cropped_frame.cols / 2;
    if (!centers_small.empty()) {
        const int idx = adaptive_target_index(centers_small, left_small, right_small);
        cv::Point target(
            static_cast<int>(std::round((centers_small[idx].x + single_lane_target_bias(left_small, right_small, mask_small.cols)) * sx + target_x_bias_px_)),
            static_cast<int>(std::round(centers_small[idx].y * sy))
        );
        target.x = std::clamp(target.x, 0, result.cropped_frame.cols - 1);
        if (!prev_target_x_) {
            prev_target_x_ = target.x;
            prev_target_y_ = target.y;
        } else {
            target.x = static_cast<int>(target_alpha_ * target.x + (1.0 - target_alpha_) * *prev_target_x_);
            target.y = static_cast<int>(target_alpha_ * target.y + (1.0 - target_alpha_) * *prev_target_y_);
            prev_target_x_ = target.x;
            prev_target_y_ = target.y;
        }
        result.target_point = target;
        result.steering_error = target.x - img_center_x;
    }

    prev_frame_left_x_ = !left_small.empty() ? std::optional<int>(left_small.front().x) : std::nullopt;
    prev_frame_right_x_ = !right_small.empty() ? std::optional<int>(right_small.front().x) : std::nullopt;
    result.timings.scale_and_mask_ms = elapsed_ms(t4);
    result.timings.total_ms = result.timings.crop_ms + result.timings.predict_ms + result.timings.postprocess_ms +
                              result.timings.centerline_ms + result.timings.scale_and_mask_ms;

    result.lane_state.left_count = static_cast<int>(left_small.size());
    result.lane_state.right_count = static_cast<int>(right_small.size());
    result.lane_state.both_visible = !left_small.empty() && !right_small.empty();
    result.lane_state.single_visible = left_small.empty() ^ right_small.empty();
    result.lane_state.single_lane_turn_direction = single_lane_turn_direction(left_small, right_small);
    result.lane_state.inner_lane_missing_turn = !result.lane_state.single_lane_turn_direction.empty();

    if (generate_visuals) {
        cv::Mat overlay = result.cropped_frame.clone();
        overlay.setTo(cv::Scalar(0, 255, 0), result.mask);
        overlay = draw_polyline(overlay, result.left_points, cv::Scalar(255, 0, 255), 2);
        overlay = draw_polyline(overlay, result.right_points, cv::Scalar(0, 165, 255), 2);
        overlay = draw_polyline(overlay, result.centers, cv::Scalar(255, 0, 0), 3);
        cv::line(overlay, cv::Point(img_center_x, 0), cv::Point(img_center_x, overlay.rows), cv::Scalar(255, 255, 0), 1);
        if (result.target_point) cv::circle(overlay, *result.target_point, 7, cv::Scalar(0, 0, 255), cv::FILLED);
        result.overlay = overlay;
    }
    return result;
}
