from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple
import random

import cv2
import numpy as np


# =========================================================
# Data Structure
# =========================================================
@dataclass
class LaneDetection:
    centerline_xs: List[float]
    centerline_ys: List[float]
    confidence: float
    debug_frame: Optional[np.ndarray] = None
    debug_meta: Optional[dict] = None


# =========================================================
# Main Detector
# =========================================================
class LaneDetector:
    """
    Lane detector with two modes:

    1) Real image mode:
       - detect(frame=...) runs OpenCV preprocessing for RED tape lanes
       - returns a local centerline ahead of the car

    2) Simulated path mode:
       - if path_xs/path_ys are provided and frame is None,
         it returns a forward slice of the synthetic path
    """

    def __init__(
        self,
        path_xs: Optional[List[float]] = None,
        path_ys: Optional[List[float]] = None,
        horizon_m: float = 6.0,
        noise_std: float = 0.0,
        dropout_prob: float = 0.0,
        short_horizon_prob: float = 0.0,
        *,
        frame_width: int = 640,
        frame_height: int = 360,
        roi_y0: int = 200,
        min_points: int = 12,
        show_debug: bool = False,
        lane_width_m: float = 0.9,
        x_m_per_px: float = 0.03,
        y_m_per_px: float = 0.01,

        # Main red HSV thresholds
        red_h1_low: int = 0,
        red_h1_high: int = 26,
        red_h2_low: int = 142,
        red_h2_high: int = 179,
        red_s_low: int = 55,
        red_v_low: int = 55,

        # Dark/far red recovery thresholds
        dark_red_h1_low: int = 0,
        dark_red_h1_high: int = 30,
        dark_red_h2_low: int = 142,
        dark_red_h2_high: int = 179,
        dark_red_s_low: int = 34,
        dark_red_v_low: int = 30,
        dark_red_v_high: int = 150,

        # Red-dominance gating
        red_over_green_min: int = 22,
        red_over_blue_min: int = 22,
        red_over_maxgb_min: int = 16,
        min_red_channel: int = 42,

        lab_a_low: int = 134,
        min_red_ratio: float = 1.12,
        dark_red_neighbor_dilate: int = 7,
        upper_block_y_frac: float = 0.18,

        # Turn-aware ROI expansion
        enable_turn_roi: bool = True,
        turn_trigger_dx_px: float = 34.0,
        turn_release_dx_px: float = 20.0,
        turn_hold_frames: int = 5,
        turn_extra_top_y_frac: float = 0.20,
        turn_extra_inner_top_x_frac: float = 0.22,
        turn_extra_outer_bottom_x_frac: float = 0.16,

        # Legacy / compatibility params kept so existing main.py calls do not break
        pink_h_low: int = 145,
        pink_h_high: int = 179,
        pink_s_low: int = 40,
        pink_v_low: int = 55,
        shadow_red_h1_low: int = 0,
        shadow_red_h1_high: int = 25,
        shadow_red_h2_low: int = 145,
        shadow_red_h2_high: int = 179,
        shadow_red_s_low: int = 20,
        shadow_red_v_low: int = 25,
        canny_low: int = 50,
        canny_high: int = 150,
        use_component_filter: bool = True,
        min_component_area: int = 8,
        max_component_area_frac: float = 0.45,
        min_component_height_frac: float = 0.01,
        max_component_width_frac: float = 0.85,
        min_component_aspect_ratio: float = 1.15,

        # Outside-lane corner expansion on sharp turns
        enable_outside_corner_roi: bool = True,
        outside_corner_top_y_frac: float = 0.10,
        outside_corner_inner_x_frac: float = 0.62,
        outside_corner_mid_y_frac: float = 0.34,
        outside_corner_bottom_inset_frac: float = 0.18,

        # Line params
        hough_threshold: int = 12,
        hough_min_line_length: int = 12,
        hough_max_line_gap: int = 18,

        # Temporal / stability params
        temporal_alpha_fit: float = 0.72,
        temporal_alpha_center: float = 0.70,
        fit_jump_limit_px: float = 90.0,
        hold_last_good_frames: int = 2,

        single_side_temporal_alpha: float = 0.86,
        single_side_max_center_shift_px: float = 45.0,
        single_side_min_y_frac: float = 0.28,
        single_side_width_blend: float = 0.75,

        # Pair validation params
        min_pair_valid_fraction: float = 0.20,
        max_crossed_fraction: float = 0.40,
        max_width_std_px: float = 180.0,

        # Relaxed midpoint acceptance
        relaxed_pair_valid_fraction: float = 0.28,
    ):
        # Simulated-path support
        self.path_xs = path_xs
        self.path_ys = path_ys
        self.horizon_m = float(horizon_m)
        self.noise_std = float(noise_std)
        self.dropout_prob = float(dropout_prob)
        self.short_horizon_prob = float(short_horizon_prob)

        # Real detector settings
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.roi_y0 = int(roi_y0)
        self.min_points = int(min_points)
        self.show_debug = bool(show_debug)

        # Temporary image->local scaling approximation
        self.x_m_per_px = float(x_m_per_px)
        self.y_m_per_px = float(y_m_per_px)

        # Used if only one lane side is visible
        self.lane_width_m = float(lane_width_m)

        # Active preprocessing thresholds
        self.red_h1_low = int(red_h1_low)
        self.red_h1_high = int(red_h1_high)
        self.red_h2_low = int(red_h2_low)
        self.red_h2_high = int(red_h2_high)
        self.red_s_low = int(red_s_low)
        self.red_v_low = int(red_v_low)

        self.dark_red_h1_low = int(dark_red_h1_low)
        self.dark_red_h1_high = int(dark_red_h1_high)
        self.dark_red_h2_low = int(dark_red_h2_low)
        self.dark_red_h2_high = int(dark_red_h2_high)
        self.dark_red_s_low = int(dark_red_s_low)
        self.dark_red_v_low = int(dark_red_v_low)
        self.dark_red_v_high = int(dark_red_v_high)

        self.red_over_green_min = int(red_over_green_min)
        self.red_over_blue_min = int(red_over_blue_min)
        self.red_over_maxgb_min = int(red_over_maxgb_min)
        self.min_red_channel = int(min_red_channel)

        self.lab_a_low = int(lab_a_low)
        self.min_red_ratio = float(min_red_ratio)
        self.dark_red_neighbor_dilate = int(dark_red_neighbor_dilate)
        self.upper_block_y_frac = float(upper_block_y_frac)

        self.enable_turn_roi = bool(enable_turn_roi)
        self.turn_trigger_dx_px = float(turn_trigger_dx_px)
        self.turn_release_dx_px = float(turn_release_dx_px)
        self.turn_hold_frames = int(turn_hold_frames)
        self.turn_extra_top_y_frac = float(turn_extra_top_y_frac)
        self.turn_extra_inner_top_x_frac = float(turn_extra_inner_top_x_frac)
        self.turn_extra_outer_bottom_x_frac = float(turn_extra_outer_bottom_x_frac)

        # Legacy / compatibility params retained on self in case external code expects them
        self.pink_h_low = int(pink_h_low)
        self.pink_h_high = int(pink_h_high)
        self.pink_s_low = int(pink_s_low)
        self.pink_v_low = int(pink_v_low)
        self.shadow_red_h1_low = int(shadow_red_h1_low)
        self.shadow_red_h1_high = int(shadow_red_h1_high)
        self.shadow_red_h2_low = int(shadow_red_h2_low)
        self.shadow_red_h2_high = int(shadow_red_h2_high)
        self.shadow_red_s_low = int(shadow_red_s_low)
        self.shadow_red_v_low = int(shadow_red_v_low)
        self.canny_low = int(canny_low)
        self.canny_high = int(canny_high)
        self.use_component_filter = bool(use_component_filter)
        self.min_component_area = int(min_component_area)
        self.max_component_area_frac = float(max_component_area_frac)
        self.min_component_height_frac = float(min_component_height_frac)
        self.max_component_width_frac = float(max_component_width_frac)
        self.min_component_aspect_ratio = float(min_component_aspect_ratio)

        # Outside-lane corner expansion on sharp turns
        self.enable_outside_corner_roi = bool(enable_outside_corner_roi)
        self.outside_corner_top_y_frac = float(outside_corner_top_y_frac)
        self.outside_corner_inner_x_frac = float(outside_corner_inner_x_frac)
        self.outside_corner_mid_y_frac = float(outside_corner_mid_y_frac)
        self.outside_corner_bottom_inset_frac = float(outside_corner_bottom_inset_frac)

        # Hough params
        self.hough_threshold = int(hough_threshold)
        self.hough_min_line_length = int(hough_min_line_length)
        self.hough_max_line_gap = int(hough_max_line_gap)

        # Stability params
        self.temporal_alpha_fit = float(temporal_alpha_fit)
        self.temporal_alpha_center = float(temporal_alpha_center)
        self.fit_jump_limit_px = float(fit_jump_limit_px)
        self.hold_last_good_frames = int(hold_last_good_frames)

        self.single_side_temporal_alpha = float(single_side_temporal_alpha)
        self.single_side_max_center_shift_px = float(single_side_max_center_shift_px)
        self.single_side_min_y_frac = float(single_side_min_y_frac)
        self.single_side_width_blend = float(single_side_width_blend)

        # Pair validation params
        self.min_pair_valid_fraction = float(min_pair_valid_fraction)
        self.max_crossed_fraction = float(max_crossed_fraction)
        self.max_width_std_px = float(max_width_std_px)

        # Temporal memory
        self.prev_left_fit: Optional[Tuple[float, float, float]] = None
        self.prev_right_fit: Optional[Tuple[float, float, float]] = None
        self.prev_center_pts_img: List[Tuple[float, float]] = []
        self.prev_good_center_pts_img: List[Tuple[float, float]] = []
        self.prev_fit_mode: str = "none"
        self.frames_since_good_pair: int = 999999

        self.prev_fallback_side: str = ""
        self.prev_fallback_side_frames: int = 0
        self.min_fallback_hold_frames: int = 6
        self.fallback_switch_margin: float = 0.10
        self.fallback_keep_bias: float = 0.08

        self.prev_lane_width_profile: Optional[np.ndarray] = None

        self.relaxed_pair_valid_fraction = float(relaxed_pair_valid_fraction)

        self.turn_mode: str = "straight"
        self.turn_mode_hold_count: int = 0

        self.single_side_entry_delay_frames: int = 1
        self.bad_pair_run_length: int = 0

        self.turn_debug_right_dx: float = 0.0
        self.turn_debug_left_dx: float = 0.0
        self.turn_debug_source: str = "none"
        self.turn_debug_center_dx: float = 0.0

    # =========================================================
    # Public API
    # =========================================================
    def detect(self, frame=None, vehicle_x: float = 0.0) -> LaneDetection:
        if frame is not None:
            try:
                return self._detect_from_frame(frame)
            except Exception as e:
                print(f"[LaneDetector ERROR] {type(e).__name__}: {e}")
                return self._empty_detection(mode="real", error=str(e))

        return self._detect_simulated(vehicle_x=vehicle_x)

    # =========================================================
    # Small Reusable Helpers
    # =========================================================
    def _empty_detection(self, *, mode: str, error: str = "") -> LaneDetection:
        if mode == "real":
            meta = {
                "mode": "real",
                "error": error,
                "fit_mode": "none",
                "left_fit_ok": 0,
                "right_fit_ok": 0,
                "lane_seg_count": 0,
                "center_pt_count": 0,
            }
        else:
            meta = {
                "mode": "sim",
                "fit_mode": "none",
                "left_fit_ok": "",
                "right_fit_ok": "",
                "lane_seg_count": "",
                "center_pt_count": 0,
            }
            if error:
                meta["error"] = error

        return LaneDetection(
            centerline_xs=[],
            centerline_ys=[],
            confidence=0.0,
            debug_frame=None,
            debug_meta=meta,
        )

    def _make_dual_hsv_mask(
        self,
        hsv: np.ndarray,
        h1_low: int,
        h1_high: int,
        h2_low: int,
        h2_high: int,
        s_low: int,
        v_low: int,
        v_high: int = 255,
    ) -> np.ndarray:
        mask1 = cv2.inRange(
            hsv,
            np.array([h1_low, s_low, v_low], dtype=np.uint8),
            np.array([h1_high, 255, v_high], dtype=np.uint8),
        )
        mask2 = cv2.inRange(
            hsv,
            np.array([h2_low, s_low, v_low], dtype=np.uint8),
            np.array([h2_high, 255, v_high], dtype=np.uint8),
        )
        return cv2.bitwise_or(mask1, mask2)

    def _poly_mask(
        self,
        shape_ref: np.ndarray,
        points: List[Tuple[int, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.zeros_like(shape_ref)
        poly = np.array([points], dtype=np.int32)
        cv2.fillPoly(mask, poly, 255)
        return mask, poly

    def _apply_open_close(
        self,
        binary: np.ndarray,
        open_kernel: Tuple[int, int] = (3, 3),
        close_kernel: Tuple[int, int] = (5, 5),
    ) -> np.ndarray:
        out = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            np.ones(open_kernel, np.uint8),
        )
        out = cv2.morphologyEx(
            out,
            cv2.MORPH_CLOSE,
            np.ones(close_kernel, np.uint8),
        )
        return out

    def _base_lane_pair_debug(self, ys: np.ndarray) -> dict:
        return {
            "lane_pair_valid": 0,
            "lane_width_min_px": 0.0,
            "lane_width_max_px": 0.0,
            "lane_width_mean_px": 0.0,
            "lane_width_std_px": 0.0,
            "lane_width_reason": "not_checked",
            "pair_reject_fallback_side": "",
            "left_fit_quality": 0.0,
            "right_fit_quality": 0.0,
            "lane_pair_valid_rows": 0,
            "lane_pair_total_rows": int(len(ys)),
            "lane_crossed_rows": 0,
        }

    def _build_fit_points_for_draw(
        self,
        fit: Optional[Tuple[float, float, float]],
        ys: np.ndarray,
        roi_y0: int,
    ) -> Optional[np.ndarray]:
        if fit is None:
            return None

        pts = []
        for yy in ys:
            xx = self._poly2_x(*fit, yy)
            pts.append((int(xx), int(yy) + roi_y0))

        if len(pts) < 2:
            return None

        return np.array(pts, dtype=np.int32)

    def _draw_fit_polyline(
        self,
        vis: np.ndarray,
        fit: Optional[Tuple[float, float, float]],
        ys: np.ndarray,
        roi_y0: int,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
    ) -> None:
        pts = self._build_fit_points_for_draw(fit, ys, roi_y0)
        if pts is not None:
            cv2.polylines(vis, [pts], False, color, thickness)

    def _score_side_fit(
        self,
        fit: Optional[Tuple[float, float, float]],
        pts: List[Tuple[int, int]],
        ys: np.ndarray,
        roi_w: int,
        roi_h: int,
    ) -> float:
        return self._score_fit_quality(
            fit=fit,
            points_xy=pts,
            ys_eval=ys,
            roi_w=roi_w,
            roi_h=roi_h,
        )

    def _build_single_side_centerline(
        self,
        fit: Tuple[float, float, float],
        pts: List[Tuple[int, int]],
        ys: np.ndarray,
        roi_w: int,
        roi_h: int,
        side: str,
        lane_pair_debug: dict,
        mode_name: str,
    ) -> Tuple[List[Tuple[float, float]], str, dict]:
        lane_pair_debug[f"{side}_fit_quality"] = self._score_side_fit(
            fit=fit,
            pts=pts,
            ys=ys,
            roi_w=roi_w,
            roi_h=roi_h,
        )

        center_pts_img: List[Tuple[float, float]] = []

        # Primary path: reconstruct from previous centerline shape and previous side offset profile.
        center_pts_img = self._build_single_side_center_from_prev_shape(
            fit=fit,
            ys=ys,
            roi_h=roi_h,
            side=side,
        )

        # Fallback only if that previous-shape reconstruction is unavailable.
        if len(center_pts_img) < 3:
            self._append_center_from_single_side(
                center_pts_img=center_pts_img,
                fit=fit,
                ys=ys,
                roi_w=roi_w,
                roi_h=roi_h,
                side=side,
            )

        if len(center_pts_img) >= 3:
            xs = np.array([p[0] for p in center_pts_img], dtype=np.float64)
            ys_img = np.array([p[1] for p in center_pts_img], dtype=np.float64)
            xs = np.array(self._smooth_1d(xs.tolist(), window=9), dtype=np.float64)
            center_pts_img = [(float(x), float(y)) for x, y in zip(xs, ys_img)]

        center_pts_img = self._blend_single_side_center_with_previous(center_pts_img)

        return center_pts_img, mode_name, lane_pair_debug

    def _centerline_from_midpoints(
        self,
        mids: np.ndarray,
        ys: np.ndarray,
        valid_rows: np.ndarray,
        roi_w: int,
    ) -> List[Tuple[float, float]]:
        center_pts_img: List[Tuple[float, float]] = []

        ys_valid = ys[valid_rows]
        mids_valid = mids[valid_rows]
        mids_valid = np.array(self._smooth_1d(mids_valid.tolist(), window=7), dtype=np.float64)

        for xc, yy in zip(mids_valid, ys_valid):
            if -0.10 * roi_w <= xc <= 1.10 * roi_w:
                center_pts_img.append((float(xc), float(yy + self.roi_y0)))

        return center_pts_img

    def _merge_seg_points(
        self,
        segs: List[Tuple[int, int, int, int]],
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
        pts: List[Tuple[int, int]] = []
        lane_segs: List[Tuple[int, int, int, int]] = []

        for x1, y1, x2, y2 in segs:
            pts.append((x1, y1))
            pts.append((x2, y2))
            lane_segs.append((x1, y1, x2, y2))

        return pts, lane_segs

    # =========================================================
    # Simulated Mode
    # =========================================================
    def _detect_simulated(self, vehicle_x: float = 0.0) -> LaneDetection:
        if random.random() < self.dropout_prob:
            return self._empty_detection(mode="sim")

        if self.path_xs is None or self.path_ys is None:
            return self._empty_detection(mode="sim")

        xs: List[float] = []
        ys: List[float] = []

        horizon = self.horizon_m
        if random.random() < self.short_horizon_prob:
            horizon *= 0.4

        max_x = vehicle_x + horizon

        for x, y in zip(self.path_xs, self.path_ys):
            if vehicle_x <= x <= max_x:
                yy = y
                if self.noise_std > 0.0:
                    yy += random.gauss(0.0, self.noise_std)
                xs.append(float(x))
                ys.append(float(yy))

        if len(xs) < 2:
            return self._empty_detection(mode="sim")

        return LaneDetection(
            centerline_xs=xs,
            centerline_ys=ys,
            confidence=1.0 - self.dropout_prob,
            debug_frame=None,
            debug_meta={
                "mode": "sim",
                "fit_mode": "path",
                "left_fit_ok": "",
                "right_fit_ok": "",
                "lane_seg_count": "",
                "center_pt_count": len(xs),
            },
        )

    # =========================================================
    # Real Image Mode
    # =========================================================
    def _detect_from_frame(self, frame: np.ndarray) -> LaneDetection:
        frame = self._prepare_frame(frame)
        h, w = frame.shape[:2]

        roi, roi_h, roi_w = self._extract_roi(frame)
        preprocess = self._preprocess_roi(roi, roi_h, roi_w)

        hsv_vis = preprocess["hsv_vis"]
        red_mask_raw = preprocess["red_mask_raw"]
        red_mask_clean = preprocess["red_mask_clean"]
        lane_binary = preprocess["lane_binary"]

        red_mask_raw_full = preprocess["red_mask_raw_full"]
        red_mask_clean_full = preprocess["red_mask_clean_full"]
        lane_binary_full = preprocess["lane_binary_full"]
        active_search_mask = preprocess["active_search_mask"]

        roi_poly = preprocess["roi_poly"]
        hood_poly = preprocess["hood_poly"]
        left_mask = preprocess["left_mask"]
        right_mask = preprocess["right_mask"]
        left_poly = preprocess["left_poly"]
        right_poly = preprocess["right_poly"]
        
        left_extra_poly = preprocess["left_extra_poly"]
        right_extra_poly = preprocess["right_extra_poly"]

        left_corner_poly = preprocess["left_corner_poly"]
        right_corner_poly = preprocess["right_corner_poly"]

        turn_mode = preprocess["turn_mode"]

        left_pts, right_pts, lane_segs, seg_debug = self._extract_lane_points(
            binary_full=lane_binary_full,
            binary_search=lane_binary,
            left_mask=left_mask,
            right_mask=right_mask,
            roi_w=roi_w,
            roi_h=roi_h,
            turn_mode=turn_mode,
        )

        left_fit_raw = self._poly2_fit_x_of_y(left_pts)
        right_fit_raw = self._poly2_fit_x_of_y(right_pts)

        ys = self._build_sample_ys(roi_h)

        left_fit, left_stab_debug = self._stabilize_fit(
            new_fit=left_fit_raw,
            prev_fit=self.prev_left_fit,
            ys=ys,
            roi_h=roi_h,
            roi_w=roi_w,
            side_name="left",
        )
        right_fit, right_stab_debug = self._stabilize_fit(
            new_fit=right_fit_raw,
            prev_fit=self.prev_right_fit,
            ys=ys,
            roi_h=roi_h,
            roi_w=roi_w,
            side_name="right",
        )

        center_pts_img, fit_mode, lane_pair_debug = self._build_centerline_points(
            left_fit=left_fit,
            right_fit=right_fit,
            left_pts=left_pts,
            right_pts=right_pts,
            ys=ys,
            roi_w=roi_w,
            roi_h=roi_h,
        )

        center_pts_img, hold_debug = self._maybe_hold_last_good_centerline(
            center_pts_img=center_pts_img,
            fit_mode=fit_mode,
            lane_pair_valid=lane_pair_debug.get("lane_pair_valid", 0),
        )
        if hold_debug["hold_used"]:
            fit_mode = hold_debug["fit_mode_after_hold"]

        self.current_fit_mode_for_smoothing = fit_mode

        center_pts_img = self._smooth_centerline_img_points(
            center_pts_img=center_pts_img,
            prev_center_pts_img=self.prev_center_pts_img,
        )

        centerline_xs, centerline_ys = self._image_centerline_to_local(
            center_pts_img=center_pts_img,
            frame_height=h,
            frame_width=w,
        )

        if len(centerline_xs) > 1:
            pts_sorted = sorted(zip(centerline_xs, centerline_ys), key=lambda p: p[0])
            centerline_xs = [p[0] for p in pts_sorted]
            centerline_ys = [p[1] for p in pts_sorted]

        confidence = self._compute_confidence(
            left_fit=left_fit,
            right_fit=right_fit,
            center_pts=center_pts_img,
            lane_segs=lane_segs,
            fit_mode=fit_mode,
            lane_pair_valid=lane_pair_debug.get("lane_pair_valid", 0),
            pair_reason=str(lane_pair_debug.get("lane_width_reason", "unknown")),
            left_fit_quality=float(lane_pair_debug.get("left_fit_quality", 0.0)),
            right_fit_quality=float(lane_pair_debug.get("right_fit_quality", 0.0)),
        )

        if len(centerline_xs) < 2 or fit_mode == "pair_rejected":
            confidence = 0.0

        if lane_pair_debug.get("lane_pair_valid", 0) == 1 and len(center_pts_img) >= self.min_points:
            self.prev_good_center_pts_img = list(center_pts_img)
            self.frames_since_good_pair = 0
        else:
            self.frames_since_good_pair += 1

        if lane_pair_debug.get("lane_pair_valid", 0) == 1:
            self.bad_pair_run_length = 0
        else:
            self.bad_pair_run_length += 1

        if left_fit is not None and right_fit is not None:
            width_profile = self._compute_lane_width_profile(
                left_fit=left_fit,
                right_fit=right_fit,
                ys=ys,
                roi_h=roi_h,
            )

            if self.prev_lane_width_profile is None or len(self.prev_lane_width_profile) != len(width_profile):
                self.prev_lane_width_profile = width_profile
            else:
                self.prev_lane_width_profile = (
                    0.85 * self.prev_lane_width_profile + 0.15 * width_profile
                )

        self.prev_left_fit = left_fit
        self.prev_right_fit = right_fit
        self.prev_center_pts_img = list(center_pts_img)
        self.prev_fit_mode = fit_mode
        self.current_fit_mode_for_smoothing = fit_mode

        debug_meta = {
            "mode": "real",
            "fit_mode": fit_mode,
            "turn_mode": str(turn_mode),
            "left_fit_ok": int(left_fit is not None),
            "right_fit_ok": int(right_fit is not None),
            "left_pt_count": int(len(left_pts)),
            "right_pt_count": int(len(right_pts)),
            "lane_seg_count": int(len(lane_segs)),
            "center_pt_count": int(len(center_pts_img)),
            "roi_y0": int(self.roi_y0),
            "frame_width": int(w),
            "frame_height": int(h),
            "red_mask_raw_count": int(np.count_nonzero(red_mask_raw)),
            "red_mask_clean_count": int(np.count_nonzero(red_mask_clean)),
            "lane_binary_count": int(np.count_nonzero(lane_binary)),
            "red_mask_raw_full_count": int(np.count_nonzero(red_mask_raw_full)),
            "red_mask_clean_full_count": int(np.count_nonzero(red_mask_clean_full)),
            "lane_binary_full_count": int(np.count_nonzero(lane_binary_full)),
            "active_search_mask_count": int(np.count_nonzero(active_search_mask)),
            "red_h1_low": int(self.red_h1_low),
            "red_h1_high": int(self.red_h1_high),
            "red_h2_low": int(self.red_h2_low),
            "red_h2_high": int(self.red_h2_high),
            "red_s_low": int(self.red_s_low),
            "red_v_low": int(self.red_v_low),

            "turn_debug_source": str(self.turn_debug_source),
            "turn_debug_left_dx": float(self.turn_debug_left_dx),
            "turn_debug_right_dx": float(self.turn_debug_right_dx),
            "turn_debug_center_dx": float(self.turn_debug_center_dx),

            **seg_debug,
            **lane_pair_debug,
            **left_stab_debug,
            **right_stab_debug,
            **hold_debug,
        }

        debug_frame = self._build_debug_view(
            frame=frame,
            roi_y0=self.roi_y0,
            roi_poly=roi_poly,
            hood_poly=hood_poly,
            left_poly=left_poly,
            right_poly=right_poly,
            left_extra_poly=left_extra_poly,
            right_extra_poly=right_extra_poly,
            left_corner_poly=left_corner_poly,
            right_corner_poly=right_corner_poly,
            turn_mode=turn_mode,
            lane_segs=lane_segs,
            left_fit=left_fit,
            right_fit=right_fit,
            center_pts=center_pts_img,
            ys=ys,
            confidence=confidence,
            hsv_vis=hsv_vis,
            red_mask_raw=red_mask_raw,
            red_mask_clean=red_mask_clean,
            lane_binary=lane_binary,
        )

        if self.show_debug:
            self._show_debug_view(debug_frame)

        return LaneDetection(
            centerline_xs=centerline_xs,
            centerline_ys=centerline_ys,
            confidence=confidence,
            debug_frame=debug_frame,
            debug_meta=debug_meta,
        )

    # =========================================================
    # Pipeline Stages
    # =========================================================
    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            raise ValueError("LaneDetector._detect_from_frame received frame=None")

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        return cv2.resize(
            frame,
            (self.frame_width, self.frame_height),
            interpolation=cv2.INTER_AREA,
        )

    def _extract_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        h, _ = frame.shape[:2]
        y0 = min(max(0, self.roi_y0), h - 1)
        roi = frame[y0:h, :, :]
        roi_h, roi_w = roi.shape[:2]

        if roi_h < 20:
            raise ValueError("ROI height is too small; check roi_y0")

        return roi, roi_h, roi_w

    def _preprocess_roi(
        self,
        roi: np.ndarray,
        roi_h: int,
        roi_w: int,
    ) -> dict:
        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)

        b = blur[:, :, 0].astype(np.int16)
        g = blur[:, :, 1].astype(np.int16)
        r = blur[:, :, 2].astype(np.int16)
        a_chan = lab[:, :, 1].astype(np.int16)

        red_mask_main = self._make_dual_hsv_mask(
            hsv,
            self.red_h1_low,
            self.red_h1_high,
            self.red_h2_low,
            self.red_h2_high,
            self.red_s_low,
            self.red_v_low,
        )

        red_mask_dark = self._make_dual_hsv_mask(
            hsv,
            self.dark_red_h1_low,
            self.dark_red_h1_high,
            self.dark_red_h2_low,
            self.dark_red_h2_high,
            self.dark_red_s_low,
            self.dark_red_v_low,
            self.dark_red_v_high,
        )

        red_dominance_mask = (
            ((r - g) > self.red_over_green_min)
            & ((r - b) > self.red_over_blue_min)
            & ((r - np.maximum(g, b)) > self.red_over_maxgb_min)
            & (r > self.min_red_channel)
        ).astype(np.uint8) * 255

        max_gb = np.maximum(g, b).astype(np.float32)
        red_ratio = r.astype(np.float32) / np.maximum(max_gb, 1.0)
        red_ratio_mask = (red_ratio > self.min_red_ratio).astype(np.uint8) * 255

        lab_red_mask = (a_chan > self.lab_a_low).astype(np.uint8) * 255

        # Slightly softer masks used ONLY for dark-red recovery
        dark_red_ratio_mask = (red_ratio > (self.min_red_ratio - 0.06)).astype(np.uint8) * 255
        dark_lab_red_mask = (a_chan > (self.lab_a_low - 6)).astype(np.uint8) * 255

        confirmed_red = cv2.bitwise_and(red_mask_main, red_dominance_mask)
        confirmed_red = cv2.bitwise_and(confirmed_red, red_ratio_mask)
        confirmed_red = cv2.bitwise_and(confirmed_red, lab_red_mask)

        k = max(3, int(self.dark_red_neighbor_dilate))
        if k % 2 == 0:
            k += 1

        strong_red_dilated = cv2.dilate(
            confirmed_red,
            np.ones((k, k), np.uint8),
            iterations=1,
        )

        dark_red_gated = cv2.bitwise_and(red_mask_dark, red_dominance_mask)
        dark_red_gated = cv2.bitwise_and(dark_red_gated, dark_red_ratio_mask)
        dark_red_gated = cv2.bitwise_and(dark_red_gated, dark_lab_red_mask)
        dark_red_gated = cv2.bitwise_and(dark_red_gated, strong_red_dilated)

        red_mask_raw_full = cv2.bitwise_or(confirmed_red, dark_red_gated)

        # Block top of ROI globally
        upper_block_y = int(self.upper_block_y_frac * roi_h)
        if upper_block_y > 0:
            red_mask_raw_full[:upper_block_y, :] = 0

        # Hood mask is still global
        hood_mask, hood_poly = self._poly_mask(
            red_mask_raw_full,
            [
                (int(0.30 * roi_w), roi_h - 1),
                (int(0.70 * roi_w), roi_h - 1),
                (int(0.60 * roi_w), int(0.72 * roi_h)),
                (int(0.40 * roi_w), int(0.72 * roi_h)),
            ],
        )

        red_mask_raw_full = cv2.bitwise_and(
            red_mask_raw_full,
            cv2.bitwise_not(hood_mask),
        )

        # --------------------------------
        # Geometry masks are built separately
        # --------------------------------
        roi_mask, roi_poly = self._poly_mask(
            red_mask_raw_full,
            [
                (int(0.00 * roi_w), roi_h - 1),
                (int(1.00 * roi_w), roi_h - 1),
                (int(0.74 * roi_w), int(0.18 * roi_h)),
                (int(0.26 * roi_w), int(0.18 * roi_h)),
            ],
        )

        base_road_mask = cv2.bitwise_and(roi_mask, cv2.bitwise_not(hood_mask))

        turn_mode = self._estimate_turn_mode(roi_h=roi_h, roi_w=roi_w)

        left_extra_mask, right_extra_mask, left_extra_poly, right_extra_poly = self._build_turn_extra_masks(
            shape_ref=red_mask_raw_full,
            roi_h=roi_h,
            roi_w=roi_w,
        )

        left_corner_mask, right_corner_mask, left_corner_poly, right_corner_poly = self._build_outside_corner_masks(
            shape_ref=red_mask_raw_full,
            roi_h=roi_h,
            roi_w=roi_w,
        )

        active_extra_mask = np.zeros_like(base_road_mask)

        if turn_mode == "left":
            active_extra_mask = left_extra_mask.copy()

            if self.enable_outside_corner_roi:
                # Outside lane on a left turn is the right lane,
                # so add upper-right corner access.
                active_extra_mask = cv2.bitwise_or(active_extra_mask, right_corner_mask)

        elif turn_mode == "right":
            active_extra_mask = right_extra_mask.copy()

            if self.enable_outside_corner_roi:
                # Outside lane on a right turn is the left lane,
                # so add upper-left corner access.
                active_extra_mask = cv2.bitwise_or(active_extra_mask, left_corner_mask)

        active_extra_mask = cv2.bitwise_and(active_extra_mask, cv2.bitwise_not(hood_mask))

        # Search mask:
        # - straight: trapezoid only
        # - turn: trapezoid + side extension + outside corner assist
        active_search_mask = cv2.bitwise_or(base_road_mask, active_extra_mask)

        # Base side masks
        left_mask_base, left_poly = self._poly_mask(
            red_mask_raw_full,
            [
                (int(0.00 * roi_w), roi_h - 1),
                (int(0.53 * roi_w), roi_h - 1),
                (int(0.45 * roi_w), int(0.22 * roi_h)),
                (int(0.14 * roi_w), int(0.22 * roi_h)),
            ],
        )
        left_mask_base = cv2.bitwise_and(left_mask_base, active_search_mask)

        right_mask_base, right_poly = self._poly_mask(
            red_mask_raw_full,
            [
                (int(0.47 * roi_w), roi_h - 1),
                (int(1.00 * roi_w), roi_h - 1),
                (int(0.86 * roi_w), int(0.22 * roi_h)),
                (int(0.55 * roi_w), int(0.22 * roi_h)),
            ],
        )
        right_mask_base = cv2.bitwise_and(right_mask_base, active_search_mask)

        left_mask = left_mask_base.copy()
        right_mask = right_mask_base.copy()

        if turn_mode == "left":
            # Keep left mask only mildly expanded
            left_side_extra = cv2.bitwise_and(left_extra_mask, active_search_mask)
            left_side_extra = cv2.erode(left_side_extra, np.ones((5, 5), np.uint8), iterations=1)
            left_mask = cv2.bitwise_or(left_mask, left_side_extra)

            # Stronger expansion for the outside lane (right lane) into upper-right corner
            if self.enable_outside_corner_roi:
                right_corner_extra = cv2.bitwise_and(right_corner_mask, active_search_mask)
                right_corner_extra = cv2.erode(right_corner_extra, np.ones((3, 3), np.uint8), iterations=1)
                right_mask = cv2.bitwise_or(right_mask, right_corner_extra)

        elif turn_mode == "right":
            # Keep right mask only mildly expanded
            right_side_extra = cv2.bitwise_and(right_extra_mask, active_search_mask)
            right_side_extra = cv2.erode(right_side_extra, np.ones((5, 5), np.uint8), iterations=1)
            right_mask = cv2.bitwise_or(right_mask, right_side_extra)

            # Stronger expansion for the outside lane (left lane) into upper-left corner
            if self.enable_outside_corner_roi:
                left_corner_extra = cv2.bitwise_and(left_corner_mask, active_search_mask)
                left_corner_extra = cv2.erode(left_corner_extra, np.ones((3, 3), np.uint8), iterations=1)
                left_mask = cv2.bitwise_or(left_mask, left_corner_extra)

        # --------------------------------
        # Morphology on FULL ROI preprocessed red
        # --------------------------------
        red_mask_clean_full = self._apply_open_close(
            red_mask_raw_full,
            open_kernel=(3, 3),
            close_kernel=(7, 7),
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask_clean_full, 8)
        filtered = np.zeros_like(red_mask_clean_full)
        min_area = 20
        min_height = 8

        for i in range(1, num_labels):
            x, y, w_box, h_box, area = stats[i]
            if area >= min_area and h_box >= min_height:
                filtered[labels == i] = 255

        red_mask_clean_full = filtered
        lane_binary_full = red_mask_clean_full.copy()

        # For debug and default extraction view
        lane_binary_search = cv2.bitwise_and(lane_binary_full, active_search_mask)
        red_mask_raw_search = cv2.bitwise_and(red_mask_raw_full, active_search_mask)
        red_mask_clean_search = cv2.bitwise_and(red_mask_clean_full, active_search_mask)

        hsv_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return {
            "hsv_vis": hsv_vis,
            "red_mask_raw": red_mask_raw_search,
            "red_mask_clean": red_mask_clean_search,
            "lane_binary": lane_binary_search,
            "red_mask_raw_full": red_mask_raw_full,
            "red_mask_clean_full": red_mask_clean_full,
            "lane_binary_full": lane_binary_full,
            "active_search_mask": active_search_mask,
            "base_road_mask": base_road_mask,
            "roi_poly": roi_poly,
            "hood_poly": hood_poly,
            "left_mask": left_mask,
            "right_mask": right_mask,
            "left_poly": left_poly,
            "right_poly": right_poly,
            "left_extra_poly": left_extra_poly,
            "right_extra_poly": right_extra_poly,
            "left_corner_poly": left_corner_poly,
            "right_corner_poly": right_corner_poly,
            "turn_mode": turn_mode,
        }

    def _nominal_lane_width_px(self) -> float:
        nominal = self.lane_width_m / max(self.y_m_per_px, 1e-6)
        return float(np.clip(nominal, 60.0, 170.0))

    def _expected_lane_width_px(self, y: float, roi_h: int) -> float:
        nominal = self._nominal_lane_width_px()
        y_norm = float(np.clip(y / max(1.0, roi_h - 1), 0.0, 1.0))
        scale_top = 0.72
        scale_bottom = 1.45
        scale = (1.0 - y_norm) * scale_top + y_norm * scale_bottom
        return float(nominal * scale)

    def _width_bounds_px(self, y: float, roi_h: int) -> Tuple[float, float]:
        expected = self._expected_lane_width_px(y, roi_h)
        min_w = max(16.0, 0.36 * expected)
        max_w = min(0.99 * self.frame_width, 2.35 * expected)
        return float(min_w), float(max_w)

    def _estimate_turn_from_single_fit(
        self,
        fit: Optional[Tuple[float, float, float]],
        roi_h: int,
        side: str,
    ) -> Tuple[str, float]:
        if fit is None:
            return "straight", 0.0

        a, b, c = fit

        # Reject obviously over-curved fits
        if abs(a) > 0.0035:
            if side == "right":
                self.turn_debug_right_dx = 0.0
            else:
                self.turn_debug_left_dx = 0.0
            return "straight", 0.0

        ys = np.array([0.30 * roi_h, 0.45 * roi_h, 0.60 * roi_h, 0.75 * roi_h], dtype=np.float64)
        xs = np.array([self._poly2_x(a, b, c, yy) for yy in ys], dtype=np.float64)

        # Fit a straight trend line through sampled lane positions
        try:
            m, k = np.polyfit(ys, xs, 1)
        except Exception:
            return "straight", 0.0

        # Convert slope to a comparable "dx-like" signal across the sampled span
        dx = float(m * (ys[0] - ys[-1]))

        # Clamp absurd values
        if abs(dx) > 100.0:
            dx = 0.0

        if side == "right":
            self.turn_debug_right_dx = dx
            if dx <= -self.turn_trigger_dx_px:
                return "left", dx
            if dx <= -self.turn_release_dx_px:
                return "left_soft", dx

        elif side == "left":
            self.turn_debug_left_dx = dx
            if dx >= self.turn_trigger_dx_px:
                return "right", dx
            if dx >= self.turn_release_dx_px:
                return "right_soft", dx

        return "straight", dx
    
    def _estimate_turn_mode(
        self,
        roi_h: int,
        roi_w: int,
    ) -> str:
        if not self.enable_turn_roi:
            self.turn_mode = "straight"
            self.turn_mode_hold_count = 0
            self.turn_debug_source = "disabled"
            self.turn_debug_center_dx = 0.0
            return "straight"

        raw_mode = "straight"
        self.turn_debug_source = "none"
        self.turn_debug_center_dx = 0.0

        if not hasattr(self, "turn_consistency_count"):
            self.turn_consistency_count = 0

        right_mode, right_dx = self._estimate_turn_from_single_fit(
            fit=self.prev_right_fit,
            roi_h=roi_h,
            side="right",
        )
        left_mode, left_dx = self._estimate_turn_from_single_fit(
            fit=self.prev_left_fit,
            roi_h=roi_h,
            side="left",
        )

        MAX_REASONABLE_DX = 120.0
        if abs(right_dx) > MAX_REASONABLE_DX:
            right_mode = "straight"
        if abs(left_dx) > MAX_REASONABLE_DX:
            left_mode = "straight"

        if right_mode == "left" and self.prev_left_fit is not None:
            raw_mode = "left"
            self.turn_debug_source = "right_outside_lane"
        elif left_mode == "right" and self.prev_right_fit is not None:
            raw_mode = "right"
            self.turn_debug_source = "left_outside_lane"

        if raw_mode == "straight" and self.prev_left_fit is not None and self.prev_right_fit is not None:
            left_strength = max(0.0, left_dx)
            right_strength = max(0.0, -right_dx)

            BOTH_LANES_MARGIN = 12.0
            BOTH_LANES_TRIGGER = self.turn_trigger_dx_px + 2.0

            if right_strength >= BOTH_LANES_TRIGGER and right_strength >= left_strength + BOTH_LANES_MARGIN:
                raw_mode = "left"
                self.turn_debug_source = "both_lanes_right_stronger"
            elif left_strength >= BOTH_LANES_TRIGGER and left_strength >= right_strength + BOTH_LANES_MARGIN:
                raw_mode = "right"
                self.turn_debug_source = "both_lanes_left_stronger"

        if raw_mode == "straight" and len(self.prev_center_pts_img) >= 8:
            pts = list(self.prev_center_pts_img)

            xs = np.array([p[0] for p in pts], dtype=np.float64)
            ys_img = np.array([p[1] for p in pts], dtype=np.float64)

            y_low_thresh = np.percentile(ys_img, 75)
            y_high_thresh = np.percentile(ys_img, 25)

            bottom_x = float(np.mean(xs[ys_img >= y_low_thresh]))
            top_x = float(np.mean(xs[ys_img <= y_high_thresh]))
            dx_center = float(top_x - bottom_x)

            self.turn_debug_center_dx = dx_center

            if dx_center <= -self.turn_trigger_dx_px:
                raw_mode = "left"
                self.turn_debug_source = "centerline_fallback"
            elif dx_center >= self.turn_trigger_dx_px:
                raw_mode = "right"
                self.turn_debug_source = "centerline_fallback"
            else:
                raw_mode = "straight"

        if raw_mode in ("left", "right"):
            if raw_mode == self.turn_mode:
                self.turn_consistency_count = min(self.turn_consistency_count + 1, 10)
            else:
                self.turn_consistency_count += 1

            if self.turn_consistency_count < 2:
                raw_mode = "straight"
        else:
            self.turn_consistency_count = 0

        if raw_mode == self.turn_mode:
            self.turn_mode_hold_count = self.turn_hold_frames

        else:
            # Stronger hysteresis: once we are in a turn, do not leave it quickly.
            if raw_mode == "straight" and self.turn_mode in ("left", "right"):
                if self.turn_mode_hold_count > 0:
                    self.turn_mode_hold_count -= 1
                    raw_mode = self.turn_mode
                    if self.turn_debug_source == "none":
                        self.turn_debug_source = "hold_counter"
                else:
                    # Even after hold counter expires, require a stronger reason to exit.
                    # If centerline still suggests turning at all, stay in turn mode.
                    if abs(self.turn_debug_center_dx) >= self.turn_release_dx_px:
                        raw_mode = self.turn_mode
                        self.turn_debug_source = "release_guard"
                    else:
                        self.turn_mode = "straight"
                        self.turn_mode_hold_count = 0
            else:
                self.turn_mode = raw_mode
                self.turn_mode_hold_count = self.turn_hold_frames if raw_mode in ("left", "right") else 0

        return self.turn_mode

    def _build_turn_extra_masks(
        self,
        shape_ref: np.ndarray,
        roi_h: int,
        roi_w: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        top_y = int(self.turn_extra_top_y_frac * roi_h)
        inner_top_x = int(self.turn_extra_inner_top_x_frac * roi_w)
        outer_bottom_x = int(self.turn_extra_outer_bottom_x_frac * roi_w)

        left_extra_mask, left_extra_poly = self._poly_mask(
            shape_ref,
            [
                (0, roi_h - 1),
                (outer_bottom_x, roi_h - 1),
                (inner_top_x, top_y),
                (0, top_y),
            ],
        )

        right_extra_mask, right_extra_poly = self._poly_mask(
            shape_ref,
            [
                (roi_w - outer_bottom_x, roi_h - 1),
                (roi_w - 1, roi_h - 1),
                (roi_w - 1, top_y),
                (roi_w - inner_top_x, top_y),
            ],
        )

        return left_extra_mask, right_extra_mask, left_extra_poly, right_extra_poly

    def _build_outside_corner_masks(
        self,
        shape_ref: np.ndarray,
        roi_h: int,
        roi_w: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        top_y = int(0.20 * roi_h)
        mid_y = int(0.20 * roi_h)
        inner_x = int(1.00 * roi_w)
        bottom_inset = int(0.20 * roi_w)

        left_corner_mask, left_corner_poly = self._poly_mask(
            shape_ref,
            [
                (0, top_y),
                (inner_x, mid_y),
                (bottom_inset, roi_h - 1),
                (0, roi_h - 1),
            ],
        )
 
        right_corner_mask, right_corner_poly = self._poly_mask(
            shape_ref,
            [
                (roi_w - 1, top_y),
                (roi_w - 1 - inner_x, mid_y),
                (roi_w - 1 - bottom_inset, roi_h - 1),
                (roi_w - 1, roi_h - 1),
            ],
        )

        return left_corner_mask, right_corner_mask, left_corner_poly, right_corner_poly

    def _compute_lane_width_profile(
        self,
        left_fit: Optional[Tuple[float, float, float]],
        right_fit: Optional[Tuple[float, float, float]],
        ys: np.ndarray,
        roi_h: int,
    ) -> np.ndarray:
        widths = []

        for yy in ys:
            if left_fit is not None and right_fit is not None:
                xl = self._poly2_x(*left_fit, yy)
                xr = self._poly2_x(*right_fit, yy)
                width = xr - xl
                min_w, max_w = self._width_bounds_px(yy, roi_h)

                if np.isfinite(width) and min_w <= width <= max_w:
                    widths.append(float(width))
                else:
                    widths.append(float(self._expected_lane_width_px(yy, roi_h)))
            else:
                widths.append(float(self._expected_lane_width_px(yy, roi_h)))

        return np.array(widths, dtype=np.float64)

    def _get_single_side_width_profile(
        self,
        ys: np.ndarray,
        roi_h: int,
    ) -> np.ndarray:
        expected = np.array(
            [self._expected_lane_width_px(yy, roi_h) for yy in ys],
            dtype=np.float64,
        )

        if self.prev_lane_width_profile is None or len(self.prev_lane_width_profile) != len(ys):
            return expected

        return (
            self.single_side_width_blend * self.prev_lane_width_profile
            + (1.0 - self.single_side_width_blend) * expected
        )

    def _build_prev_side_offset_profile(
        self,
        ys: np.ndarray,
        side: str,
        fallback_width_profile: np.ndarray,
    ) -> np.ndarray:
        """
        Build a per-row center offset relative to the requested lane side
        using the previous stable centerline and previous side fit.

        For left side:
            offset = prev_center_x - prev_left_lane_x   (usually positive)
        For right side:
            offset = prev_right_lane_x - prev_center_x  (usually positive)

        If we do not have enough previous information, fall back to half-width.
        """
        if len(self.prev_center_pts_img) != len(ys):
            return 0.5 * fallback_width_profile

        side_fit = self.prev_left_fit if side == "left" else self.prev_right_fit
        if side_fit is None:
            return 0.5 * fallback_width_profile

        offsets: List[float] = []

        for i, yy in enumerate(ys):
            prev_center_x, _ = self.prev_center_pts_img[i]
            prev_lane_x = self._poly2_x(*side_fit, yy)

            if side == "left":
                off = prev_center_x - prev_lane_x
            else:
                off = prev_lane_x - prev_center_x

            if not np.isfinite(off):
                off = 0.5 * float(fallback_width_profile[i])

            offsets.append(float(off))

        offsets_arr = np.array(offsets, dtype=np.float64)

        # Smooth row-to-row noise
        offsets_arr = np.array(
            self._smooth_1d(offsets_arr.tolist(), window=9),
            dtype=np.float64,
        )

        # Keep offsets physically reasonable
        min_offsets = 0.35 * fallback_width_profile
        max_offsets = 0.75 * fallback_width_profile
        offsets_arr = np.clip(offsets_arr, min_offsets, max_offsets)

        return offsets_arr

    def _blend_single_side_center_with_previous(
        self,
        center_pts_img: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        if len(center_pts_img) < 3:
            return center_pts_img

        if len(self.prev_center_pts_img) != len(center_pts_img):
            return center_pts_img

        out = []
        prev_xs = [p[0] for p in self.prev_center_pts_img]
        prev_ys = [p[1] for p in self.prev_center_pts_img]

        for i, (xc, yc) in enumerate(center_pts_img):
            prev_x = prev_xs[i]
            prev_y = prev_ys[i]

            # Stronger hold to previous centerline in single-side mode
            blended_x = 0.90 * prev_x + 0.10 * xc

            dx = blended_x - prev_x

            # Clamp small per-row change to prevent visible snapping
            if i < len(center_pts_img) // 3:
                max_step = 8.0
            elif i < (2 * len(center_pts_img)) // 3:
                max_step = 12.0
            else:
                max_step = 16.0

            dx = float(np.clip(dx, -max_step, max_step))
            blended_x = prev_x + dx

            out.append((float(blended_x), float(prev_y)))

        return out

    def _build_single_side_center_from_prev_shape(
        self,
        fit: Tuple[float, float, float],
        ys: np.ndarray,
        roi_h: int,
        side: str,
    ) -> List[Tuple[float, float]]:
        if len(self.prev_center_pts_img) != len(ys):
            return []

        width_profile = self._get_single_side_width_profile(ys=ys, roi_h=roi_h)
        offset_profile = self._build_prev_side_offset_profile(
            ys=ys,
            side=side,
            fallback_width_profile=width_profile,
        )

        out: List[Tuple[float, float]] = []

        for i, yy in enumerate(ys):
            prev_x, prev_y = self.prev_center_pts_img[i]
            x_lane = self._poly2_x(*fit, yy)

            if side == "left":
                target_center_x = x_lane + float(offset_profile[i])
            else:
                target_center_x = x_lane - float(offset_profile[i])

            # Limit how much single-side mode can reshape the centerline.
            # Allow a little more motion near the bottom of the ROI.
            if yy < 0.45 * roi_h:
                max_shift = 8.0
            elif yy < 0.65 * roi_h:
                max_shift = 12.0
            else:
                max_shift = 18.0

            dx = target_center_x - prev_x
            dx = float(np.clip(dx, -max_shift, max_shift))
            blended_x = prev_x + dx

            out.append((float(blended_x), float(prev_y)))

        # Smooth the rebuilt x profile
        xs = np.array([p[0] for p in out], dtype=np.float64)
        ys_img = np.array([p[1] for p in out], dtype=np.float64)
        xs = np.array(self._smooth_1d(xs.tolist(), window=9), dtype=np.float64)

        return [(float(x), float(y)) for x, y in zip(xs, ys_img)]

    def _filter_segments_by_cluster(
        self,
        segs: List[Tuple[int, int, int, int]],
        roi_w: int,
    ) -> List[Tuple[int, int, int, int]]:
        if len(segs) <= 6:
            return segs

        x_mids = np.array([0.5 * (x1 + x2) for x1, _, x2, _ in segs], dtype=np.float32)
        x_med = float(np.median(x_mids))
        tol = max(18.0, 0.12 * roi_w)

        filtered = []
        for seg in segs:
            x1, _, x2, _ = seg
            x_mid = 0.5 * (x1 + x2)
            if abs(x_mid - x_med) <= tol:
                filtered.append(seg)

        if len(filtered) < max(2, len(segs) // 3):
            return segs

        return filtered

    def _extract_lane_points_from_binary(
        self,
        binary: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
        roi_h: int,
        roi_w: int,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        left_pts: List[Tuple[int, int]] = []
        right_pts: List[Tuple[int, int]] = []

        ys = self._build_binary_sample_rows(roi_h)

        left_target_default = 0.30 * roi_w
        right_target_default = 0.70 * roi_w

        prev_left_row_x = None
        prev_right_row_x = None

        for y in ys:
            row = binary[y]
            y_norm = y / max(1.0, roi_h - 1)

            left_target = (
                self._poly2_x(*self.prev_left_fit, y)
                if self.prev_left_fit is not None
                else left_target_default
            )
            right_target = (
                self._poly2_x(*self.prev_right_fit, y)
                if self.prev_right_fit is not None
                else right_target_default
            )

            max_row_jump = 45.0 + 55.0 * y_norm

            left_candidates = np.where((row > 0) & (left_mask[y] > 0))[0]
            if left_candidates.size >= 3:
                target_for_left = (
                    left_target
                    if prev_left_row_x is None
                    else 0.7 * prev_left_row_x + 0.3 * left_target
                )
                x_left = self._choose_run_center(left_candidates, target_for_left)
                if x_left is not None and abs(x_left - target_for_left) <= max_row_jump:
                    left_pts.append((x_left, int(y)))
                    prev_left_row_x = x_left

            right_candidates = np.where((row > 0) & (right_mask[y] > 0))[0]
            if right_candidates.size >= 3:
                target_for_right = (
                    right_target
                    if prev_right_row_x is None
                    else 0.7 * prev_right_row_x + 0.3 * right_target
                )
                x_right = self._choose_run_center(right_candidates, target_for_right)
                if x_right is not None and abs(x_right - target_for_right) <= max_row_jump:
                    right_pts.append((x_right, int(y)))
                    prev_right_row_x = x_right

        return left_pts, right_pts
    
    def _get_segment_slope_limits(
        self,
        turn_mode: str,
    ) -> dict:
        """
        Return slope limits for left/right lane segment acceptance.

        Uses the existing signed-slope convention in image coordinates.
        In hard turns:
        - outside lane is allowed to get flatter
        - inside lane gets a steeper threshold baseline
        """
        limits = {
            "left_min": -1.73,
            "left_max": -0.4,
            "right_min": 0.4,
            "right_max": 1.73,
        }

        if turn_mode == "left":
            limits = {
                "left_min": -20.0,
                "left_max": -0.84,
                "right_min": 0.017,
                "right_max": 1.73,
            }

        elif turn_mode == "right":
            limits = {
                "left_min": -1.73,
                "left_max": -0.017,
                "right_min": 0.84,
                "right_max": 20.0,
            }

        return limits

    def _extract_lane_points(
        self,
        binary_full: np.ndarray,
        binary_search: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
        roi_w: int,
        roi_h: int,
        turn_mode: str,
    ) -> Tuple[
        List[Tuple[int, int]],
        List[Tuple[int, int]],
        List[Tuple[int, int, int, int]],
        dict,
    ]:
        lines = cv2.HoughLinesP(
            binary_search,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )

        left_lane_segs: List[Tuple[int, int, int, int]] = []
        right_lane_segs: List[Tuple[int, int, int, int]] = []

        slope_limits = self._get_segment_slope_limits(turn_mode=turn_mode)

        left_reject_mask_count = 0
        right_reject_mask_count = 0

        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                dx = x2 - x1
                dy = y2 - y1

                if dx == 0:
                    continue

                slope = dy / dx
                length = float(np.hypot(dx, dy))
                y_span = abs(dy)
                x_mid = int(round(0.5 * (x1 + x2)))
                y_mid = int(round(0.5 * (y1 + y2)))

                if length < 8:
                    continue
                if y_span < 2:
                    continue
                if y_mid < int(0.08 * roi_h):
                    continue
                if not (0 <= x_mid < roi_w and 0 <= y_mid < roi_h):
                    continue

                left_x_limit = 0.72
                right_x_limit = 0.28

                if turn_mode == "left":
                    left_x_limit = 0.78
                    right_x_limit = 0.22
                elif turn_mode == "right":
                    left_x_limit = 0.78
                    right_x_limit = 0.22

                # Base signed-slope rules
                left_slope_ok = slope_limits["left_min"] < slope < slope_limits["left_max"]
                right_slope_ok = slope_limits["right_min"] < slope < slope_limits["right_max"]

                # Inside-lane hard-turn rule:
                if turn_mode == "left":
                    # Left lane on hard left turn:
                    # allow -90..-30 and 40..90
                    left_slope_ok = (
                        (slope <= -0.577)   # about -30 deg cutoff
                        or (slope >= 0.84)  # about +40 deg cutoff
                    )

                elif turn_mode == "right":
                    # Right lane on hard right turn:
                    # mirrored equivalent: allow 30..90 and -90..-40
                    right_slope_ok = (
                        (slope >= 0.577)    # about +30 deg cutoff
                        or (slope <= -0.84) # about -40 deg cutoff
                    )

                left_allowed = (
                    left_slope_ok
                    and x_mid < int(left_x_limit * roi_w)
                )

                right_allowed = (
                    right_slope_ok
                    and x_mid > int(right_x_limit * roi_w)
                )

                if left_allowed:
                    if left_mask[y_mid, x_mid] != 0:
                        left_lane_segs.append((x1, y1, x2, y2))
                    else:
                        left_reject_mask_count += 1

                elif right_allowed:
                    if right_mask[y_mid, x_mid] != 0:
                        right_lane_segs.append((x1, y1, x2, y2))
                    else:
                        right_reject_mask_count += 1

        left_lane_segs = self._filter_segments_by_cluster(left_lane_segs, roi_w=roi_w)
        right_lane_segs = self._filter_segments_by_cluster(right_lane_segs, roi_w=roi_w)

        left_pts, left_vis_segs = self._merge_seg_points(left_lane_segs)
        right_pts, right_vis_segs = self._merge_seg_points(right_lane_segs)
        lane_segs = left_vis_segs + right_vis_segs

        debug_counts = {
            "hough_count": 0 if lines is None else int(len(lines)),
            "left_seg_count": int(len(left_lane_segs)),
            "right_seg_count": int(len(right_lane_segs)),
            "left_raw_pt_count": int(len(left_pts)),
            "right_raw_pt_count": int(len(right_pts)),
            "left_reject_mask_count": int(left_reject_mask_count),
            "right_reject_mask_count": int(right_reject_mask_count),
            "extract_turn_mode": str(turn_mode),
        }

        # Fallback row-sampling uses the full preprocessed binary,
        fb_left, fb_right = self._extract_lane_points_from_binary(
            binary=binary_full,
            left_mask=left_mask,
            right_mask=right_mask,
            roi_h=roi_h,
            roi_w=roi_w,
        )

        left_pts.extend(fb_left)
        right_pts.extend(fb_right)

        debug_counts["fallback_used"] = 1
        debug_counts["fallback_left_pts"] = int(len(fb_left))
        debug_counts["fallback_right_pts"] = int(len(fb_right))

        return left_pts, right_pts, lane_segs, debug_counts

    def _build_sample_ys(self, roi_h: int) -> np.ndarray:
        return np.linspace(int(0.18 * roi_h), roi_h - 1, 60)

    def _score_fit_quality(
        self,
        fit: Optional[Tuple[float, float, float]],
        points_xy: List[Tuple[int, int]],
        ys_eval: np.ndarray,
        roi_w: int,
        roi_h: int,
    ) -> float:
        if fit is None or len(points_xy) < 6:
            return 0.0

        a, b, c = fit

        xs = np.array([p[0] for p in points_xy], dtype=np.float64)
        ys = np.array([p[1] for p in points_xy], dtype=np.float64)

        x_pred = a * ys * ys + b * ys + c
        residuals = np.abs(xs - x_pred)
        mean_res = float(np.mean(residuals))
        med_res = float(np.median(residuals))

        y_span = float(np.max(ys) - np.min(ys)) if len(ys) > 1 else 0.0
        y_span_score = min(1.0, y_span / max(1.0, 0.60 * roi_h))

        unique_y_count = len(np.unique(ys.astype(int)))
        row_score = min(1.0, unique_y_count / 18.0)

        x_eval = np.array([self._poly2_x(a, b, c, yy) for yy in ys_eval], dtype=np.float64)
        in_bounds = np.logical_and(x_eval >= -0.10 * roi_w, x_eval <= 1.10 * roi_w)
        in_bounds_score = float(np.mean(in_bounds)) if x_eval.size > 0 else 0.0

        curvature_mag = abs(a)
        if curvature_mag <= 0.0025:
            curve_score = 1.0
        elif curvature_mag <= 0.006:
            curve_score = 0.7
        elif curvature_mag <= 0.012:
            curve_score = 0.35
        else:
            curve_score = 0.1

        if mean_res <= 4.0 and med_res <= 3.0:
            residual_score = 1.0
        elif mean_res <= 7.0 and med_res <= 5.0:
            residual_score = 0.8
        elif mean_res <= 11.0 and med_res <= 8.0:
            residual_score = 0.55
        elif mean_res <= 16.0:
            residual_score = 0.30
        else:
            residual_score = 0.10

        score = (
            0.38 * residual_score
            + 0.22 * y_span_score
            + 0.18 * row_score
            + 0.14 * in_bounds_score
            + 0.08 * curve_score
        )

        return float(max(0.0, min(1.0, score)))

    def _fit_distance_px(
        self,
        fit_a: Tuple[float, float, float],
        fit_b: Tuple[float, float, float],
        ys: np.ndarray,
    ) -> float:
        xa = np.array([self._poly2_x(*fit_a, yy) for yy in ys], dtype=np.float64)
        xb = np.array([self._poly2_x(*fit_b, yy) for yy in ys], dtype=np.float64)
        return float(np.mean(np.abs(xa - xb))) if xa.size > 0 else 1e9

    def _blend_fit(
        self,
        prev_fit: Tuple[float, float, float],
        new_fit: Tuple[float, float, float],
        alpha: float,
    ) -> Tuple[float, float, float]:
        a = alpha * prev_fit[0] + (1.0 - alpha) * new_fit[0]
        b = alpha * prev_fit[1] + (1.0 - alpha) * new_fit[1]
        c = alpha * prev_fit[2] + (1.0 - alpha) * new_fit[2]
        return float(a), float(b), float(c)

    def _stabilize_fit(
        self,
        new_fit: Optional[Tuple[float, float, float]],
        prev_fit: Optional[Tuple[float, float, float]],
        ys: np.ndarray,
        roi_h: int,
        roi_w: int,
        side_name: str,
    ) -> Tuple[Optional[Tuple[float, float, float]], dict]:
        debug = {
            f"{side_name}_fit_raw_ok": int(new_fit is not None),
            f"{side_name}_fit_stabilized": 0,
            f"{side_name}_fit_reused_prev": 0,
            f"{side_name}_fit_jump_px": 0.0,
        }

        if new_fit is None and prev_fit is None:
            return None, debug

        if new_fit is None and prev_fit is not None:
            debug[f"{side_name}_fit_reused_prev"] = 1
            return prev_fit, debug

        if new_fit is not None and prev_fit is None:
            return new_fit, debug

        jump_px = self._fit_distance_px(prev_fit, new_fit, ys)
        debug[f"{side_name}_fit_jump_px"] = float(jump_px)

        if jump_px > self.fit_jump_limit_px:
            debug[f"{side_name}_fit_reused_prev"] = 1
            return prev_fit, debug

        blended = self._blend_fit(prev_fit, new_fit, self.temporal_alpha_fit)
        debug[f"{side_name}_fit_stabilized"] = 1
        return blended, debug

    def _choose_single_side_from_reject(
        self,
        reason: str,
        left_fit: Optional[Tuple[float, float, float]],
        right_fit: Optional[Tuple[float, float, float]],
        left_pts: List[Tuple[int, int]],
        right_pts: List[Tuple[int, int]],
        ys_eval: np.ndarray,
        roi_w: int,
        roi_h: int,
    ) -> Tuple[str, dict]:
        left_score = self._score_side_fit(
            fit=left_fit,
            pts=left_pts,
            ys=ys_eval,
            roi_w=roi_w,
            roi_h=roi_h,
        )
        right_score = self._score_side_fit(
            fit=right_fit,
            pts=right_pts,
            ys=ys_eval,
            roi_w=roi_w,
            roi_h=roi_h,
        )

        raw_choice = "left" if left_score >= right_score else "right"
        chosen = raw_choice
        switched = 0

        prev_side = self.prev_fallback_side
        prev_frames = self.prev_fallback_side_frames

        adj_left = left_score
        adj_right = right_score

        if self.turn_mode == "left":
            adj_right += 0.22
        elif self.turn_mode == "right":
            adj_left += 0.22

        if prev_side == "left":
            adj_left += self.fallback_keep_bias
        elif prev_side == "right":
            adj_right += self.fallback_keep_bias

        preferred = "left" if adj_left >= adj_right else "right"

        if self.turn_mode == "left" and right_fit is not None and right_score >= 0.45:
            if (adj_right + 0.04) >= adj_left:
                preferred = "right"

        if self.turn_mode == "right" and left_fit is not None and left_score >= 0.45:
            if (adj_left + 0.04) >= adj_right:
                preferred = "left"

        if prev_side in ("left", "right"):
            if prev_side == "left":
                switch_advantage = right_score - left_score
                if preferred == "right":
                    if prev_frames < self.min_fallback_hold_frames and switch_advantage < (self.fallback_switch_margin + 0.05):
                        chosen = "left"
                    elif switch_advantage < self.fallback_switch_margin:
                        chosen = "left"
                    else:
                        chosen = "right"
                else:
                    chosen = "left"
            else:
                switch_advantage = left_score - right_score
                if preferred == "left":
                    if prev_frames < self.min_fallback_hold_frames and switch_advantage < (self.fallback_switch_margin + 0.05):
                        chosen = "right"
                    elif switch_advantage < self.fallback_switch_margin:
                        chosen = "right"
                    else:
                        chosen = "left"
                else:
                    chosen = "right"
        else:
            chosen = preferred

        if prev_side and chosen != prev_side:
            switched = 1
            self.prev_fallback_side_frames = 1
        else:
            self.prev_fallback_side_frames += 1 if chosen else 0

        self.prev_fallback_side = chosen

        return chosen, {
            "pair_reject_reason": str(reason),
            "left_fit_quality": float(left_score),
            "right_fit_quality": float(right_score),
            "pair_reject_fallback_side": chosen,
            "pair_reject_raw_choice": raw_choice,
            "pair_reject_switched_side": int(switched),
            "pair_reject_prev_side": prev_side,
            "pair_reject_prev_side_frames": int(prev_frames),
        }

    def _append_center_from_single_side(
        self,
        center_pts_img: List[Tuple[float, float]],
        fit: Tuple[float, float, float],
        ys: np.ndarray,
        roi_w: int,
        roi_h: int,
        side: str,
    ) -> None:
        width_profile = self._get_single_side_width_profile(ys=ys, roi_h=roi_h)
        min_y = 0.40 * roi_h

        for i, yy in enumerate(ys):
            if yy < min_y:
                continue

            x_lane = self._poly2_x(*fit, yy)
            lane_half_width_px = 0.5 * float(width_profile[i])

            xc = x_lane + lane_half_width_px if side == "left" else x_lane - lane_half_width_px

            if -0.10 * roi_w <= xc <= 1.10 * roi_w:
                center_pts_img.append((float(xc), float(yy + self.roi_y0)))
                
    def _single_side_continuation_candidate(
        self,
        left_fit: Optional[Tuple[float, float, float]],
        right_fit: Optional[Tuple[float, float, float]],
        left_pts: List[Tuple[int, int]],
        right_pts: List[Tuple[int, int]],
        ys: np.ndarray,
        roi_w: int,
        roi_h: int,
    ) -> Tuple[str, dict]:
        """
        Decide whether a pair failure is really just a temporary one-visible-lane case.

        Returns:
            ("left", debug)   -> prefer controlled left-only continuation
            ("right", debug)  -> prefer controlled right-only continuation
            ("", debug)       -> no strong continuation candidate
        """
        left_score = self._score_side_fit(
            fit=left_fit,
            pts=left_pts,
            ys=ys,
            roi_w=roi_w,
            roi_h=roi_h,
        )
        right_score = self._score_side_fit(
            fit=right_fit,
            pts=right_pts,
            ys=ys,
            roi_w=roi_w,
            roi_h=roi_h,
        )

        left_rows = len({int(y) for _, y in left_pts})
        right_rows = len({int(y) for _, y in right_pts})

        left_strong = (
            left_fit is not None
            and left_score >= 0.45
            and left_rows >= 8
            and len(left_pts) >= 12
        )

        right_strong = (
            right_fit is not None
            and right_score >= 0.45
            and right_rows >= 8
            and len(right_pts) >= 12
        )

        left_weak = (
            left_fit is None
            or left_score < 0.30
            or left_rows < 6
            or len(left_pts) < 10
        )

        right_weak = (
            right_fit is None
            or right_score < 0.30
            or right_rows < 6
            or len(right_pts) < 10
        )

        choice = ""

        if right_strong and left_weak:
            choice = "right"
        elif left_strong and right_weak:
            choice = "left"

        debug = {
            "single_side_cont_candidate": choice,
            "single_side_left_score": float(left_score),
            "single_side_right_score": float(right_score),
            "single_side_left_rows": int(left_rows),
            "single_side_right_rows": int(right_rows),
        }

        return choice, debug
                
    def _build_centerline_points(
        self,
        left_fit: Optional[Tuple[float, float, float]],
        right_fit: Optional[Tuple[float, float, float]],
        left_pts: List[Tuple[int, int]],
        right_pts: List[Tuple[int, int]],
        ys: np.ndarray,
        roi_w: int,
        roi_h: int,
    ) -> Tuple[List[Tuple[float, float]], str, dict]:
        center_pts_img: List[Tuple[float, float]] = []
        lane_pair_debug = self._base_lane_pair_debug(ys)

        if left_fit is not None and right_fit is not None:
            pair_valid, lane_pair_debug_base = self._validate_lane_pair(
                left_fit=left_fit,
                right_fit=right_fit,
                ys=ys,
                roi_h=roi_h,
                roi_w=roi_w,
            )
            lane_pair_debug.update(lane_pair_debug_base)

            # =========================================================
            # EARLY DIRECT SINGLE-SIDE CONTINUATION TEST
            # =========================================================
            left_fit_quality = self._score_side_fit(
                fit=left_fit,
                pts=left_pts,
                ys=ys,
                roi_w=roi_w,
                roi_h=roi_h,
            )
            right_fit_quality = self._score_side_fit(
                fit=right_fit,
                pts=right_pts,
                ys=ys,
                roi_w=roi_w,
                roi_h=roi_h,
            )

            lane_pair_debug["left_fit_quality"] = float(left_fit_quality)
            lane_pair_debug["right_fit_quality"] = float(right_fit_quality)

            left_rows = len({int(y) for _, y in left_pts})
            right_rows = len({int(y) for _, y in right_pts})

            left_strong = (
                left_fit is not None
                and left_fit_quality >= 0.45
                and left_rows >= 8
                and len(left_pts) >= 12
            )
            right_strong = (
                right_fit is not None
                and right_fit_quality >= 0.45
                and right_rows >= 8
                and len(right_pts) >= 12
            )

            left_weak = (
                left_fit is None
                or left_fit_quality < 0.30
                or left_rows < 6
                or len(left_pts) < 10
            )
            right_weak = (
                right_fit is None
                or right_fit_quality < 0.30
                or right_rows < 6
                or len(right_pts) < 10
            )

            # Hard LEFT turn -> prefer RIGHT lane directly
            if self.turn_mode == "left" and right_strong and left_weak:
                lane_pair_debug["pair_reject_fallback_side"] = "right_direct_test"
                lane_pair_debug["lane_width_reason"] = "direct_right_test"
                return self._build_single_side_centerline(
                    fit=right_fit,
                    pts=right_pts,
                    ys=ys,
                    roi_w=roi_w,
                    roi_h=roi_h,
                    side="right",
                    lane_pair_debug=lane_pair_debug,
                    mode_name="right_only_from_pair_reject",
                )

            # Hard RIGHT turn -> prefer LEFT lane directly
            if self.turn_mode == "right" and left_strong and right_weak:
                lane_pair_debug["pair_reject_fallback_side"] = "left_direct_test"
                lane_pair_debug["lane_width_reason"] = "direct_left_test"
                return self._build_single_side_centerline(
                    fit=left_fit,
                    pts=left_pts,
                    ys=ys,
                    roi_w=roi_w,
                    roi_h=roi_h,
                    side="left",
                    lane_pair_debug=lane_pair_debug,
                    mode_name="left_only_from_pair_reject",
                )

            mids = []
            valid_rows = []
            relaxed_valid_rows = []

            for yy in ys:
                xl = self._poly2_x(*left_fit, yy)
                xr = self._poly2_x(*right_fit, yy)
                width = xr - xl
                mid = 0.5 * (xl + xr)

                min_w, max_w = self._width_bounds_px(yy, roi_h)

                strict_ok = (
                    np.isfinite(width)
                    and np.isfinite(mid)
                    and (width > min_w)
                    and (width < max_w)
                )

                relaxed_ok = (
                    np.isfinite(width)
                    and np.isfinite(mid)
                    and (width > 0.82 * min_w)
                    and (width < 1.18 * max_w)
                )

                mids.append(float(mid))
                valid_rows.append(bool(strict_ok))
                relaxed_valid_rows.append(bool(relaxed_ok))

            row_pair_center = self._build_centerline_from_row_pairs(
                left_pts=left_pts,
                right_pts=right_pts,
                roi_h=roi_h,
                roi_w=roi_w,
            )

            if len(row_pair_center) >= max(8, int(0.75 * self.min_points)):
                lane_pair_debug["lane_pair_valid"] = 1
                lane_pair_debug["lane_width_reason"] = "row_pairs"
                lane_pair_debug["row_pair_center_count"] = int(len(row_pair_center))
                return row_pair_center, "row_pair_midpoint", lane_pair_debug

            mids_arr = np.array(mids, dtype=np.float64)
            valid_rows_arr = np.array(valid_rows, dtype=bool)
            relaxed_valid_rows_arr = np.array(relaxed_valid_rows, dtype=bool)

            strict_count = int(np.count_nonzero(valid_rows_arr))
            relaxed_count = int(np.count_nonzero(relaxed_valid_rows_arr))

            enough_rows = strict_count >= max(
                10,
                int(self.min_pair_valid_fraction * len(ys)),
            )

            enough_rows_repaired = strict_count >= max(
                6,
                int(0.18 * len(ys)),
            )

            enough_rows_relaxed = relaxed_count >= max(
                7,
                int(0.22 * len(ys)),
            )

            lane_pair_debug["strict_midpoint_rows"] = strict_count
            lane_pair_debug["relaxed_midpoint_rows"] = relaxed_count

            if pair_valid and enough_rows:
                center_pts_img = self._centerline_from_midpoints(
                    mids=mids_arr,
                    ys=ys,
                    valid_rows=valid_rows_arr,
                    roi_w=roi_w,
                )

                if lane_pair_debug.get("lane_width_reason") == "relaxed_midpoints":
                    lane_pair_debug["lane_pair_valid"] = 1
                    return center_pts_img, "both_relaxed", lane_pair_debug

                return center_pts_img, "both", lane_pair_debug

            if enough_rows_repaired:
                use_rows = valid_rows_arr
                mode_name = "both_midpoint_repaired"

                relaxed_rows = int(lane_pair_debug.get("relaxed_midpoint_rows", 0))
                total_rows = max(1, int(lane_pair_debug.get("lane_pair_total_rows", len(ys))))

                if (relaxed_rows / total_rows) >= self.relaxed_pair_valid_fraction or enough_rows_relaxed:
                    use_rows = relaxed_valid_rows_arr
                    lane_pair_debug["lane_pair_valid"] = 1
                    lane_pair_debug["lane_width_reason"] = "relaxed_midpoints"
                    mode_name = "both_relaxed"
                else:
                    lane_pair_debug["lane_width_reason"] = "strict_midpoints_repaired"

                center_pts_img = self._centerline_from_midpoints(
                    mids=mids_arr,
                    ys=ys,
                    valid_rows=use_rows,
                    roi_w=roi_w,
                )
                return center_pts_img, mode_name, lane_pair_debug

            hold_relaxed_thresh = 5
            hold_strict_thresh = 3

            if self.turn_mode in ("left", "right"):
                hold_relaxed_thresh = 4
                hold_strict_thresh = 2

            # TEMPORARY DIAGNOSTIC:
            # disable held_prev_centerline so we can see whether live one-lane
            # continuation works on its own.
            if False:
                lane_pair_debug["lane_width_reason"] = str(
                    lane_pair_debug.get("lane_width_reason", "pair_rejected")
                )
                lane_pair_debug["pair_reject_fallback_side"] = "hold_prev"
                return list(self.prev_center_pts_img), "held_prev_centerline", lane_pair_debug

            cont_side, cont_debug = self._single_side_continuation_candidate(
                left_fit=left_fit,
                right_fit=right_fit,
                left_pts=left_pts,
                right_pts=right_pts,
                ys=ys,
                roi_w=roi_w,
                roi_h=roi_h,
            )
            lane_pair_debug.update(cont_debug)

            if cont_side == "left":
                lane_pair_debug["pair_reject_fallback_side"] = "left_continuation"
                return self._build_single_side_centerline(
                    fit=left_fit,
                    pts=left_pts,
                    ys=ys,
                    roi_w=roi_w,
                    roi_h=roi_h,
                    side="left",
                    lane_pair_debug=lane_pair_debug,
                    mode_name="left_only_from_pair_reject",
                )

            if cont_side == "right":
                lane_pair_debug["pair_reject_fallback_side"] = "right_continuation"
                return self._build_single_side_centerline(
                    fit=right_fit,
                    pts=right_pts,
                    ys=ys,
                    roi_w=roi_w,
                    roi_h=roi_h,
                    side="right",
                    lane_pair_debug=lane_pair_debug,
                    mode_name="right_only_from_pair_reject",
                )

            return center_pts_img, "pair_rejected", lane_pair_debug

        if left_fit is not None:
            return self._build_single_side_centerline(
                fit=left_fit,
                pts=left_pts,
                ys=ys,
                roi_w=roi_w,
                roi_h=roi_h,
                side="left",
                lane_pair_debug=lane_pair_debug,
                mode_name="left_only",
            )

        if right_fit is not None:
            return self._build_single_side_centerline(
                fit=right_fit,
                pts=right_pts,
                ys=ys,
                roi_w=roi_w,
                roi_h=roi_h,
                side="right",
                lane_pair_debug=lane_pair_debug,
                mode_name="right_only",
            )

        return center_pts_img, "none", lane_pair_debug

    def _smooth_centerline_img_points(
        self,
        center_pts_img: List[Tuple[float, float]],
        prev_center_pts_img: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        if len(center_pts_img) < 3:
            return center_pts_img

        xs = np.array([p[0] for p in center_pts_img], dtype=np.float64)
        ys = np.array([p[1] for p in center_pts_img], dtype=np.float64)

        current_mode = getattr(self, "current_fit_mode_for_smoothing", "none")

        spatial_window = 5
        temporal_alpha = self.temporal_alpha_center

        if current_mode in ("both_midpoint_repaired", "row_pair_midpoint"):
            spatial_window = 5
            temporal_alpha = 0.68
        elif current_mode in (
            "left_only_from_pair_reject",
            "right_only_from_pair_reject",
            "left_only",
            "right_only",
        ):
            spatial_window = 9
            temporal_alpha = 0.88
        elif current_mode in ("held_prev_centerline", "held_last_good"):
            spatial_window = 9
            temporal_alpha = 0.92
        else:
            spatial_window = 5
            temporal_alpha = self.temporal_alpha_center

        xs = np.array(self._smooth_1d(xs.tolist(), window=spatial_window), dtype=np.float64)

        if len(prev_center_pts_img) == len(center_pts_img) and len(prev_center_pts_img) >= 3:
            prev_xs = np.array([p[0] for p in prev_center_pts_img], dtype=np.float64)
            xs = temporal_alpha * prev_xs + (1.0 - temporal_alpha) * xs

        return [(float(x), float(y)) for x, y in zip(xs, ys)]

    def _maybe_hold_last_good_centerline(
        self,
        center_pts_img: List[Tuple[float, float]],
        fit_mode: str,
        lane_pair_valid: int,
    ) -> Tuple[List[Tuple[float, float]], dict]:
        hold_used = 0
        hold_reason = ""
        fit_mode_after_hold = fit_mode

        current_good = len(center_pts_img) >= self.min_points and (
            lane_pair_valid == 1 or fit_mode in ("left_only", "right_only", "both", "both_midpoint_repaired")
        )

        if current_good:
            return center_pts_img, {
                "hold_used": hold_used,
                "hold_reason": hold_reason,
                "fit_mode_after_hold": fit_mode_after_hold,
                "frames_since_good_pair": int(self.frames_since_good_pair),
            }

        # TEMPORARY DIAGNOSTIC:
        # During sharp turns, do not reuse the old "last good" centerline.
        # This lets us see whether the fresh live one-lane logic works on its own.
        if self.turn_mode in ("left", "right"):
            return center_pts_img, {
                "hold_used": 0,
                "hold_reason": "",
                "fit_mode_after_hold": fit_mode,
                "frames_since_good_pair": int(self.frames_since_good_pair),
            }

        if 0 < len(self.prev_good_center_pts_img) and self.frames_since_good_pair <= self.hold_last_good_frames:
            hold_used = 1
            hold_reason = "reuse_last_good_centerline"
            fit_mode_after_hold = "held_last_good"
            return list(self.prev_good_center_pts_img), {
                "hold_used": hold_used,
                "hold_reason": hold_reason,
                "fit_mode_after_hold": fit_mode_after_hold,
                "frames_since_good_pair": int(self.frames_since_good_pair),
            }

        return center_pts_img, {
            "hold_used": hold_used,
            "hold_reason": hold_reason,
            "fit_mode_after_hold": fit_mode_after_hold,
            "frames_since_good_pair": int(self.frames_since_good_pair),
        }

    def _image_centerline_to_local(
        self,
        center_pts_img: List[Tuple[float, float]],
        frame_height: int,
        frame_width: int,
    ) -> Tuple[List[float], List[float]]:
        centerline_xs: List[float] = []
        centerline_ys: List[float] = []

        for xc, y_img in center_pts_img:
            x_local = (frame_height - 1 - y_img) * self.x_m_per_px
            y_local = ((frame_width * 0.5) - xc) * self.y_m_per_px
            centerline_xs.append(float(x_local))
            centerline_ys.append(float(y_local))

        if len(centerline_ys) >= 5:
            centerline_ys = self._smooth_1d(centerline_ys, window=5)

        return centerline_xs, centerline_ys

    def _validate_lane_pair(
        self,
        left_fit: Tuple[float, float, float],
        right_fit: Tuple[float, float, float],
        ys: np.ndarray,
        roi_h: int,
        roi_w: int,
    ) -> Tuple[bool, dict]:
        widths = []
        strict_valid_rows = 0
        relaxed_valid_rows = 0
        crossed_rows = 0

        # Region diagnostics
        bottom_valid_rows = 0
        mid_valid_rows = 0
        top_valid_rows = 0

        bottom_relaxed_rows = 0
        mid_relaxed_rows = 0
        top_relaxed_rows = 0

        bottom_crossed_rows = 0
        mid_crossed_rows = 0
        top_crossed_rows = 0

        bottom_widths: List[float] = []
        mid_widths: List[float] = []
        top_widths: List[float] = []

        for yy in ys:
            xl = self._poly2_x(*left_fit, yy)
            xr = self._poly2_x(*right_fit, yy)
            width = float(xr - xl)
            widths.append(width)

            y_norm = float(yy / max(1.0, roi_h - 1))
            if y_norm >= 0.66:
                region = "bottom"
            elif y_norm >= 0.33:
                region = "mid"
            else:
                region = "top"

            if width <= 0.0:
                crossed_rows += 1
                if region == "bottom":
                    bottom_crossed_rows += 1
                elif region == "mid":
                    mid_crossed_rows += 1
                else:
                    top_crossed_rows += 1

            min_w, max_w = self._width_bounds_px(yy, roi_h)

            strict_ok = (min_w < width < max_w)

            relaxed_min_w = 0.75 * min_w
            relaxed_max_w = 1.25 * max_w
            relaxed_ok = (relaxed_min_w < width < relaxed_max_w)

            if strict_ok:
                strict_valid_rows += 1
                if region == "bottom":
                    bottom_valid_rows += 1
                elif region == "mid":
                    mid_valid_rows += 1
                else:
                    top_valid_rows += 1

            if relaxed_ok:
                relaxed_valid_rows += 1
                if region == "bottom":
                    bottom_relaxed_rows += 1
                elif region == "mid":
                    mid_relaxed_rows += 1
                else:
                    top_relaxed_rows += 1

            if np.isfinite(width):
                if region == "bottom":
                    bottom_widths.append(width)
                elif region == "mid":
                    mid_widths.append(width)
                else:
                    top_widths.append(width)

        widths_arr = np.array(widths, dtype=np.float64)

        if widths_arr.size == 0:
            return False, {
                "lane_pair_valid": 0,
                "lane_width_min_px": 0.0,
                "lane_width_max_px": 0.0,
                "lane_width_mean_px": 0.0,
                "lane_width_std_px": 0.0,
                "lane_width_reason": "empty",
                "lane_pair_valid_rows": 0,
                "lane_pair_total_rows": 0,
                "lane_crossed_rows": 0,
                "strict_midpoint_rows": 0,
                "relaxed_midpoint_rows": 0,
                "bottom_valid_rows": 0,
                "mid_valid_rows": 0,
                "top_valid_rows": 0,
                "bottom_relaxed_rows": 0,
                "mid_relaxed_rows": 0,
                "top_relaxed_rows": 0,
                "bottom_crossed_rows": 0,
                "mid_crossed_rows": 0,
                "top_crossed_rows": 0,
                "bottom_width_std_px": 0.0,
                "mid_width_std_px": 0.0,
                "top_width_std_px": 0.0,
            }

        widths_smooth = np.array(self._smooth_1d(widths_arr.tolist(), window=7), dtype=np.float64)

        min_w_s = float(np.min(widths_smooth))
        max_w_s = float(np.max(widths_smooth))
        mean_w_s = float(np.mean(widths_smooth))
        std_w_s = float(np.std(widths_smooth))

        total_rows = max(1, len(ys))
        crossed_frac = float(crossed_rows / total_rows)
        strict_valid_frac = float(strict_valid_rows / total_rows)
        relaxed_valid_frac = float(relaxed_valid_rows / total_rows)

        def _safe_std(vals: List[float]) -> float:
            if len(vals) < 2:
                return 0.0
            return float(np.std(np.array(vals, dtype=np.float64)))

        bottom_width_std = _safe_std(bottom_widths)
        mid_width_std = _safe_std(mid_widths)
        top_width_std = _safe_std(top_widths)
        """
        valid = True
        reason = "ok"

        if crossed_frac > self.max_crossed_fraction:
            valid = False
            reason = "crossed_often"
        elif strict_valid_frac >= self.min_pair_valid_fraction:
            valid = True
            reason = "ok"
        elif relaxed_valid_frac >= self.relaxed_pair_valid_fraction:
            valid = True
            reason = "relaxed_midpoints"
        elif std_w_s > self.max_width_std_px and (min_w_s < 20.0 or max_w_s > 0.92 * roi_w):
            valid = False
            reason = "too_inconsistent"
        else:
            valid = False
            reason = "too_few_valid_width_rows"
        """

        # TEMPORARY DIAGNOSTIC MODE:
        # Keep pair metrics/debug the same, but greatly reduce how aggressively
        # the pair validator rejects the geometry.
        valid = False
        reason = "diag_too_few_rows"

        # Only reject if the pair is truly unusable.
        if crossed_frac > 0.85:
            valid = False
            reason = "diag_crossed_often"
        elif strict_valid_frac >= 0.10:
            valid = True
            reason = "diag_strict_ok"
        elif relaxed_valid_frac >= 0.18:
            valid = True
            reason = "diag_relaxed_ok"
        else:
            valid = False
            reason = "diag_too_few_rows"

        return valid, {
            "lane_pair_valid": int(valid),
            "lane_width_min_px": min_w_s,
            "lane_width_max_px": max_w_s,
            "lane_width_mean_px": mean_w_s,
            "lane_width_std_px": std_w_s,
            "lane_width_reason": reason,
            "lane_pair_valid_rows": int(strict_valid_rows),
            "lane_pair_total_rows": int(total_rows),
            "lane_crossed_rows": int(crossed_rows),
            "strict_midpoint_rows": int(strict_valid_rows),
            "relaxed_midpoint_rows": int(relaxed_valid_rows),

            "bottom_valid_rows": int(bottom_valid_rows),
            "mid_valid_rows": int(mid_valid_rows),
            "top_valid_rows": int(top_valid_rows),

            "bottom_relaxed_rows": int(bottom_relaxed_rows),
            "mid_relaxed_rows": int(mid_relaxed_rows),
            "top_relaxed_rows": int(top_relaxed_rows),

            "bottom_crossed_rows": int(bottom_crossed_rows),
            "mid_crossed_rows": int(mid_crossed_rows),
            "top_crossed_rows": int(top_crossed_rows),

            "bottom_width_std_px": float(bottom_width_std),
            "mid_width_std_px": float(mid_width_std),
            "top_width_std_px": float(top_width_std),
        }

    # =========================================================
    # Polynomial Helpers
    # =========================================================
    @staticmethod
    def _poly2_fit_x_of_y(
        points_xy: List[Tuple[int, int]]
    ) -> Optional[Tuple[float, float, float]]:
        if len(points_xy) < 6:
            return None

        xs = np.array([p[0] for p in points_xy], dtype=np.float64)
        ys = np.array([p[1] for p in points_xy], dtype=np.float64)

        try:
            y_min = float(np.min(ys))
            y_max = float(np.max(ys))
            y_norm = (ys - y_min) / max(1.0, (y_max - y_min))
            weights = 0.6 + 1.4 * y_norm

            a, b, c = np.polyfit(ys, xs, 2, w=weights)

            x_pred = a * ys * ys + b * ys + c
            residuals = np.abs(xs - x_pred)

            med = float(np.median(residuals))
            mad = float(np.median(np.abs(residuals - med)))
            robust_thresh = max(10.0, med + 2.5 * max(mad, 1.0))

            keep = residuals <= robust_thresh

            if int(np.count_nonzero(keep)) >= 6:
                xs_in = xs[keep]
                ys_in = ys[keep]

                y_min = float(np.min(ys_in))
                y_max = float(np.max(ys_in))
                y_norm = (ys_in - y_min) / max(1.0, (y_max - y_min))
                weights = 0.6 + 1.4 * y_norm

                a, b, c = np.polyfit(ys_in, xs_in, 2, w=weights)

            return float(a), float(b), float(c)

        except Exception:
            return None

    @staticmethod
    def _poly2_x(a: float, b: float, c: float, y: float) -> float:
        return a * y * y + b * y + c

    @staticmethod
    def _smooth_1d(vals: List[float], window: int = 5) -> List[float]:
        if len(vals) < 3 or window <= 1:
            return vals

        w = max(3, int(window))
        if w % 2 == 0:
            w += 1

        pad = w // 2
        arr = np.array(vals, dtype=np.float64)
        arr_pad = np.pad(arr, (pad, pad), mode="edge")
        kernel = np.ones(w, dtype=np.float64) / float(w)
        smoothed = np.convolve(arr_pad, kernel, mode="valid")
        return smoothed.tolist()

    def _extract_runs(self, xs: np.ndarray) -> List[Tuple[int, int]]:
        if xs.size == 0:
            return []

        runs: List[Tuple[int, int]] = []
        start = int(xs[0])
        prev = int(xs[0])

        for x in xs[1:]:
            x = int(x)
            if x == prev + 1:
                prev = x
            else:
                runs.append((start, prev))
                start = x
                prev = x

        runs.append((start, prev))
        return runs

    def _choose_run_center(
        self,
        xs: np.ndarray,
        target_x: float,
    ) -> Optional[int]:
        runs = self._extract_runs(xs)
        if not runs:
            return None

        best_center = None
        best_cost = None

        for run_start, run_end in runs:
            run_len = run_end - run_start + 1
            center = 0.5 * (run_start + run_end)

            dist_cost = abs(center - target_x)
            length_bonus = min(20.0, float(run_len))
            cost = dist_cost - 0.35 * length_bonus

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_center = int(round(center))

        return best_center

    def _build_binary_sample_rows(self, roi_h: int) -> np.ndarray:
        y_top = int(0.18 * roi_h)
        y_bottom = roi_h - 1

        t = np.linspace(0.0, 1.0, 72)
        ys = y_top + (y_bottom - y_top) * (t ** 0.55)
        ys = np.clip(np.round(ys).astype(int), y_top, y_bottom)

        return np.unique(ys)

    def _build_centerline_from_row_pairs(
        self,
        left_pts: List[Tuple[int, int]],
        right_pts: List[Tuple[int, int]],
        roi_h: int,
        roi_w: int,
    ) -> List[Tuple[float, float]]:
        left_by_y = {int(y): float(x) for x, y in left_pts}
        right_by_y = {int(y): float(x) for x, y in right_pts}

        common_ys = sorted(set(left_by_y.keys()) & set(right_by_y.keys()))
        center_pts_img: List[Tuple[float, float]] = []

        for yy in common_ys:
            xl = left_by_y[yy]
            xr = right_by_y[yy]
            width = xr - xl

            min_w, max_w = self._width_bounds_px(yy, roi_h)
            if not (min_w <= width <= max_w):
                continue

            xc = 0.5 * (xl + xr)
            if -0.10 * roi_w <= xc <= 1.10 * roi_w:
                center_pts_img.append((float(xc), float(yy + self.roi_y0)))

        if len(center_pts_img) >= 5:
            xs = [p[0] for p in center_pts_img]
            ys = [p[1] for p in center_pts_img]
            xs = self._smooth_1d(xs, window=7)
            center_pts_img = list(zip(xs, ys))

        return center_pts_img

    def _should_hold_centerline_on_side_switch(
        self,
        left_fit_quality: float,
        right_fit_quality: float,
        chosen_side: str,
    ) -> bool:
        if not self.prev_center_pts_img:
            return False

        quality_gap = abs(left_fit_quality - right_fit_quality)
        return quality_gap < 0.08

    # =========================================================
    # Confidence
    # =========================================================
    def _compute_confidence(
        self,
        left_fit,
        right_fit,
        center_pts,
        lane_segs,
        fit_mode,
        lane_pair_valid,
        pair_reason,
        left_fit_quality,
        right_fit_quality,
    ):
        if len(center_pts) < self.min_points:
            return 0.0

        seg_score = min(1.0, len(lane_segs) / 18.0)
        pt_score = min(1.0, len(center_pts) / 32.0)

        if fit_mode in ("both", "row_pair_midpoint"):
            fit_score = 1.0
        elif fit_mode == "both_relaxed":
            fit_score = 0.88
        elif fit_mode == "both_midpoint_repaired":
            fit_score = 0.82
        elif fit_mode == "held_last_good":
            fit_score = 0.72
        elif fit_mode == "held_prev_centerline":
            fit_score = 0.74
        elif fit_mode in (
            "left_only",
            "right_only",
            "left_only_from_pair_reject",
            "right_only_from_pair_reject",
        ):
            fit_score = 0.66
        elif left_fit is not None or right_fit is not None:
            fit_score = 0.45
        else:
            fit_score = 0.0

        side_quality = max(left_fit_quality, right_fit_quality)
        if left_fit is not None and right_fit is not None:
            side_quality = 0.5 * (left_fit_quality + right_fit_quality)
        side_quality = float(np.clip(side_quality, 0.0, 1.0))

        confidence = (
            0.45 * fit_score
            + 0.18 * seg_score
            + 0.17 * pt_score
            + 0.20 * side_quality
        )

        if lane_pair_valid == 0 and fit_mode == "both":
            confidence *= 0.70
        elif lane_pair_valid == 0 and fit_mode == "both_midpoint_repaired":
            confidence *= 0.90
        elif lane_pair_valid == 0 and fit_mode in ("left_only_from_pair_reject", "right_only_from_pair_reject"):
            confidence *= 0.90

        if fit_mode == "both_relaxed":
            confidence = max(confidence, 0.64)

        if pair_reason == "crossed_often":
            confidence *= 0.68
        elif pair_reason == "too_inconsistent":
            confidence *= 0.86
        elif pair_reason == "too_few_valid_width_rows":
            confidence *= 0.92

        return float(np.clip(confidence, 0.0, 1.0))

    # =========================================================
    # Debug Visualization
    # =========================================================
    def _build_debug_view(
        self,
        frame: np.ndarray,
        roi_y0: int,
        roi_poly: np.ndarray,
        hood_poly: np.ndarray,
        left_poly: np.ndarray,
        right_poly: np.ndarray,
        left_extra_poly: np.ndarray,
        right_extra_poly: np.ndarray,
        left_corner_poly: np.ndarray,
        right_corner_poly: np.ndarray,
        turn_mode: str,
        lane_segs: List[Tuple[int, int, int, int]],
        left_fit: Optional[Tuple[float, float, float]],
        right_fit: Optional[Tuple[float, float, float]],
        center_pts: List[Tuple[float, float]],
        ys: np.ndarray,
        confidence: float,
        hsv_vis: np.ndarray,
        red_mask_raw: np.ndarray,
        red_mask_clean: np.ndarray,
        lane_binary: np.ndarray,
    ) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]

        cv2.rectangle(vis, (0, roi_y0), (w - 1, h - 1), (0, 255, 0), 1)

        for poly, color in (
            (roi_poly, (0, 255, 255)),
            (hood_poly, (0, 0, 255)),
            (left_poly, (255, 0, 255)),
            (right_poly, (255, 255, 0)),
        ):
            poly_vis = poly.copy()
            poly_vis[:, :, 1] += roi_y0
            cv2.polylines(vis, [poly_vis], True, color, 1)

        # Main turn wedge
        if turn_mode == "left":
            poly_vis = left_extra_poly.copy()
            poly_vis[:, :, 1] += roi_y0
            cv2.polylines(vis, [poly_vis], True, (0, 165, 255), 2)
        elif turn_mode == "right":
            poly_vis = right_extra_poly.copy()
            poly_vis[:, :, 1] += roi_y0
            cv2.polylines(vis, [poly_vis], True, (0, 165, 255), 2)

        # Outside-lane corner assist
        if turn_mode == "left":
            poly_vis = right_corner_poly.copy()
            poly_vis[:, :, 1] += roi_y0
            cv2.polylines(vis, [poly_vis], True, (0, 100, 255), 2)
        elif turn_mode == "right":
            poly_vis = left_corner_poly.copy()
            poly_vis[:, :, 1] += roi_y0
            cv2.polylines(vis, [poly_vis], True, (0, 100, 255), 2)

        cv2.putText(
            vis,
            f"turn_mode={turn_mode}",
            (10, 79),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for x1, y1, x2, y2 in lane_segs:
            cv2.line(vis, (x1, y1 + roi_y0), (x2, y2 + roi_y0), (255, 0, 0), 2)

        self._draw_fit_polyline(vis, left_fit, ys, roi_y0, color=(0, 255, 255), thickness=2)
        self._draw_fit_polyline(vis, right_fit, ys, roi_y0, color=(0, 255, 255), thickness=2)

        if len(center_pts) > 1:
            pts = np.array([(int(x), int(y)) for x, y in center_pts], dtype=np.int32)
            cv2.polylines(vis, [pts], False, (0, 0, 255), 3)

        cv2.line(vis, (w // 2, h - 1), (w // 2, h - 60), (255, 255, 0), 2)

        cv2.putText(
            vis,
            f"conf={confidence:.2f} segs={len(lane_segs)} pts={len(center_pts)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        left_ok = 1 if left_fit is not None else 0
        right_ok = 1 if right_fit is not None else 0
        cv2.putText(
            vis,
            f"L={left_ok} R={right_ok}",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        thumb_h = 70
        thumb_w = 140

        red_mask_raw_bgr = cv2.cvtColor(red_mask_raw, cv2.COLOR_GRAY2BGR)
        red_mask_clean_bgr = cv2.cvtColor(red_mask_clean, cv2.COLOR_GRAY2BGR)
        lane_binary_bgr = cv2.cvtColor(lane_binary, cv2.COLOR_GRAY2BGR)

        thumbs = [
            ("hsv_vis", cv2.resize(hsv_vis, (thumb_w, thumb_h))),
            ("red_raw", cv2.resize(red_mask_raw_bgr, (thumb_w, thumb_h))),
            ("red_clean", cv2.resize(red_mask_clean_bgr, (thumb_w, thumb_h))),
            ("lane_binary", cv2.resize(lane_binary_bgr, (thumb_w, thumb_h))),
        ]

        x0 = w - thumb_w - 5
        y = 5
        for name, img in thumbs:
            if y + thumb_h <= h:
                vis[y:y + thumb_h, x0:x0 + thumb_w] = img
                cv2.putText(
                    vis,
                    name,
                    (x0 + 5, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                y += thumb_h + 5

        return vis

    def _show_debug_view(self, debug_frame: np.ndarray) -> None:
        cv2.imshow("lane_detector_debug", debug_frame)
        cv2.waitKey(1)