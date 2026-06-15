from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

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
    Real-image lane detector for red tape lanes.

    Keeps strong preprocessing but simplifies downstream logic:
      1) preprocess ROI and masks
      2) extract left/right lane points
      3) fit left/right curves
      4) use midpoint if lane pair is valid
      5) otherwise use single visible side + expected lane width
      6) briefly hold last good centerline if needed
    """

    def __init__(
        self,
        *,
        frame_width: int = 640,
        frame_height: int = 360,
        roi_y0: int = 200,
        min_points: int = 12,
        show_debug: bool = False,
        lane_width_m: float = 0.9,
        x_m_per_px: float = 0.03,
        y_m_per_px: float = 0.01,

        # Red HSV thresholds
        red_h1_low: int = 0,
        red_h1_high: int = 20,
        red_h2_low: int = 150,
        red_h2_high: int = 179,
        red_s_low: int = 50,
        red_v_low: int = 55,

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

        # Edge / line params
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 12,
        hough_min_line_length: int = 12,
        hough_max_line_gap: int = 18,

        # Component filtering
        use_component_filter: bool = True,
        min_component_area: int = 8,
        max_component_area_frac: float = 0.45,
        min_component_height_frac: float = 0.01,
        max_component_width_frac: float = 0.85,
        min_component_aspect_ratio: float = 1.15,

        # Temporal / stability params
        temporal_alpha_fit: float = 0.72,
        temporal_alpha_center: float = 0.70,
        fit_jump_limit_px: float = 90.0,
        hold_last_good_frames: int = 4,

        # Pair validation params
        min_pair_valid_fraction: float = 0.30,
        max_crossed_fraction: float = 0.25,
        max_width_std_px: float = 110.0,
    ):
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.roi_y0 = int(roi_y0)
        self.min_points = int(min_points)
        self.show_debug = bool(show_debug)

        self.x_m_per_px = float(x_m_per_px)
        self.y_m_per_px = float(y_m_per_px)
        self.lane_width_m = float(lane_width_m)

        self.red_h1_low = int(red_h1_low)
        self.red_h1_high = int(red_h1_high)
        self.red_h2_low = int(red_h2_low)
        self.red_h2_high = int(red_h2_high)
        self.red_s_low = int(red_s_low)
        self.red_v_low = int(red_v_low)

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
        self.hough_threshold = int(hough_threshold)
        self.hough_min_line_length = int(hough_min_line_length)
        self.hough_max_line_gap = int(hough_max_line_gap)

        self.use_component_filter = bool(use_component_filter)
        self.min_component_area = int(min_component_area)
        self.max_component_area_frac = float(max_component_area_frac)
        self.min_component_height_frac = float(min_component_height_frac)
        self.max_component_width_frac = float(max_component_width_frac)
        self.min_component_aspect_ratio = float(min_component_aspect_ratio)

        self.temporal_alpha_fit = float(temporal_alpha_fit)
        self.temporal_alpha_center = float(temporal_alpha_center)
        self.fit_jump_limit_px = float(fit_jump_limit_px)
        self.hold_last_good_frames = int(hold_last_good_frames)

        self.min_pair_valid_fraction = float(min_pair_valid_fraction)
        self.max_crossed_fraction = float(max_crossed_fraction)
        self.max_width_std_px = float(max_width_std_px)

        # Temporal memory
        self.prev_left_fit: Optional[Tuple[float, float, float]] = None
        self.prev_right_fit: Optional[Tuple[float, float, float]] = None
        self.prev_center_pts_img: List[Tuple[float, float]] = []
        self.prev_good_center_pts_img: List[Tuple[float, float]] = []
        self.frames_since_good_pair: int = 999999

    # =========================================================
    # Public API
    # =========================================================
    def detect(self, frame=None, vehicle_x: float = 0.0) -> LaneDetection:
        del vehicle_x  # kept for call compatibility

        if frame is None:
            return LaneDetection(
                centerline_xs=[],
                centerline_ys=[],
                confidence=0.0,
                debug_frame=None,
                debug_meta={
                    "mode": "real",
                    "fit_mode": "none",
                    "error": "frame=None",
                },
            )

        try:
            return self._detect_from_frame(frame)
        except Exception as e:
            print(f"[LaneDetector ERROR] {type(e).__name__}: {e}")
            return LaneDetection(
                centerline_xs=[],
                centerline_ys=[],
                confidence=0.0,
                debug_frame=None,
                debug_meta={
                    "mode": "real",
                    "error": str(e),
                    "fit_mode": "none",
                    "left_fit_ok": 0,
                    "right_fit_ok": 0,
                    "lane_seg_count": 0,
                    "center_pt_count": 0,
                },
            )

    # =========================================================
    # Main Detection
    # =========================================================
    def _detect_from_frame(self, frame: np.ndarray) -> LaneDetection:
        frame = self._prepare_frame(frame)
        h, w = frame.shape[:2]

        roi, roi_h, roi_w = self._extract_roi(frame)
        preprocess = self._preprocess_roi(roi, roi_h, roi_w)

        hsv_vis = preprocess["hsv_vis"]
        red_mask_raw = preprocess["red_mask_raw"]
        red_mask_clean = preprocess["red_mask_clean"]
        used_binary = preprocess["used_binary"]
        edges = preprocess["edges"]
        candidate_edges = preprocess["candidate_edges"]
        roi_poly = preprocess["roi_poly"]
        hood_poly = preprocess["hood_poly"]
        left_mask = preprocess["left_mask"]
        right_mask = preprocess["right_mask"]
        left_poly = preprocess["left_poly"]
        right_poly = preprocess["right_poly"]
        lower_focus_poly = preprocess["lower_focus_poly"]
        component_count = preprocess["component_count"]
        component_kept_count = preprocess["component_kept_count"]
        component_reject_count = preprocess["component_reject_count"]
        component_filtered_count = preprocess["component_filtered_count"]

        left_pts, right_pts, lane_segs, seg_debug = self._extract_lane_points(
            candidate_edges=candidate_edges,
            used_binary=used_binary,
            left_mask=left_mask,
            right_mask=right_mask,
            roi_w=roi_w,
            roi_h=roi_h,
        )

        left_fit_raw = self._poly2_fit_x_of_y(left_pts)
        right_fit_raw = self._poly2_fit_x_of_y(right_pts)

        ys = self._build_sample_ys(roi_h)

        left_fit, left_stab_debug = self._stabilize_fit(
            new_fit=left_fit_raw,
            prev_fit=self.prev_left_fit,
            ys=ys,
            side_name="left",
        )
        right_fit, right_stab_debug = self._stabilize_fit(
            new_fit=right_fit_raw,
            prev_fit=self.prev_right_fit,
            ys=ys,
            side_name="right",
        )

        center_pts_img, fit_mode, lane_pair_debug = self._build_centerline_points(
            left_fit=left_fit,
            right_fit=right_fit,
            ys=ys,
            roi_w=roi_w,
            roi_h=roi_h,
        )

        center_pts_img, hold_debug = self._maybe_hold_last_good_centerline(
            center_pts_img=center_pts_img,
            fit_mode=fit_mode,
            lane_pair_valid=lane_pair_debug.get("lane_pair_valid", 0),
        )
        fit_mode = hold_debug["fit_mode_after_hold"]

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
        )

        if len(centerline_xs) < 2:
            confidence = 0.0

        if lane_pair_debug.get("lane_pair_valid", 0) == 1 and len(center_pts_img) >= self.min_points:
            self.prev_good_center_pts_img = list(center_pts_img)
            self.frames_since_good_pair = 0
        else:
            self.frames_since_good_pair += 1

        self.prev_left_fit = left_fit
        self.prev_right_fit = right_fit
        self.prev_center_pts_img = list(center_pts_img)

        debug_meta = {
            "mode": "real",
            "fit_mode": fit_mode,
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
            "used_binary_count": int(np.count_nonzero(used_binary)),
            "edge_count": int(np.count_nonzero(edges)),
            "candidate_edge_count": int(np.count_nonzero(candidate_edges)),
            "component_count": int(component_count),
            "component_kept_count": int(component_kept_count),
            "component_reject_count": int(component_reject_count),
            "component_filtered_count": int(component_filtered_count),
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
            lower_focus_poly=lower_focus_poly,
            lane_segs=lane_segs,
            left_fit=left_fit,
            right_fit=right_fit,
            center_pts=center_pts_img,
            ys=ys,
            confidence=confidence,
            hsv_vis=hsv_vis,
            red_mask_raw=red_mask_raw,
            red_mask_clean=red_mask_clean,
            used_binary=used_binary,
            edges=edges,
            candidate_edges=candidate_edges,
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
        min_w = max(18.0, 0.40 * expected)
        max_w = min(0.98 * self.frame_width, 2.35 * expected)
        return float(min_w), float(max_w)

    def _filter_segments_by_cluster(
        self,
        segs: List[Tuple[int, int, int, int]],
        roi_w: int,
    ) -> List[Tuple[int, int, int, int]]:
        if len(segs) <= 3:
            return segs

        x_mids = np.array(
            [0.5 * (x1 + x2) for x1, _, x2, _ in segs],
            dtype=np.float32,
        )
        x_med = float(np.median(x_mids))
        tol = max(18.0, 0.12 * roi_w)

        filtered = []
        for seg in segs:
            x1, y1, x2, y2 = seg
            x_mid = 0.5 * (x1 + x2)
            if abs(x_mid - x_med) <= tol:
                filtered.append(seg)

        if len(filtered) < max(2, len(segs) // 3):
            return segs

        return filtered

    def _filter_candidate_components(
        self,
        candidate_mask: np.ndarray,
        roi_h: int,
        roi_w: int,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)

        filtered = np.zeros_like(candidate_mask)

        kept_count = 0
        rejected_count = 0

        max_area = max(1, int(self.max_component_area_frac * roi_h * roi_w))
        min_height = max(1, int(self.min_component_height_frac * roi_h))
        max_width = max(1, int(self.max_component_width_frac * roi_w))

        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])

            if area < self.min_component_area:
                rejected_count += 1
                continue
            if area > max_area:
                rejected_count += 1
                continue
            if h < min_height:
                rejected_count += 1
                continue
            if w > max_width:
                rejected_count += 1
                continue

            aspect = float(max(h, w)) / max(1.0, float(min(h, w)))
            if aspect < self.min_component_aspect_ratio:
                rejected_count += 1
                continue

            component_mask = (labels == label).astype(np.uint8) * 255

            left_overlap = int(np.count_nonzero(cv2.bitwise_and(component_mask, left_mask)))
            right_overlap = int(np.count_nonzero(cv2.bitwise_and(component_mask, right_mask)))

            if left_overlap == 0 and right_overlap == 0:
                rejected_count += 1
                continue

            filtered[labels == label] = 255
            kept_count += 1

        debug = {
            "component_count": int(max(0, num_labels - 1)),
            "component_kept_count": int(kept_count),
            "component_reject_count": int(rejected_count),
            "component_filtered_count": int(np.count_nonzero(filtered)),
        }

        return filtered, debug

    def _preprocess_roi(
        self,
        roi: np.ndarray,
        roi_h: int,
        roi_w: int,
    ) -> dict:
        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([self.red_h1_low, self.red_s_low, self.red_v_low], dtype=np.uint8)
        upper_red1 = np.array([self.red_h1_high, 255, 255], dtype=np.uint8)

        lower_red2 = np.array([self.red_h2_low, self.red_s_low, self.red_v_low], dtype=np.uint8)
        upper_red2 = np.array([self.red_h2_high, 255, 255], dtype=np.uint8)

        lower_pink = np.array([self.pink_h_low, self.pink_s_low, self.pink_v_low], dtype=np.uint8)
        upper_pink = np.array([self.pink_h_high, 255, 255], dtype=np.uint8)

        red_mask_1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask_2 = cv2.inRange(hsv, lower_red2, upper_red2)
        pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)

        red_mask_main = cv2.bitwise_or(red_mask_1, red_mask_2)
        red_mask_main = cv2.bitwise_or(red_mask_main, pink_mask)

        lower_shadow_red1 = np.array(
            [self.shadow_red_h1_low, self.shadow_red_s_low, self.shadow_red_v_low],
            dtype=np.uint8,
        )
        upper_shadow_red1 = np.array(
            [self.shadow_red_h1_high, 255, 255],
            dtype=np.uint8,
        )

        lower_shadow_red2 = np.array(
            [self.shadow_red_h2_low, self.shadow_red_s_low, self.shadow_red_v_low],
            dtype=np.uint8,
        )
        upper_shadow_red2 = np.array(
            [self.shadow_red_h2_high, 255, 255],
            dtype=np.uint8,
        )

        shadow_red_mask_1 = cv2.inRange(hsv, lower_shadow_red1, upper_shadow_red1)
        shadow_red_mask_2 = cv2.inRange(hsv, lower_shadow_red2, upper_shadow_red2)
        shadow_red_mask = cv2.bitwise_or(shadow_red_mask_1, shadow_red_mask_2)
        shadow_red_mask = cv2.medianBlur(shadow_red_mask, 3)

        roi_mask = np.zeros_like(red_mask_main)
        roi_poly = np.array([[(
            int(0.00 * roi_w), roi_h - 1
        ), (
            int(1.00 * roi_w), roi_h - 1
        ), (
            int(0.76 * roi_w), int(0.14 * roi_h)
        ), (
            int(0.24 * roi_w), int(0.14 * roi_h)
        )]], dtype=np.int32)
        cv2.fillPoly(roi_mask, roi_poly, 255)

        lower_focus_mask = np.zeros_like(roi_mask)
        lower_focus_poly = np.array([[(
            int(0.00 * roi_w), roi_h - 1
        ), (
            int(1.00 * roi_w), roi_h - 1
        ), (
            int(0.72 * roi_w), int(0.32 * roi_h)
        ), (
            int(0.28 * roi_w), int(0.32 * roi_h)
        )]], dtype=np.int32)
        cv2.fillPoly(lower_focus_mask, lower_focus_poly, 255)

        hood_exclusion_mask = np.zeros_like(roi_mask)
        hood_poly = np.array([[(
            int(0.28 * roi_w), roi_h - 1
        ), (
            int(0.72 * roi_w), roi_h - 1
        ), (
            int(0.66 * roi_w), int(0.72 * roi_h)
        ), (
            int(0.34 * roi_w), int(0.72 * roi_h)
        )]], dtype=np.int32)
        cv2.fillPoly(hood_exclusion_mask, hood_poly, 255)

        roi_mask = cv2.bitwise_and(roi_mask, cv2.bitwise_not(hood_exclusion_mask))

        left_mask = np.zeros_like(red_mask_main)
        left_poly = np.array([[(
            int(0.00 * roi_w), roi_h - 1
        ), (
            int(0.53 * roi_w), roi_h - 1
        ), (
            int(0.45 * roi_w), int(0.20 * roi_h)
        ), (
            int(0.18 * roi_w), int(0.20 * roi_h)
        )]], dtype=np.int32)
        cv2.fillPoly(left_mask, left_poly, 255)

        right_mask = np.zeros_like(red_mask_main)
        right_poly = np.array([[(
            int(0.47 * roi_w), roi_h - 1
        ), (
            int(1.00 * roi_w), roi_h - 1
        ), (
            int(0.82 * roi_w), int(0.20 * roi_h)
        ), (
            int(0.55 * roi_w), int(0.20 * roi_h)
        )]], dtype=np.int32)
        cv2.fillPoly(right_mask, right_poly, 255)

        left_mask = cv2.bitwise_and(left_mask, roi_mask)
        right_mask = cv2.bitwise_and(right_mask, roi_mask)

        red_mask_main_roi = cv2.bitwise_and(red_mask_main, roi_mask)
        shadow_red_mask_roi = cv2.bitwise_and(shadow_red_mask, roi_mask)

        red_mask_main_dilated = cv2.dilate(
            red_mask_main_roi,
            np.ones((9, 9), np.uint8),
            iterations=1,
        )
        shadow_red_mask_roi = cv2.bitwise_and(shadow_red_mask_roi, red_mask_main_dilated)

        red_mask_raw = cv2.bitwise_or(red_mask_main_roi, shadow_red_mask_roi)

        red_mask_raw = cv2.morphologyEx(
            red_mask_raw,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )

        red_mask_clean = cv2.morphologyEx(
            red_mask_raw,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8),
        )
        red_mask_clean = cv2.medianBlur(red_mask_clean, 5)
        red_mask_clean = cv2.dilate(
            red_mask_clean,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )

        candidate_mask = cv2.bitwise_and(red_mask_clean, roi_mask)
        candidate_mask = cv2.bitwise_and(candidate_mask, lower_focus_mask)

        component_debug = {
            "component_count": 0,
            "component_kept_count": 0,
            "component_reject_count": 0,
            "component_filtered_count": int(np.count_nonzero(candidate_mask)),
        }

        if self.use_component_filter:
            filtered_mask, component_debug = self._filter_candidate_components(
                candidate_mask=candidate_mask,
                roi_h=roi_h,
                roi_w=roi_w,
                left_mask=left_mask,
                right_mask=right_mask,
            )

            if np.count_nonzero(filtered_mask) > max(40, int(0.05 * max(1, np.count_nonzero(candidate_mask)))):
                used_binary = filtered_mask
            else:
                used_binary = candidate_mask.copy()
        else:
            used_binary = candidate_mask.copy()

        used_binary = cv2.morphologyEx(
            used_binary,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8),
        )

        edges = cv2.Canny(used_binary, self.canny_low, self.canny_high)
        candidate_edges = cv2.bitwise_and(edges, roi_mask)
        candidate_edges = cv2.bitwise_and(candidate_edges, lower_focus_mask)

        hsv_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return {
            "hsv_vis": hsv_vis,
            "red_mask_raw": red_mask_raw,
            "red_mask_clean": red_mask_clean,
            "used_binary": used_binary,
            "edges": edges,
            "candidate_edges": candidate_edges,
            "roi_poly": roi_poly,
            "hood_poly": hood_poly,
            "left_mask": left_mask,
            "right_mask": right_mask,
            "left_poly": left_poly,
            "right_poly": right_poly,
            "lower_focus_poly": lower_focus_poly,
            **component_debug,
        }

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

            if self.prev_left_fit is not None:
                left_target = self._poly2_x(*self.prev_left_fit, y)
            else:
                left_target = left_target_default

            if self.prev_right_fit is not None:
                right_target = self._poly2_x(*self.prev_right_fit, y)
            else:
                right_target = right_target_default

            max_row_jump = 45.0 + 55.0 * y_norm

            left_candidates = np.where((row > 0) & (left_mask[y] > 0))[0]
            if left_candidates.size >= 3:
                target_for_left = left_target if prev_left_row_x is None else 0.7 * prev_left_row_x + 0.3 * left_target
                x_left = self._choose_run_center(left_candidates, target_for_left)
                if x_left is not None and abs(x_left - target_for_left) <= max_row_jump:
                    left_pts.append((x_left, int(y)))
                    prev_left_row_x = x_left

            right_candidates = np.where((row > 0) & (right_mask[y] > 0))[0]
            if right_candidates.size >= 3:
                target_for_right = right_target if prev_right_row_x is None else 0.7 * prev_right_row_x + 0.3 * right_target
                x_right = self._choose_run_center(right_candidates, target_for_right)
                if x_right is not None and abs(x_right - target_for_right) <= max_row_jump:
                    right_pts.append((x_right, int(y)))
                    prev_right_row_x = x_right

        return left_pts, right_pts

    def _extract_lane_points(
        self,
        candidate_edges: np.ndarray,
        used_binary: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
        roi_w: int,
        roi_h: int,
    ) -> Tuple[
        List[Tuple[int, int]],
        List[Tuple[int, int]],
        List[Tuple[int, int, int, int]],
        dict,
    ]:
        lines = cv2.HoughLinesP(
            candidate_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )

        left_lane_segs: List[Tuple[int, int, int, int]] = []
        right_lane_segs: List[Tuple[int, int, int, int]] = []

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

                if -6.0 < slope < -0.10:
                    if x_mid < int(0.66 * roi_w):
                        if left_mask[y_mid, x_mid] != 0:
                            left_lane_segs.append((x1, y1, x2, y2))
                        else:
                            left_reject_mask_count += 1

                elif 0.10 < slope < 6.0:
                    if x_mid > int(0.34 * roi_w):
                        if right_mask[y_mid, x_mid] != 0:
                            right_lane_segs.append((x1, y1, x2, y2))
                        else:
                            right_reject_mask_count += 1

        left_lane_segs = self._filter_segments_by_cluster(left_lane_segs, roi_w=roi_w)
        right_lane_segs = self._filter_segments_by_cluster(right_lane_segs, roi_w=roi_w)

        left_pts: List[Tuple[int, int]] = []
        right_pts: List[Tuple[int, int]] = []
        lane_segs: List[Tuple[int, int, int, int]] = []

        for x1, y1, x2, y2 in left_lane_segs:
            left_pts.append((x1, y1))
            left_pts.append((x2, y2))
            lane_segs.append((x1, y1, x2, y2))

        for x1, y1, x2, y2 in right_lane_segs:
            right_pts.append((x1, y1))
            right_pts.append((x2, y2))
            lane_segs.append((x1, y1, x2, y2))

        debug_counts = {
            "hough_count": 0 if lines is None else int(len(lines)),
            "left_seg_count": int(len(left_lane_segs)),
            "right_seg_count": int(len(right_lane_segs)),
            "left_raw_pt_count": int(len(left_pts)),
            "right_raw_pt_count": int(len(right_pts)),
            "left_reject_mask_count": int(left_reject_mask_count),
            "right_reject_mask_count": int(right_reject_mask_count),
        }

        if len(left_pts) < 6 or len(right_pts) < 6:
            fb_left, fb_right = self._extract_lane_points_from_binary(
                binary=used_binary,
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
        else:
            debug_counts["fallback_used"] = 0
            debug_counts["fallback_left_pts"] = 0
            debug_counts["fallback_right_pts"] = 0

        return left_pts, right_pts, lane_segs, debug_counts

    def _build_sample_ys(self, roi_h: int) -> np.ndarray:
        return np.linspace(int(0.18 * roi_h), roi_h - 1, 60)

    def _build_binary_sample_rows(self, roi_h: int) -> np.ndarray:
        y_top = int(0.18 * roi_h)
        y_bottom = roi_h - 1

        t = np.linspace(0.0, 1.0, 72)
        ys = y_top + (y_bottom - y_top) * (t ** 0.55)
        ys = np.clip(np.round(ys).astype(int), y_top, y_bottom)

        return np.unique(ys)

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

    def _append_center_from_single_side(
        self,
        center_pts_img: List[Tuple[float, float]],
        fit: Tuple[float, float, float],
        ys: np.ndarray,
        roi_w: int,
        roi_h: int,
        side: str,
    ) -> None:
        for yy in ys:
            x_lane = self._poly2_x(*fit, yy)
            lane_half_width_px = 0.5 * self._expected_lane_width_px(yy, roi_h)

            if side == "left":
                xc = x_lane + lane_half_width_px
            else:
                xc = x_lane - lane_half_width_px

            if -0.10 * roi_w <= xc <= 1.10 * roi_w:
                center_pts_img.append((xc, yy + self.roi_y0))

    def _build_centerline_points(
        self,
        left_fit: Optional[Tuple[float, float, float]],
        right_fit: Optional[Tuple[float, float, float]],
        ys: np.ndarray,
        roi_w: int,
        roi_h: int,
    ) -> Tuple[List[Tuple[float, float]], str, dict]:
        center_pts_img: List[Tuple[float, float]] = []

        lane_pair_debug = {
            "lane_pair_valid": 0,
            "lane_width_min_px": 0.0,
            "lane_width_max_px": 0.0,
            "lane_width_mean_px": 0.0,
            "lane_width_std_px": 0.0,
            "lane_width_reason": "not_checked",
            "lane_pair_valid_rows": 0,
            "lane_pair_total_rows": int(len(ys)),
            "lane_crossed_rows": 0,
        }

        if left_fit is not None and right_fit is not None:
            pair_valid, lane_pair_debug_base = self._validate_lane_pair(
                left_fit=left_fit,
                right_fit=right_fit,
                ys=ys,
                roi_h=roi_h,
                roi_w=roi_w,
            )
            lane_pair_debug.update(lane_pair_debug_base)

            if pair_valid:
                mids = []
                mids_ys = []

                for yy in ys:
                    xl = self._poly2_x(*left_fit, yy)
                    xr = self._poly2_x(*right_fit, yy)
                    width = xr - xl

                    min_w, max_w = self._width_bounds_px(yy, roi_h)
                    if min_w <= width <= max_w:
                        mids.append(0.5 * (xl + xr))
                        mids_ys.append(yy)

                if len(mids) >= self.min_points:
                    mids = self._smooth_1d(mids, window=7)
                    for xc, yy in zip(mids, mids_ys):
                        if -0.10 * roi_w <= xc <= 1.10 * roi_w:
                            center_pts_img.append((float(xc), float(yy + self.roi_y0)))
                    return center_pts_img, "both", lane_pair_debug

        if left_fit is not None:
            self._append_center_from_single_side(
                center_pts_img=center_pts_img,
                fit=left_fit,
                ys=ys,
                roi_w=roi_w,
                roi_h=roi_h,
                side="left",
            )
            return center_pts_img, "left_only", lane_pair_debug

        if right_fit is not None:
            self._append_center_from_single_side(
                center_pts_img=center_pts_img,
                fit=right_fit,
                ys=ys,
                roi_w=roi_w,
                roi_h=roi_h,
                side="right",
            )
            return center_pts_img, "right_only", lane_pair_debug

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

        xs = np.array(self._smooth_1d(xs.tolist(), window=5), dtype=np.float64)

        if len(prev_center_pts_img) == len(center_pts_img) and len(prev_center_pts_img) >= 3:
            prev_xs = np.array([p[0] for p in prev_center_pts_img], dtype=np.float64)
            xs = self.temporal_alpha_center * prev_xs + (1.0 - self.temporal_alpha_center) * xs

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
            lane_pair_valid == 1 or fit_mode in ("left_only", "right_only", "both")
        )

        if current_good:
            return center_pts_img, {
                "hold_used": hold_used,
                "hold_reason": hold_reason,
                "fit_mode_after_hold": fit_mode_after_hold,
                "frames_since_good_pair": int(self.frames_since_good_pair),
            }

        if (
            0 < len(self.prev_good_center_pts_img)
            and self.frames_since_good_pair <= self.hold_last_good_frames
        ):
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
        valid_rows = 0
        crossed_rows = 0

        for yy in ys:
            xl = self._poly2_x(*left_fit, yy)
            xr = self._poly2_x(*right_fit, yy)
            width = float(xr - xl)
            widths.append(width)

            if width <= 0.0:
                crossed_rows += 1

            min_w, max_w = self._width_bounds_px(yy, roi_h)
            if min_w < width < max_w:
                valid_rows += 1

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
            }

        widths_smooth = np.array(self._smooth_1d(widths_arr.tolist(), window=7), dtype=np.float64)

        min_w = float(np.min(widths_smooth))
        max_w = float(np.max(widths_smooth))
        mean_w = float(np.mean(widths_smooth))
        std_w = float(np.std(widths_smooth))

        total_rows = max(1, len(ys))
        crossed_frac = float(crossed_rows / total_rows)
        valid_frac = float(valid_rows / total_rows)

        valid = True
        reason = "ok"

        if crossed_frac > self.max_crossed_fraction:
            valid = False
            reason = "crossed_often"
        elif valid_frac < self.min_pair_valid_fraction:
            valid = False
            reason = "too_few_valid_width_rows"
        elif std_w > self.max_width_std_px and (min_w < 20.0 or max_w > 0.92 * roi_w):
            valid = False
            reason = "too_inconsistent"

        return valid, {
            "lane_pair_valid": int(valid),
            "lane_width_min_px": min_w,
            "lane_width_max_px": max_w,
            "lane_width_mean_px": mean_w,
            "lane_width_std_px": std_w,
            "lane_width_reason": reason,
            "lane_pair_valid_rows": int(valid_rows),
            "lane_pair_total_rows": int(total_rows),
            "lane_crossed_rows": int(crossed_rows),
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

    # =========================================================
    # Confidence
    # =========================================================
    def _compute_confidence(
        self,
        left_fit: Optional[Tuple[float, float, float]],
        right_fit: Optional[Tuple[float, float, float]],
        center_pts: List[Tuple[float, float]],
        lane_segs: List[Tuple[int, int, int, int]],
        fit_mode: str,
        lane_pair_valid: int,
    ) -> float:
        if len(center_pts) < self.min_points:
            return 0.0

        seg_score = min(1.0, len(lane_segs) / 20.0)
        pt_score = min(1.0, len(center_pts) / 40.0)

        if fit_mode == "both":
            fit_score = 1.0
        elif fit_mode == "held_last_good":
            fit_score = 0.72
        elif fit_mode in ("left_only", "right_only"):
            fit_score = 0.62
        else:
            fit_score = 0.0

        side_count_score = 0.0
        if left_fit is not None and right_fit is not None:
            side_count_score = 1.0
        elif left_fit is not None or right_fit is not None:
            side_count_score = 0.6

        confidence = (
            0.42 * fit_score
            + 0.22 * seg_score
            + 0.18 * pt_score
            + 0.18 * side_count_score
        )

        if fit_mode == "both" and lane_pair_valid == 0:
            confidence *= 0.65

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
        lower_focus_poly: np.ndarray,
        lane_segs: List[Tuple[int, int, int, int]],
        left_fit: Optional[Tuple[float, float, float]],
        right_fit: Optional[Tuple[float, float, float]],
        center_pts: List[Tuple[float, float]],
        ys: np.ndarray,
        confidence: float,
        hsv_vis: np.ndarray,
        red_mask_raw: np.ndarray,
        red_mask_clean: np.ndarray,
        used_binary: np.ndarray,
        edges: np.ndarray,
        candidate_edges: np.ndarray,
    ) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]

        cv2.rectangle(vis, (0, roi_y0), (w - 1, h - 1), (0, 255, 0), 1)

        poly_vis = roi_poly.copy()
        poly_vis[:, :, 1] += roi_y0
        cv2.polylines(vis, [poly_vis], True, (0, 255, 255), 1)

        hood_poly_vis = hood_poly.copy()
        hood_poly_vis[:, :, 1] += roi_y0
        cv2.polylines(vis, [hood_poly_vis], True, (0, 0, 255), 1)

        lower_focus_poly_vis = lower_focus_poly.copy()
        lower_focus_poly_vis[:, :, 1] += roi_y0
        cv2.polylines(vis, [lower_focus_poly_vis], True, (0, 180, 255), 1)

        left_poly_vis = left_poly.copy()
        left_poly_vis[:, :, 1] += roi_y0
        cv2.polylines(vis, [left_poly_vis], True, (255, 0, 255), 1)

        right_poly_vis = right_poly.copy()
        right_poly_vis[:, :, 1] += roi_y0
        cv2.polylines(vis, [right_poly_vis], True, (255, 255, 0), 1)

        for x1, y1, x2, y2 in lane_segs:
            cv2.line(vis, (x1, y1 + roi_y0), (x2, y2 + roi_y0), (255, 0, 0), 2)

        if left_fit is not None:
            pts = []
            for yy in ys:
                xx = self._poly2_x(*left_fit, yy)
                pts.append((int(xx), int(yy) + roi_y0))
            if len(pts) > 1:
                cv2.polylines(vis, [np.array(pts, dtype=np.int32)], False, (0, 255, 255), 2)

        if right_fit is not None:
            pts = []
            for yy in ys:
                xx = self._poly2_x(*right_fit, yy)
                pts.append((int(xx), int(yy) + roi_y0))
            if len(pts) > 1:
                cv2.polylines(vis, [np.array(pts, dtype=np.int32)], False, (0, 255, 255), 2)

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
        used_bgr = cv2.cvtColor(used_binary, cv2.COLOR_GRAY2BGR)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        candidate_edges_bgr = cv2.cvtColor(candidate_edges, cv2.COLOR_GRAY2BGR)

        thumbs = [
            ("hsv_vis", cv2.resize(hsv_vis, (thumb_w, thumb_h))),
            ("red_raw", cv2.resize(red_mask_raw_bgr, (thumb_w, thumb_h))),
            ("red_clean", cv2.resize(red_mask_clean_bgr, (thumb_w, thumb_h))),
            ("used_binary", cv2.resize(used_bgr, (thumb_w, thumb_h))),
            ("edges", cv2.resize(edges_bgr, (thumb_w, thumb_h))),
            ("cand_edges", cv2.resize(candidate_edges_bgr, (thumb_w, thumb_h))),
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