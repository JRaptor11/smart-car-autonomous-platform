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
    Simplified baseline lane detector with two modes:

    1) Real image mode:
       - detect(frame=...) runs basic OpenCV preprocessing
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
        roi_y0: int = 205,
        min_points: int = 12,
        show_debug: bool = False,
        lane_width_m: float = 0.9,
        x_m_per_px: float = 0.03,
        y_m_per_px: float = 0.01,
        dark_thresh: int = 75,
        canny_low: int = 45,
        canny_high: int = 135,
        blackhat_kernel_w: int = 11,
        blackhat_kernel_h: int = 11,
        blackhat_thresh: int = 32,

        use_component_filter: bool = True,
        min_component_area: int = 10,
        max_component_area_frac: float = 0.28,
        min_component_aspect_ratio: float = 0.55,
        min_component_height_frac: float = 0.04,
        max_component_width_frac: float = 0.50,
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

        # Simple preprocessing params
        self.dark_thresh = int(dark_thresh)
        self.canny_low = int(canny_low)
        self.canny_high = int(canny_high)
        self.blackhat_kernel_w = int(blackhat_kernel_w)
        self.blackhat_kernel_h = int(blackhat_kernel_h)
        self.blackhat_thresh = int(blackhat_thresh)

        self.use_component_filter = bool(use_component_filter)
        self.min_component_area = int(min_component_area)
        self.max_component_area_frac = float(max_component_area_frac)
        self.min_component_aspect_ratio = float(min_component_aspect_ratio)
        self.min_component_height_frac = float(min_component_height_frac)
        self.max_component_width_frac = float(max_component_width_frac)

    # =========================================================
    # Public API
    # =========================================================
    def detect(self, frame=None, vehicle_x: float = 0.0) -> LaneDetection:
        if frame is not None:
            try:
                return self._detect_from_frame(frame)
            except Exception as e:
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

        return self._detect_simulated(vehicle_x=vehicle_x)

    # =========================================================
    # Simulated Mode
    # =========================================================
    def _detect_simulated(self, vehicle_x: float = 0.0) -> LaneDetection:
        if random.random() < self.dropout_prob:
            return LaneDetection(
                centerline_xs=[],
                centerline_ys=[],
                confidence=0.0,
                debug_frame=None,
                debug_meta={
                    "mode": "sim",
                    "fit_mode": "none",
                    "left_fit_ok": "",
                    "right_fit_ok": "",
                    "lane_seg_count": "",
                    "center_pt_count": 0,
                },
            )

        if self.path_xs is None or self.path_ys is None:
            return LaneDetection(
                centerline_xs=[],
                centerline_ys=[],
                confidence=0.0,
                debug_frame=None,
                debug_meta={
                    "mode": "sim",
                    "fit_mode": "none",
                    "left_fit_ok": "",
                    "right_fit_ok": "",
                    "lane_seg_count": "",
                    "center_pt_count": 0,
                },
            )

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
            return LaneDetection(
                centerline_xs=[],
                centerline_ys=[],
                confidence=0.0,
                debug_frame=None,
                debug_meta={
                    "mode": "sim",
                    "fit_mode": "none",
                    "left_fit_ok": "",
                    "right_fit_ok": "",
                    "lane_seg_count": "",
                    "center_pt_count": 0,
                },
            )

        confidence = 1.0 - self.dropout_prob

        return LaneDetection(
            centerline_xs=xs,
            centerline_ys=ys,
            confidence=confidence,
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

        gray_eq = preprocess["gray_eq"]
        dark_mask = preprocess["dark_mask"]
        blackhat = preprocess["blackhat"]
        blackhat_mask = preprocess["blackhat_mask"]
        combined_binary = preprocess["combined_binary"]
        used_binary = preprocess["used_binary"]
        edges = preprocess["edges"]
        candidate_edges = preprocess["candidate_edges"]
        roi_poly = preprocess["roi_poly"]
        left_mask = preprocess["left_mask"]
        right_mask = preprocess["right_mask"]
        left_poly = preprocess["left_poly"]
        right_poly = preprocess["right_poly"]
        component_count = preprocess["component_count"]
        component_kept_count = preprocess["component_kept_count"]
        component_reject_count = preprocess["component_reject_count"]
        component_filtered_count = preprocess["component_filtered_count"]

        left_pts, right_pts, lane_segs, seg_debug = self._extract_lane_points(
            candidate_edges=candidate_edges,
            left_mask=left_mask,
            right_mask=right_mask,
            roi_w=roi_w,
            roi_h=roi_h,
        )

        left_fit = self._poly2_fit_x_of_y(left_pts)
        right_fit = self._poly2_fit_x_of_y(right_pts)

        ys = self._build_sample_ys(roi_h)
        center_pts_img, fit_mode, lane_pair_debug = self._build_centerline_points(
            left_fit=left_fit,
            right_fit=right_fit,
            ys=ys,
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
        )

        if len(centerline_xs) < 2:
            confidence = 0.0

        if fit_mode == "pair_rejected":
            confidence = 0.0
        elif lane_pair_debug.get("lane_pair_valid", 1) == 0 and left_fit is not None and right_fit is not None:
            confidence *= 0.5

        debug_meta = {
            "mode": "real",
            "fit_mode": fit_mode,
            "left_fit_ok": int(left_fit is not None),
            "right_fit_ok": int(right_fit is not None),
            "lane_seg_count": int(len(lane_segs)),
            "center_pt_count": int(len(center_pts_img)),
            "roi_y0": int(self.roi_y0),
            "frame_width": int(w),
            "frame_height": int(h),
            "dark_thresh": int(self.dark_thresh),
            "blackhat_thresh": int(self.blackhat_thresh),
            "canny_low": int(self.canny_low),
            "canny_high": int(self.canny_high),
            "dark_mask_count": int(np.count_nonzero(dark_mask)),
            "blackhat_mask_count": int(np.count_nonzero(blackhat_mask)),
            "combined_binary_count": int(np.count_nonzero(combined_binary)),
            "used_binary_count": int(np.count_nonzero(used_binary)),
            "edge_count": int(np.count_nonzero(edges)),
            "candidate_edge_count": int(np.count_nonzero(candidate_edges)),
            "component_count": int(component_count),
            "component_kept_count": int(component_kept_count),
            "component_reject_count": int(component_reject_count),
            "component_filtered_count": int(component_filtered_count),
            **seg_debug,
            **lane_pair_debug,
        }

        debug_frame = self._build_debug_view(
            frame=frame,
            roi_y0=self.roi_y0,
            roi_poly=roi_poly,
            left_poly=left_poly,
            right_poly=right_poly,
            lane_segs=lane_segs,
            left_fit=left_fit,
            right_fit=right_fit,
            center_pts=center_pts_img,
            ys=ys,
            confidence=confidence,
            gray_eq=gray_eq,
            dark_mask=dark_mask,
            blackhat=blackhat,
            blackhat_mask=blackhat_mask,
            used_binary=used_binary,
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

            aspect = float(h) / max(float(w), 1.0)
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
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)

        # -----------------------------
        # Basic dark mask
        # -----------------------------
        dark_mask = cv2.inRange(blur, 0, self.dark_thresh)

        # -----------------------------
        # Basic blackhat mask
        # -----------------------------
        kh = max(3, self.blackhat_kernel_h | 1)
        kw = max(3, self.blackhat_kernel_w | 1)
        blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, blackhat_kernel)

        _, blackhat_mask = cv2.threshold(
            blackhat,
            self.blackhat_thresh,
            255,
            cv2.THRESH_BINARY,
        )

        # -----------------------------
        # Initial candidate mask
        # -----------------------------
        # Start from the darker, more literal tape candidate
        candidate_mask = dark_mask.copy()

        # Only allow blackhat to contribute where it also overlaps
        # a slightly expanded dark region. This reduces blackhat clutter.
        dark_support = cv2.dilate(dark_mask, np.ones((3, 3), np.uint8), iterations=1)
        blackhat_supported = cv2.bitwise_and(blackhat_mask, dark_support)

        candidate_mask = cv2.bitwise_or(candidate_mask, blackhat_supported)

        candidate_mask = cv2.morphologyEx(
            candidate_mask,
            cv2.MORPH_OPEN,
            np.ones((2, 2), np.uint8),
        )

        candidate_mask = cv2.morphologyEx(
            candidate_mask,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8),
        )
        
        candidate_mask = cv2.erode(
            candidate_mask,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )
        
        # -----------------------------
        # Main trapezoid ROI mask
        # -----------------------------
        roi_mask = np.zeros_like(candidate_mask)
        roi_poly = np.array([[
            (int(0.00 * roi_w), roi_h - 1),
            (int(1.00 * roi_w), roi_h - 1),
            (int(0.58 * roi_w), int(0.40 * roi_h)),
            (int(0.42 * roi_w), int(0.40 * roi_h)),
        ]], dtype=np.int32)
        cv2.fillPoly(roi_mask, roi_poly, 255)

        candidate_mask = cv2.bitwise_and(candidate_mask, roi_mask)
        # candidate_mask = cv2.bitwise_and(candidate_mask, lower_focus_mask)

        # -----------------------------
        # Left and right search masks
        # -----------------------------
        left_mask = np.zeros_like(candidate_mask)
        left_poly = np.array([[
            (int(0.00 * roi_w), roi_h - 1),
            (int(0.50 * roi_w), roi_h - 1),
            (int(0.50 * roi_w), int(0.42 * roi_h)),
            (int(0.20 * roi_w), int(0.42 * roi_h)),
        ]], dtype=np.int32)
        cv2.fillPoly(left_mask, left_poly, 255)

        right_mask = np.zeros_like(candidate_mask)
        right_poly = np.array([[
            (int(0.50 * roi_w), roi_h - 1),
            (int(1.00 * roi_w), roi_h - 1),
            (int(0.80 * roi_w), int(0.42 * roi_h)),
            (int(0.50 * roi_w), int(0.42 * roi_h)),
        ]], dtype=np.int32)
        cv2.fillPoly(right_mask, right_poly, 255)

        left_mask = cv2.bitwise_and(left_mask, roi_mask)
        right_mask = cv2.bitwise_and(right_mask, roi_mask)

        # -----------------------------
        # Lower-ROI preference mask
        # -----------------------------
        lower_focus_mask = np.zeros_like(candidate_mask)
        lower_focus_poly = np.array([[
            (int(0.00 * roi_w), roi_h - 1),
            (int(1.00 * roi_w), roi_h - 1),
            (int(0.70 * roi_w), int(0.20 * roi_h)),
            (int(0.30 * roi_w), int(0.20 * roi_h)),
        ]], dtype=np.int32)
        cv2.fillPoly(lower_focus_mask, lower_focus_poly, 255)

        lower_focus_mask = cv2.bitwise_and(lower_focus_mask, roi_mask)

        # -----------------------------
        # Connected-component filtering
        # -----------------------------
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

            # Safety fallback: if filtering removes too much,
            # fall back to the original candidate mask.
            if np.count_nonzero(filtered_mask) > max(40, int(0.05 * np.count_nonzero(candidate_mask))):
                used_binary = filtered_mask
            else:
                used_binary = candidate_mask.copy()
        else:
            used_binary = candidate_mask.copy()

        # Small reconnect after filtering
        used_binary = cv2.morphologyEx(
            used_binary,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
        )

        # -----------------------------
        # Edges from cleaned mask support
        # -----------------------------
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)

        used_binary_for_edges = cv2.dilate(
            used_binary,
            np.ones((5, 5), np.uint8),
            iterations=1,
        )

        candidate_edges = cv2.bitwise_and(edges, used_binary_for_edges)
        candidate_edges = cv2.bitwise_and(candidate_edges, roi_mask)

        return {
            "gray_eq": gray_eq,
            "dark_mask": dark_mask,
            "blackhat": blackhat,
            "blackhat_mask": blackhat_mask,
            "edges": edges,
            "combined_binary": candidate_mask,
            "used_binary": used_binary,
            "candidate_edges": candidate_edges,
            "roi_poly": roi_poly,
            "left_mask": left_mask,
            "right_mask": right_mask,
            "left_poly": left_poly,
            "right_poly": right_poly,
            **component_debug,
        }

    def _extract_lane_points(
        self,
        candidate_edges: np.ndarray,
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
            threshold=20,
            minLineLength=22,
            maxLineGap=10,
        )

        left_pts: List[Tuple[int, int]] = []
        right_pts: List[Tuple[int, int]] = []
        lane_segs: List[Tuple[int, int, int, int]] = []

        left_seg_count = 0
        right_seg_count = 0
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

                if length < 14:
                    continue
                if y_span < 5:
                    continue
                if y_mid < int(0.18 * roi_h):
                    continue
                if not (0 <= x_mid < roi_w and 0 <= y_mid < roi_h):
                    continue

                # Left lane candidate
                if -3.5 < slope < -0.30:
                    if x_mid < int(0.60 * roi_w):
                        if left_mask[y_mid, x_mid] != 0:
                            left_pts.append((x1, y1))
                            left_pts.append((x2, y2))
                            lane_segs.append((x1, y1, x2, y2))
                            left_seg_count += 1
                        else:
                            left_reject_mask_count += 1

                # Right lane candidate
                elif 0.30 < slope < 3.5:
                    if x_mid > int(0.40 * roi_w):
                        if right_mask[y_mid, x_mid] != 0:
                            right_pts.append((x1, y1))
                            right_pts.append((x2, y2))
                            lane_segs.append((x1, y1, x2, y2))
                            right_seg_count += 1
                        else:
                            right_reject_mask_count += 1

        debug_counts = {
            "hough_count": 0 if lines is None else int(len(lines)),
            "left_seg_count": int(left_seg_count),
            "right_seg_count": int(right_seg_count),
            "left_raw_pt_count": int(len(left_pts)),
            "right_raw_pt_count": int(len(right_pts)),
            "left_reject_mask_count": int(left_reject_mask_count),
            "right_reject_mask_count": int(right_reject_mask_count),
        }

        return left_pts, right_pts, lane_segs, debug_counts

    def _build_sample_ys(self, roi_h: int) -> np.ndarray:
        return np.linspace(int(0.42 * roi_h), roi_h - 1, 60)

    def _build_centerline_points(
        self,
        left_fit: Optional[Tuple[float, float, float]],
        right_fit: Optional[Tuple[float, float, float]],
        ys: np.ndarray,
    ) -> Tuple[List[Tuple[float, float]], str, dict]:
        center_pts_img: List[Tuple[float, float]] = []

        lane_pair_debug = {
            "lane_pair_valid": 0,
            "lane_width_min_px": 0.0,
            "lane_width_max_px": 0.0,
            "lane_width_mean_px": 0.0,
            "lane_width_std_px": 0.0,
            "lane_width_reason": "not_checked",
        }

        if left_fit is not None and right_fit is not None:
            pair_valid, lane_pair_debug = self._validate_lane_pair(left_fit, right_fit, ys)

            if pair_valid:
                fit_mode = "both"
                for yy in ys:
                    xl = self._poly2_x(*left_fit, yy)
                    xr = self._poly2_x(*right_fit, yy)
                    xc = 0.5 * (xl + xr)
                    center_pts_img.append((xc, yy + self.roi_y0))
            else:
                fit_mode = "pair_rejected"
                center_pts_img = []

        elif left_fit is not None:
            fit_mode = "left_only"
            lane_half_width_px = (self.lane_width_m * 0.42) / max(self.y_m_per_px, 1e-6)
            for yy in ys:
                xl = self._poly2_x(*left_fit, yy)
                xc = xl + lane_half_width_px
                center_pts_img.append((xc, yy + self.roi_y0))

        elif right_fit is not None:
            fit_mode = "right_only"
            lane_half_width_px = (self.lane_width_m * 0.42) / max(self.y_m_per_px, 1e-6)
            for yy in ys:
                xr = self._poly2_x(*right_fit, yy)
                xc = xr - lane_half_width_px
                center_pts_img.append((xc, yy + self.roi_y0))

        else:
            fit_mode = "none"

        return center_pts_img, fit_mode, lane_pair_debug

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

        return centerline_xs, centerline_ys

    def _validate_lane_pair(
        self,
        left_fit: Tuple[float, float, float],
        right_fit: Tuple[float, float, float],
        ys: np.ndarray,
    ) -> Tuple[bool, dict]:
        widths = []

        for yy in ys:
            xl = self._poly2_x(*left_fit, yy)
            xr = self._poly2_x(*right_fit, yy)
            widths.append(float(xr - xl))

        widths = np.array(widths, dtype=np.float64)

        if widths.size == 0:
            return False, {
                "lane_pair_valid": 0,
                "lane_width_min_px": 0.0,
                "lane_width_max_px": 0.0,
                "lane_width_mean_px": 0.0,
                "lane_width_std_px": 0.0,
                "lane_width_reason": "empty",
            }

        min_w = float(np.min(widths))
        max_w = float(np.max(widths))
        mean_w = float(np.mean(widths))
        std_w = float(np.std(widths))

        # Broad sanity bounds for now.
        # These are intentionally tolerant and should be tuned from debug results.
        valid = True
        reason = "ok"

        if min_w <= 0:
            valid = False
            reason = "crossed_or_negative"
        elif mean_w < 40:
            valid = False
            reason = "too_narrow"
        elif mean_w > 260:
            valid = False
            reason = "too_wide"
        elif std_w > 45:
            valid = False
            reason = "too_inconsistent"

        return valid, {
            "lane_pair_valid": int(valid),
            "lane_width_min_px": min_w,
            "lane_width_max_px": max_w,
            "lane_width_mean_px": mean_w,
            "lane_width_std_px": std_w,
            "lane_width_reason": reason,
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
            # First fit
            a, b, c = np.polyfit(ys, xs, 2)
            x_pred = a * ys * ys + b * ys + c
            residuals = np.abs(xs - x_pred)

            # Robust rejection threshold:
            # keep points that are not too far from the first fit
            # use a floor so the threshold does not become too tiny
            med = float(np.median(residuals))
            mad = float(np.median(np.abs(residuals - med)))
            robust_thresh = max(10.0, med + 2.5 * max(mad, 1.0))

            keep = residuals <= robust_thresh

            # Refit only if enough points survive
            if int(np.count_nonzero(keep)) >= 6:
                xs_in = xs[keep]
                ys_in = ys[keep]
                a, b, c = np.polyfit(ys_in, xs_in, 2)

            return float(a), float(b), float(c)

        except Exception:
            return None

    @staticmethod
    def _poly2_x(a: float, b: float, c: float, y: float) -> float:
        return a * y * y + b * y + c

    # =========================================================
    # Confidence
    # =========================================================
    def _compute_confidence(
        self,
        left_fit: Optional[Tuple[float, float, float]],
        right_fit: Optional[Tuple[float, float, float]],
        center_pts: List[Tuple[float, float]],
        lane_segs: List[Tuple[int, int, int, int]],
    ) -> float:
        if len(center_pts) < self.min_points:
            return 0.0

        seg_score = min(1.0, len(lane_segs) / 20.0)

        if left_fit is not None and right_fit is not None:
            fit_score = 1.0
        elif left_fit is not None or right_fit is not None:
            fit_score = 0.75
        else:
            fit_score = 0.0

        pt_score = min(1.0, len(center_pts) / 40.0)

        confidence = 0.50 * fit_score + 0.25 * seg_score + 0.25 * pt_score
        return float(max(0.0, min(1.0, confidence)))

    # =========================================================
    # Debug Visualization
    # =========================================================
    def _build_debug_view(
        self,
        frame: np.ndarray,
        roi_y0: int,
        roi_poly: np.ndarray,
        left_poly: np.ndarray,
        right_poly: np.ndarray,
        lane_segs: List[Tuple[int, int, int, int]],
        left_fit: Optional[Tuple[float, float, float]],
        right_fit: Optional[Tuple[float, float, float]],
        center_pts: List[Tuple[float, float]],
        ys: np.ndarray,
        confidence: float,
        gray_eq: np.ndarray,
        dark_mask: np.ndarray,
        blackhat: np.ndarray,
        blackhat_mask: np.ndarray,
        used_binary: np.ndarray,
        candidate_edges: np.ndarray,
    ) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]

        cv2.rectangle(vis, (0, roi_y0), (w - 1, h - 1), (0, 255, 0), 1)

        poly_vis = roi_poly.copy()
        poly_vis[:, :, 1] += roi_y0
        cv2.polylines(vis, [poly_vis], True, (0, 255, 255), 1)

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

        gray_eq_bgr = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
        dark_mask_bgr = cv2.cvtColor(dark_mask, cv2.COLOR_GRAY2BGR)
        blackhat_bgr = cv2.cvtColor(blackhat, cv2.COLOR_GRAY2BGR)
        blackhat_mask_bgr = cv2.cvtColor(blackhat_mask, cv2.COLOR_GRAY2BGR)
        used_bgr = cv2.cvtColor(used_binary, cv2.COLOR_GRAY2BGR)
        candidate_edges_bgr = cv2.cvtColor(candidate_edges, cv2.COLOR_GRAY2BGR)

        thumbs = [
            ("gray_eq", cv2.resize(gray_eq_bgr, (thumb_w, thumb_h))),
            ("dark_mask", cv2.resize(dark_mask_bgr, (thumb_w, thumb_h))),
            ("blackhat", cv2.resize(blackhat_bgr, (thumb_w, thumb_h))),
            ("blackhat_mask", cv2.resize(blackhat_mask_bgr, (thumb_w, thumb_h))),
            ("used_binary", cv2.resize(used_bgr, (thumb_w, thumb_h))),
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