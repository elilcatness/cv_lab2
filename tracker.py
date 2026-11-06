from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackerConfig:
    max_points: int = 250
    quality: float = 0.01
    min_distance: int = 7
    lk_window: int = 21
    lk_levels: int = 3
    fb_threshold: float = 1.5
    min_inliers: int = 10
    replenish_at: int = 40
    sift_features: int = 1500
    sift_ratio: float = 0.75
    ransac_px: float = 3.0


@dataclass
class TrackResult:
    status: str
    method: str
    polygon: np.ndarray | None
    points: np.ndarray
    inliers: int = 0
    matches: int = 0


class HybridTracker:
    def __init__(
        self,
        first_frame: np.ndarray,
        roi: tuple[int, int, int, int],
        config: TrackerConfig | None = None,
    ):
        self.cfg = config or TrackerConfig()
        self.sift = cv2.SIFT_create(nfeatures=self.cfg.sift_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.x, self.y, self.w, self.h = (int(v) for v in roi)
        height, width = first_frame.shape[:2]
        if not (
            0 <= self.x < width
            and 0 <= self.y < height
            and self.w >= 25
            and self.h >= 25
            and self.x + self.w <= width
            and self.y + self.h <= height
        ):
            raise ValueError(
                'Недопустимая область объекта: выделите прямоугольник от 25×25 пикселей.'
            )
        self.reference = cv2.cvtColor(
            first_frame[self.y : self.y + self.h, self.x : self.x + self.w],
            cv2.COLOR_BGR2GRAY,
        )
        self.ref_keypoints, self.ref_descriptors = self.sift.detectAndCompute(
            self.reference, None
        )
        if (
            self.ref_descriptors is None
            or len(self.ref_keypoints) < self.cfg.min_inliers
        ):
            raise ValueError(
                'В выделенной области недостаточно признаков SIFT. Выберите более текстурированный объект.'
            )
        self.corners = np.float32(
            [[0, 0], [self.w - 1, 0], [self.w - 1, self.h - 1], [0, self.h - 1]]
        )
        self.initial_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.reset_first()

    def reset_first(self) -> TrackResult:
        self.last_gray = self.initial_gray.copy()
        pts = cv2.goodFeaturesToTrack(
            self.reference, self.cfg.max_points, self.cfg.quality, self.cfg.min_distance
        )
        local = pts.reshape(-1, 2) if pts is not None else np.empty((0, 2), np.float32)
        self.ref_pts = np.asarray(local, np.float32)
        self.cur_pts = self.ref_pts + [self.x, self.y]
        self.H = np.array([[1, 0, self.x], [0, 1, self.y], [0, 0, 1]], dtype=np.float64)
        self.polygon = self.corners + [self.x, self.y]
        if len(self.ref_pts) < self.cfg.min_inliers:
            self._seed_corners(self.last_gray, self.H)
        return self._result('TRACKING', 'INIT')

    def jump(self, frame: np.ndarray, first: bool = False) -> TrackResult:
        if first:
            return self.reset_first()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = self._recover(gray)
        self.last_gray = gray
        if not found:
            self._lose()
        return self._result(
            'TRACKING' if found else 'LOST', 'SIFT' if found else 'SEARCH'
        )

    def step(self, frame: np.ndarray) -> TrackResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        method, matches = 'KLT', 0
        tracked = False
        if len(self.cur_pts) >= 4:
            p0 = self.cur_pts.reshape(-1, 1, 2).astype(np.float32)
            options = {
                'winSize': (self.cfg.lk_window, self.cfg.lk_window),
                'maxLevel': self.cfg.lk_levels,
                'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            }
            # Lucas–Kanade: Ix*u + Iy*v + It = 0 on a local window (OpenCV Optical Flow tutorial).
            p1, ok1, _ = cv2.calcOpticalFlowPyrLK(
                self.last_gray, gray, p0, None, **options
            )
            if p1 is not None and ok1 is not None:
                pback, ok2, _ = cv2.calcOpticalFlowPyrLK(
                    gray, self.last_gray, p1, None, **options
                )
                if pback is not None and ok2 is not None:
                    error = np.max(
                        np.abs(p0.reshape(-1, 2) - pback.reshape(-1, 2)), axis=1
                    )
                    good = (
                        (ok1.ravel() > 0)
                        & (ok2.ravel() > 0)
                        & (error < self.cfg.fb_threshold)
                    )
                    src, dst = self.ref_pts[good], p1.reshape(-1, 2)[good]
                    fit = self._fit(src, dst, gray.shape)
                    if fit is not None:
                        self._accept(fit, src, dst)
                        tracked = True
                        matches = len(src)
        if tracked:
            if len(self.cur_pts) < self.cfg.replenish_at:
                # First restore correspondences by SIFT; then extend KLT with Shi–Tomasi.
                if self._recover(gray, region=self.polygon):
                    method = 'KLT → SIFT'
                else:
                    self._seed_corners(gray, self.H)
        else:
            method = 'SIFT'
            if not self._recover(gray):
                self._lose()
        self.last_gray = gray
        status = 'TRACKING' if self.polygon is not None else 'LOST'
        result = self._result(status, method if status == 'TRACKING' else 'SEARCH')
        if matches and method == 'KLT':
            result.matches = matches
        return result

    def _fit(self, src: np.ndarray, dst: np.ndarray, shape: tuple) -> tuple | None:
        if len(src) < self.cfg.min_inliers:
            return None
        # Planar homography: x' ~ Hx, estimated with DLT + RANSAC (OpenCV Features2D + Homography).
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.cfg.ransac_px)
        if H is None or mask is None or not np.isfinite(H).all():
            return None
        inside = mask.ravel().astype(bool)
        if inside.sum() < self.cfg.min_inliers or inside.mean() < 0.48:
            return None
        coverage = cv2.contourArea(cv2.convexHull(src[inside].astype(np.float32)))
        if coverage < self.w * self.h * 0.012:
            return None
        projected = cv2.perspectiveTransform(self.corners[None, :, :], H)[0]
        if not np.isfinite(projected).all() or not cv2.isContourConvex(projected):
            return None
        area = cv2.contourArea(projected)
        frame_area = shape[0] * shape[1]
        if area < max(30, frame_area * 0.0001) or area > frame_area * 12:
            return None
        frame_polygon = np.float32(
            [[0, 0], [shape[1] - 1, 0], [shape[1] - 1, shape[0] - 1], [0, shape[0] - 1]]
        )
        try:
            visible_area, _ = cv2.intersectConvexConvex(
                projected.astype(np.float32), frame_polygon
            )
        except cv2.error:
            return None
        if visible_area < max(15, frame_area * 0.00006):
            return None
        mapped = cv2.perspectiveTransform(src[inside][None, :, :], H)[0]
        residual = np.linalg.norm(mapped - dst[inside], axis=1)
        if np.median(residual) > self.cfg.ransac_px * 1.25:
            return None
        return H, projected, inside

    def _accept(self, fit: tuple, src: np.ndarray, dst: np.ndarray) -> None:
        self.H, self.polygon, inliers = fit
        self.ref_pts = np.ascontiguousarray(
            src[inliers][: self.cfg.max_points], np.float32
        )
        self.cur_pts = np.ascontiguousarray(
            dst[inliers][: self.cfg.max_points], np.float32
        )

    def _recover(self, gray: np.ndarray, region: np.ndarray | None = None) -> bool:
        mask = None
        if region is not None:
            mask = np.zeros_like(gray)
            cv2.fillConvexPoly(mask, np.rint(region).astype(np.int32), 255)
            radius = max(11, int(min(gray.shape) * 0.025))
            mask = cv2.dilate(
                mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2),
            )
        kp, desc = self.sift.detectAndCompute(gray, mask)
        if desc is None or len(kp) < self.cfg.min_inliers:
            return False
        pairs = self.matcher.knnMatch(self.ref_descriptors, desc, k=2)
        # Lowe ratio test rejects ambiguous nearest-neighbour descriptor matches.
        matches = [
            a
            for pair in pairs
            if len(pair) == 2
            for a, b in [pair]
            if a.distance < self.cfg.sift_ratio * b.distance
        ]
        matches.sort(key=lambda x: x.distance)
        unique, used = [], set()
        for match in matches:
            if match.trainIdx not in used:
                unique.append(match)
                used.add(match.trainIdx)
        if len(unique) < self.cfg.min_inliers:
            return False
        src = np.float32([self.ref_keypoints[m.queryIdx].pt for m in unique])
        dst = np.float32([kp[m.trainIdx].pt for m in unique])
        fit = self._fit(src, dst, gray.shape)
        if fit is None:
            return False
        if region is not None:
            # A replenishment must agree with the polygon already tracked by KLT.
            intersection, _ = cv2.intersectConvexConvex(
                fit[1].astype(np.float32), region.astype(np.float32)
            )
            union = cv2.contourArea(fit[1]) + cv2.contourArea(region) - intersection
            if union <= 0 or intersection / union < 0.3:
                return False
        self._accept(fit, src, dst)
        self._seed_corners(gray, self.H)
        return True

    def _seed_corners(self, gray: np.ndarray, H: np.ndarray) -> None:
        if len(self.cur_pts) >= self.cfg.max_points or self.polygon is None:
            return
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, np.rint(self.polygon).astype(np.int32), 255)
        for px, py in self.cur_pts:
            cv2.circle(
                mask, (round(px), round(py)), self.cfg.min_distance, 0, -1
            )
        count = self.cfg.max_points - len(self.cur_pts)
        corners = cv2.goodFeaturesToTrack(
            gray, count, self.cfg.quality, self.cfg.min_distance, mask=mask
        )
        if corners is None:
            return
        try:
            inverse = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return
        candidate = corners.reshape(-1, 2).astype(np.float32)
        reference = cv2.perspectiveTransform(candidate[None, :, :], inverse)[0]
        good = (
            (reference[:, 0] >= 0)
            & (reference[:, 0] < self.w)
            & (reference[:, 1] >= 0)
            & (reference[:, 1] < self.h)
        )
        if good.any():
            self.cur_pts = np.concatenate((self.cur_pts, candidate[good])).astype(
                np.float32
            )
            self.ref_pts = np.concatenate((self.ref_pts, reference[good])).astype(
                np.float32
            )

    def _lose(self) -> None:
        self.polygon = None
        self.H = None
        self.ref_pts = np.empty((0, 2), np.float32)
        self.cur_pts = np.empty((0, 2), np.float32)

    def _result(self, status: str, method: str) -> TrackResult:
        return TrackResult(
            status,
            method,
            self.polygon.copy() if self.polygon is not None else None,
            self.cur_pts.copy(),
            len(self.cur_pts),
        )
