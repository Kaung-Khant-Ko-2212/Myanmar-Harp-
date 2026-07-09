from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required. Install with: pip install opencv-python")


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.dtype == np.uint8:
        return gray
    return np.clip(gray, 0, 255).astype(np.uint8)


def _identity_affine() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def _compute_motion_score(m_curr_to_prev: np.ndarray) -> float:
    tx = float(m_curr_to_prev[0, 2])
    ty = float(m_curr_to_prev[1, 2])
    linear = m_curr_to_prev[:, :2].astype(np.float64)
    linear_delta = float(np.linalg.norm(linear - np.eye(2), ord="fro"))
    return float(np.hypot(tx, ty) + 10.0 * linear_delta)


def _frame_diff_motion_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    # Fallback score proxy when transform estimation fails.
    diff = cv2.absdiff(prev_gray, curr_gray).astype(np.float32)
    return float(np.mean(diff) / 255.0 * 10.0)


@dataclass
class StabilizationResult:
    stabilized_frame: np.ndarray
    motion_score: float
    method: str
    matrix_curr_to_prev: np.ndarray


class GlobalStabilizer:
    def __init__(
        self,
        ecc_iters: int = 50,
        ecc_eps: float = 1e-5,
        orb_nfeatures: int = 1000,
        min_matches: int = 12,
    ) -> None:
        _require_cv2()
        self.ecc_iters = int(ecc_iters)
        self.ecc_eps = float(ecc_eps)
        self.orb_nfeatures = int(orb_nfeatures)
        self.min_matches = int(min_matches)

    def stabilize(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
    ) -> StabilizationResult:
        if prev_frame is None or curr_frame is None:
            raise ValueError("prev_frame and curr_frame must be non-null images.")

        prev_gray = _to_gray(prev_frame)
        curr_gray = _to_gray(curr_frame)
        h, w = prev_gray.shape[:2]

        # 1) ECC first
        try:
            warp_template_to_input = _identity_affine()
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.ecc_iters,
                self.ecc_eps,
            )
            cv2.findTransformECC(
                templateImage=prev_gray,
                inputImage=curr_gray,
                warpMatrix=warp_template_to_input,
                motionType=cv2.MOTION_AFFINE,
                criteria=criteria,
            )
            stabilized = cv2.warpAffine(
                curr_frame,
                warp_template_to_input,
                (w, h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
            matrix_curr_to_prev = cv2.invertAffineTransform(warp_template_to_input).astype(np.float32)
            return StabilizationResult(
                stabilized_frame=stabilized,
                motion_score=_compute_motion_score(matrix_curr_to_prev),
                method="ecc",
                matrix_curr_to_prev=matrix_curr_to_prev,
            )
        except Exception:
            pass

        # 2) Fallback ORB + RANSAC affine
        try:
            orb = cv2.ORB_create(nfeatures=self.orb_nfeatures)
            kp_prev, des_prev = orb.detectAndCompute(prev_gray, None)
            kp_curr, des_curr = orb.detectAndCompute(curr_gray, None)
            if des_prev is None or des_curr is None:
                raise RuntimeError("ORB descriptors unavailable.")

            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            knn = matcher.knnMatch(des_curr, des_prev, k=2)
            good = []
            for pair in knn:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            if len(good) < self.min_matches:
                raise RuntimeError(f"Not enough good matches: {len(good)}")

            src_pts = np.float32([kp_curr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_prev[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            matrix_curr_to_prev, inliers = cv2.estimateAffinePartial2D(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=2000,
                confidence=0.99,
                refineIters=10,
            )
            if matrix_curr_to_prev is None:
                raise RuntimeError("RANSAC affine failed.")

            stabilized = cv2.warpAffine(
                curr_frame,
                matrix_curr_to_prev,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            matrix_curr_to_prev = matrix_curr_to_prev.astype(np.float32)
            return StabilizationResult(
                stabilized_frame=stabilized,
                motion_score=_compute_motion_score(matrix_curr_to_prev),
                method="orb",
                matrix_curr_to_prev=matrix_curr_to_prev,
            )
        except Exception:
            pass

        # 3) Last resort: no transform
        identity = _identity_affine()
        fallback_score = _frame_diff_motion_score(prev_gray, curr_gray)
        return StabilizationResult(
            stabilized_frame=curr_frame.copy(),
            motion_score=max(_compute_motion_score(identity), fallback_score),
            method="identity",
            matrix_curr_to_prev=identity,
        )


def stabilize_frame_pair(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    stabilizer: GlobalStabilizer | None = None,
) -> tuple[np.ndarray, float]:
    stab = stabilizer or GlobalStabilizer()
    result = stab.stabilize(prev_frame=prev_frame, curr_frame=curr_frame)
    return result.stabilized_frame, float(result.motion_score)
