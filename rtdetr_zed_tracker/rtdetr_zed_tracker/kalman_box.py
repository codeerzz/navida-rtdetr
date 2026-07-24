"""Constant-velocity Kalman filter for a single bounding box. Pure numpy.

State (8): [cx, cy, a, h, vcx, vcy, va, vh]   (a = aspect = w/h)
Measurement (4): [cx, cy, a, h]

SORT / ByteTrack formulation with noise std proportional to box height.
``dt`` is an EXPLICIT argument to :meth:`predict` — it is never derived from a
clock inside this class, so behaviour is deterministic and testable.
"""
from __future__ import annotations

import numpy as np


class KalmanBox:
    def __init__(self, measurement,
                 std_weight_position: float = 1.0 / 20.0,
                 std_weight_velocity: float = 1.0 / 160.0):
        self._spw = float(std_weight_position)
        self._svw = float(std_weight_velocity)

        z = np.asarray(measurement, dtype=np.float64).reshape(4)
        self.mean = np.zeros(8, dtype=np.float64)
        self.mean[:4] = z

        h = z[3]
        std = np.array([
            2 * self._spw * h, 2 * self._spw * h, 1e-2, 2 * self._spw * h,
            10 * self._svw * h, 10 * self._svw * h, 1e-5, 10 * self._svw * h,
        ], dtype=np.float64)
        self.covariance = np.diag(np.square(std))

    def _motion(self, dt: float):
        F = np.eye(8, dtype=np.float64)
        for i in range(4):
            F[i, i + 4] = dt
        h = self.mean[3]
        std = np.array([
            self._spw * h, self._spw * h, 1e-2, self._spw * h,
            self._svw * h, self._svw * h, 1e-5, self._svw * h,
        ], dtype=np.float64)
        Q = np.diag(np.square(std))
        return F, Q

    def predict(self, dt: float = 1.0) -> np.ndarray:
        """Advance the state by ``dt`` and return predicted xyah."""
        F, Q = self._motion(dt)
        self.mean = F @ self.mean
        self.covariance = F @ self.covariance @ F.T + Q
        return self.mean[:4].copy()

    def update(self, measurement) -> np.ndarray:
        """Correct with a measured xyah and return corrected xyah."""
        z = np.asarray(measurement, dtype=np.float64).reshape(4)
        h = self.mean[3]
        std = np.array([self._spw * h, self._spw * h, 1e-1, self._spw * h], dtype=np.float64)
        R = np.diag(np.square(std))

        H = np.eye(4, 8, dtype=np.float64)
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        innovation = z - H @ self.mean
        self.mean = self.mean + K @ innovation
        self.covariance = (np.eye(8) - K @ H) @ self.covariance
        return self.mean[:4].copy()

    @property
    def xyah(self) -> np.ndarray:
        return self.mean[:4].copy()
