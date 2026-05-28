import numpy as np


def kalman_smooth(observations: np.ndarray) -> np.ndarray:
    """
    1D Kalman filter for price/volume time series denoising.
    Same principle used in Apollo navigation — separates signal from noise.
    """
    n = len(observations)
    if n == 0:
        return np.array([], dtype=np.float64)

    Q = 1e-4   # process noise covariance
    R = 1.0    # measurement noise covariance

    x_est = np.zeros(n)
    P = np.ones(n)

    x_est[0] = observations[0]
    P[0] = 1.0

    for k in range(1, n):
        x_pred = x_est[k - 1]
        P_pred = P[k - 1] + Q
        K = P_pred / (P_pred + R)
        x_est[k] = x_pred + K * (observations[k] - x_pred)
        P[k] = (1 - K) * P_pred

    return x_est.astype(np.float64)
