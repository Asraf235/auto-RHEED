"""
FFT of a RHEED intensity-vs-time series → growth-rate readout.

A specular RHEED intensity oscillation has a period equal to the time to
complete one monolayer, so the dominant FFT frequency f0 is the growth
rate in monolayers/second; 1/f0 is the seconds-per-monolayer, and
f0 × (per-layer thickness) is the deposition rate in Å/s.

Before transforming we remove the slow drift (large DC + envelope) that
would otherwise dominate the spectrum, and apply a Hann window to reduce
spectral leakage.
"""
import numpy as np


def _detrend(y: np.ndarray, t: np.ndarray, method: str, alpha: float) -> np.ndarray:
    if method == "mean":
        return y - y.mean()
    if method == "linear":
        coef = np.polyfit(t, y, 1)
        return y - np.polyval(coef, t)
    if method == "ema":
        # Exponential moving average as the slow envelope; subtract it.
        s = np.empty_like(y)
        s[0] = y[0]
        a = float(alpha)
        for i in range(1, len(y)):
            s[i] = a * y[i] + (1 - a) * s[i - 1]
        return y - s
    return y - y.mean()


def intensity_fft(intensities, timestamps, detrend: str = "ema",
                  alpha: float = 0.15) -> dict:
    """
    Returns {freq, power, f0, period, growth_ml_per_s, dt}.
    `freq` in Hz, `power` is the (windowed) magnitude spectrum with the DC
    bin zeroed. f0 is the dominant non-DC frequency.
    """
    y = np.asarray(intensities, dtype=float)
    t = np.asarray(timestamps, dtype=float)
    n = len(y)
    if n < 8:
        raise ValueError("Need at least 8 frames for a meaningful FFT")

    yd = _detrend(y, t, detrend, alpha)
    yw = yd * np.hanning(n)

    dt = float(np.mean(np.diff(t))) if n > 1 else 1.0
    if dt <= 0:
        dt = 1.0
    freqs = np.fft.rfftfreq(n, d=dt)
    power = np.abs(np.fft.rfft(yw))
    power[0] = 0.0  # drop DC — the slow drift, not a growth oscillation

    peak_idx = int(np.argmax(power))
    f0 = float(freqs[peak_idx]) if power[peak_idx] > 0 else 0.0
    period = (1.0 / f0) if f0 > 0 else None

    return {
        "freq": freqs.tolist(),
        "power": power.tolist(),
        "f0": f0,                 # Hz = monolayers / second
        "period": period,         # s / monolayer
        "growth_ml_per_s": f0,    # alias, for clarity
        "dt": dt,
    }
