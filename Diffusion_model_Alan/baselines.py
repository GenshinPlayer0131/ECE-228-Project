import numpy as np
from scipy.signal import wiener as scipy_wiener


def moving_average(x, window=9):
    if window <= 1:
        return x.copy()
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def wiener(x, window=11):
    return scipy_wiener(x, mysize=window)


def wavelet_threshold(x, wavelet="db4", level=4, mode="soft"):
    try:
        import pywt
    except ImportError:
        raise ImportError("install pywt: pip install PyWavelets")
    coeffs = pywt.wavedec(x, wavelet, level=level, mode="periodization")
    # universal threshold (Donoho): sigma * sqrt(2 log N), sigma from MAD of finest detail
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail)) / 0.6745
    thresh = sigma * np.sqrt(2 * np.log(len(x)))
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, thresh, mode=mode) for c in coeffs[1:]]
    return pywt.waverec(new_coeffs, wavelet, mode="periodization")[: len(x)]


BASELINES = {
    "moving_avg": moving_average,
    "wiener": wiener,
    "wavelet": wavelet_threshold,
}


if __name__ == "__main__":
    from dm_classes import RFPhaseDataset

    dataset = RFPhaseDataset()
    print(f"loaded {len(dataset)} windows")
    print(f"\ntesting baselines on window 0:")
    clean = dataset.windows[0]
    noisy = dataset.noisy_windows[0]
    nmse_in = np.var(noisy - clean) / np.var(clean)
    print(f"  input NMSE: {nmse_in:.3f}")
    for name, fn in BASELINES.items():
        out = fn(noisy)
        nmse = np.var(out - clean) / np.var(clean)
        print(f"  {name:12s}: NMSE={nmse:.3f}")
