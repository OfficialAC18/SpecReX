import numpy as np
from scipy.signal import find_peaks


def _fid_score(range: int,
              ground_truth: np.ndarray,
              peak: np.ndarray) -> float:

    return np.max(1/(np.maximum(np.abs(ground_truth - peak) - range, 0) + 1))
    


def expected_fid_score(thereshold: float,
                      ground_truth: np.ndarray,
                      saliency_landscape: np.ndarray,
                      weighted:bool) -> float:
    """
    Calculate the Fidelity Score of the saliency landscape

    Parameters:
    thereshold: Only peaks above this thereshold are considered in the calculation,
                The threshold is relative to the maximum peak in the landscape.
    ground_truths (numpy.ndarray): The ground truth saliency positions in the landscape.
    saliency_landscape (numpy.ndarray): The saliency landscape of some input.
    weighted (bool): If True, the FID score is weighted by the peak height.

    Returns:
    float: The FID score.
    """

    # Find peaks above the threshold
    peaks, _ = find_peaks(saliency_landscape)

    #Get peak heights and sort them in descending order
    sorted_peak_indices = np.argsort(saliency_landscape[peaks])[::-1]
    peaks = peaks[sorted_peak_indices]
    peak_heights = peak_heights[sorted_peak_indices]

    #Get rid of peaks below the threshold
    unwanted_peaks = []
    for idx, peak in enumerate(peaks):
        if peak_heights[peak] < thereshold:
            unwanted_peaks.append(idx)
    
    peaks = np.delete(peaks, unwanted_peaks)
    peak_heights = np.delete(peak_heights, unwanted_peaks)

    #Calculate the FID_Score for each peak and take average (weighted or not)
    fid_scores = []
    for peak in peaks:
        fid_scores.append(_fid_score(ground_truth, peak))

    if weighted:
        fid_score = np.average(fid_scores, weights=(peak_heights/peak_heights[0])) #The array is sorted     
    else:
        fid_score = np.mean(fid_scores)

    return fid_score