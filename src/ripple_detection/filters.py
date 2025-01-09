### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from scipy.signal import butter
from numpy import asarray
from scipy.signal import filtfilt
import numpy.typing as npt
from typing import List


def butter_filter(dat: npt.NDArray, freq_range: List[int], sample_rate: int, filter_type: str, order: int, axis: int=-1):
    """Wrapper for a Butterworth filter.

    """
    # set up the filter
    freq_range = asarray(freq_range)

    # Nyquist frequency
    nyq = sample_rate/2.

    # generate the butterworth filter coefficients
    [b,a] = butter(order, freq_range/nyq, filter_type)
    dat = filtfilt(b, a, dat, axis=axis)

    return dat