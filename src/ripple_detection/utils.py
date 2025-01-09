from typing import List
import numpy as np
import mne


def create_mne_raw(
        data: np.ndarray,
        ch_names: List[str],
        ch_types: List[str],
        sampling_rate: float
) -> mne.io.RawArray:
    """
    Create an MNE RawArray object from NumPy array data.

    Parameters
    ----------
    data : np.ndarray
        A 2D NumPy array of shape (n_channels, n_samples), where each row represents a channel's data.
    ch_names : List[str]
        A list of channel names corresponding to the rows in the `data` array.
    ch_types : List[str]
        A list of channel types (e.g., 'eeg', 'meg', etc.), one for each channel.
    sampling_rate : float
        The sampling frequency of the data in Hz.

    Returns
    -------
    mne.io.RawArray
        An MNE RawArray object containing the input data and metadata.

    Raises
    ------
    ValueError
        If the number of channels in `data` does not match the length of `ch_names` or `ch_types`.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(2, 1000)  # 2 channels, 1000 samples
    >>> ch_names = ['Channel1', 'Channel2']
    >>> ch_types = ['eeg', 'eeg']
    >>> sampling_rate = 1000.0
    >>> raw = create_mne_raw(data, ch_names, ch_types, sampling_rate)
    >>> print(raw)
    """
    # Validate inputs
    if data.shape[0] != len(ch_names) or data.shape[0] != len(ch_types):
        raise ValueError(
            "The number of channels in `data` must match the length of `ch_names` and `ch_types`."
        )

    # Create MNE Info object
    info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types=ch_types)

    # Create RawArray object
    raw = mne.io.RawArray(data, info)

    return raw


def create_mne_epoch_array(
        data: np.ndarray,
        ch_names: List[str],
        ch_types: List[str],
        sampling_rate: float
) -> mne.EpochsArray:
    """
    Create an MNE RawArray object from NumPy array data.

    Parameters
    ----------
    data : np.ndarray
        A 3D NumPy array of shape (evernt, n_channels, n_samples), where each row represents a channel's data.
    ch_names : List[str]
        A list of channel names corresponding to the rows in the `data` array.
    ch_types : List[str]
        A list of channel types (e.g., 'eeg', 'meg', etc.), one for each channel.
    sampling_rate : float
        The sampling frequency of the data in Hz.

    Returns
    -------
    mne.io.RawArray
        An MNE RawArray object containing the input data and metadata.

    Raises
    ------
    ValueError
        If the number of channels in `data` does not match the length of `ch_names` or `ch_types`.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(2, 1000)  # 2 channels, 1000 samples
    >>> ch_names = ['Channel1', 'Channel2']
    >>> ch_types = ['eeg', 'eeg']
    >>> sampling_rate = 1000.0
    >>> raw = create_mne_raw(data, ch_names, ch_types, sampling_rate)
    >>> print(raw)
    """
    # Validate inputs
    if data.shape[1] != len(ch_names) or data.shape[1] != len(ch_types):
        raise ValueError(
            "The number of channels in `data` must match the length of `ch_names` and `ch_types`."
        )

    mne_evs = np.empty([data.shape[0], 3]).astype(int)
    mne_evs[:, 0] = np.arange(data.shape[0])
    mne_evs[:, 1] = data.shape[2]
    mne_evs[:, 2] = np.zeros(data.shape[0])
    event_id = dict(resting=0)

    # Create MNE Info object
    info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types=ch_types, verbose=False)

    # Create EpochsArray object
    arr = mne.EpochsArray(data, info, events=mne_evs, event_id=event_id, tmin=0, verbose=False)

    return arr
