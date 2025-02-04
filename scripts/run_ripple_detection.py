"""
run ripple detection on sleep data:
the .mat files are macro channels processed by nwbPipeline.
"""


import os
import numpy as np
from typing import Tuple, List
from nwb_pipeline.csc_reader import combine_csc
from ripple_detection.neuralynx_ripple import filter_data, get_ripple_stats

SECONDS_PER_HOUR = 3600


def read_csc_data(
        file_path: str, reference_files: List[List[str]], csc_files: List[List[str]], timestamps_files: List[str]
) -> Tuple[np.ndarray, int]:

    timestamps_files = [str(os.path.join(file_path, f)) for f in timestamps_files]
    csc_data_list = []
    sampling_interval = None
    for channel_ref_files, channel_csc_files in zip(reference_files, csc_files):

        reference_files = [str(os.path.join(file_path, f)) for f in channel_ref_files]
        csc_files = [str(os.path.join(file_path, f)) for f in channel_csc_files]

        (
            timestamps,
            signal_reference,
            sampling_interval,
            timestamps_start,

        ) = combine_csc(timestamps_files, reference_files)

        _, csc_signal, _, _ = combine_csc(timestamps_files, csc_files)
        csc_data_list.append(np.squeeze(csc_signal - signal_reference))

    csc_signal = np.vstack(csc_data_list)
    csc_signal = np.expand_dims(csc_signal, axis=0)
    sampling_rate = int(1/sampling_interval)
    duration = csc_signal.shape[2] / sampling_rate / SECONDS_PER_HOUR
    print(f"csc_data shape: {csc_signal.shape}, sampling_rate: {sampling_rate}, duration: {duration} hours")
    return csc_signal, sampling_rate


if __name__ == '__main__':

    macro_file_path = '/Users/XinNiuAdmin/HoffmanMount/data/PIPELINE_vc/ANALYSIS/MovieParadigm/566_MovieParadigm/Experiment-8/CSC_macro'
    macro_files_reference = [
            ["RMH1_001.mat", "RMH1_002.mat", "RMH1_003.mat", "RMH1_004.mat", "RMH1_005.mat"]
        ]

    macro_files = [
            ["RMH2_001.mat", "RMH2_002.mat", "RMH2_003.mat", "RMH2_004.mat", "RMH2_005.mat"]
        ]

    macro_timestamps_files = [
        "lfpTimeStamps_001.mat",
        "lfpTimeStamps_002.mat",
        "lfpTimeStamps_003.mat",
        "lfpTimeStamps_004.mat",
        "lfpTimeStamps_005.mat",
    ]

    output_file = os.path.join(os.path.dirname(macro_file_path), 'ripple_detection', 'RMH2.npz')
    macro_data, sr = read_csc_data(macro_file_path, macro_files_reference, macro_files, macro_timestamps_files)

    ch_names = ['RMH']
    ch_types = ['seeg']

    macro_ripple, macro_ied = filter_data(macro_data, sr, ch_names, ch_types)
    ripple_array = get_ripple_stats(macro_ripple, macro_ied, sr)

    if not os.path.exists(os.path.dirname(output_file)):
        os.mkdir(os.path.dirname(output_file))
    np.savez(output_file, array=ripple_array)
