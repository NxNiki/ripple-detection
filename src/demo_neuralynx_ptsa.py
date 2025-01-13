import pandas as pd
import numpy as np
import sys
import os
import warnings
import matplotlib.pyplot as plt
from copy import copy
from scipy import stats
from scipy.stats import zscore
import pickle

from scipy import stats, signal, io
import time
import mat73
import mne
from ptsa.data.filters import ButterworthFilter, ResampleFilter, MorletWaveletFilter
from scipy.signal import firwin, filtfilt, kaiserord
from ptsa.data.timeseries import TimeSeries

from ripple_detection.general import superVstack
from ripple_detection.slow_wave_ripple import (
    detect_ripples_hamming, detect_ripples_butter, detect_ripples_staresina,
    downsampleBinary, getStartEndArrays
)
from ripple_detection.neuralynx_io import load_ncs
from ripple_detection.filters import butter_filter
from ripple_detection.utils import create_mne_raw, create_mne_epoch_array

warnings.filterwarnings("ignore") # neuralynx_io gives annoying warnings but seems to work fine

plt.rcParams['pdf.fonttype'] = 42; plt.rcParams['ps.fonttype'] = 42 # fix fonts for Illustrator
pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 100)

path = '../test/data/'
# each pair will be subtracted for bipolar referencing
fns = [
    ['LAH1.ncs','LAH2.ncs'],
    ['LA1.ncs','LA2.ncs'],
]

save_values = 0
recall_type_switch = 0
remove_soz_ictal = 0 # 0 for nothing, 1 for remove SOZ, 2 for keep ONLY SOZ ###
filter_type = 'hamming'
# filter_type = 'butter'

desired_sample_rate = 500. # in Hz. This seems like lowerst common denominator recording freq.
eeg_buffer = 300 # buffer to add to either end of IRI when processing eeg

macro_data = []
for fn in fns:
    macro_ncs1 = load_ncs(os.path.join(path, fn[0]))
    macro_ncs2 = load_ncs(os.path.join(path, fn[1]))
    macro_data = superVstack(macro_data, macro_ncs1['data']-macro_ncs2['data'])

# add event dimension to convert to MNE raw array:
macro_data = np.expand_dims(macro_data, axis=0)

sr = int(macro_ncs1['header']['SamplingFrequency'])

print(f'macro_data shape: {np.shape(macro_data)}')
print(f'sr: {sr}')

time_in_sec = np.linspace(1, np.shape(macro_data)[2], np.shape(macro_data)[2]) / sr
eeg_ptsa = TimeSeries(macro_data,
                dims=('event','channel', 'time'),
                coords={'event':[0], # so can keep it 3d
                        'channel':np.arange(np.shape(macro_data)[1]),
                        'time':time_in_sec,
                        'samplerate':sr})

# line removal...don't do 120 for now (I never see any line noise there for whatever reason)
macro_data = ButterworthFilter(freq_range=[58., 62.], filt_type='stop', order=4).filter(timeseries=eeg_ptsa)
macro_data = ButterworthFilter(freq_range=[178., 182.], filt_type='stop', order=4).filter(timeseries=eeg_ptsa)

trans_width = 5. # Width of transition region, normalized so that 1 corresponds to pi radians/sample.
# That is, the frequency is expressed as a fraction of the Nyquist frequency.
ntaps = (2/3)*np.log10(1/(10*(1e-3*1e-4)))*(sr/trans_width)  # gives 400 with sr=500, trans=5
nyquist = sr/2

ch_names = [x[0].replace('.ncs', '') for x in fns]
ch_types = ['seeg'] * len(ch_names)

# filter for ripples using filter selected above
if filter_type == 'hamming':
    # need to subtract out to get the filtered signal since default is bandstop but want to keep it as PTSA
    FIR_bandstop = firwin(int(ntaps+1), [70., 178.], fs=sr, window='hamming', pass_zero='bandstop')
    eeg_rip_band = macro_data-filtfilt(FIR_bandstop, 1., macro_data)
    bandstop_25_60 = firwin(int(ntaps+1), [20., 58.], fs=sr, window='hamming', pass_zero='bandstop')  # Norman 2019 IED
    eeg_ied_band = macro_data-filtfilt(bandstop_25_60, 1., macro_data)
    ntaps40, beta40 = kaiserord(40, trans_width/nyquist)
    kaiser_40lp_filter = firwin(ntaps40, cutoff=40, window=('kaiser', beta40), scale=False, fs=sr, pass_zero='lowpass')
elif filter_type == 'butter':
    eeg_rip_band = butter_filter(macro_data, freq_range=[80.,120.], sample_rate=sr, filter_type='bandpass', order=2)
    eeg_ied_band = butter_filter(macro_data, freq_range=[250.,490.], sample_rate=sr, filter_type='bandpass', order=2)
    time_length = np.shape(eeg_rip_band)[2]/int(sr)
    eeg_raw = create_mne_epoch_array(macro_data, ch_names, ch_types, sr)
elif filter_type == 'staresina':
    FIR_bandstop_star = firwin(241, [80.,100.], fs=sr, window='hamming', pass_zero='bandstop') # order = 3*80+1
    eeg_rip_band = macro_data-filtfilt(FIR_bandstop_star, 1., macro_data)

if filter_type != 'staresina':
    time_length = np.shape(eeg_rip_band)[2]/int(sr)
    eeg_rip_band = create_mne_epoch_array(eeg_rip_band, ch_names, ch_types, sr)
    _ = eeg_rip_band.apply_hilbert(envelope=True)
    eeg_ied_band = create_mne_epoch_array(eeg_ied_band, ch_names, ch_types, sr)
    _ = eeg_ied_band.apply_hilbert(envelope=True)

ripple_array = []
HFA_array = []

trial_by_trial_correlation = []
elec_by_elec_correlation = []
elec_ripple_rate_array = []
session_ripple_rate_by_elec = []

total_channel_ct = 0
min_ripple_rate = 0.1  # Hz. # 0.1 for hamming
# max_ripple_rate = 1.5 # Hz. # 1.5 for hamming
max_ripple_rate = 1.5  # Hz. # 1.5 for hamming

max_trial_by_trial_correlation = 0.05  # if ripples correlated more than this remove them # 0.05 for hamming
max_electrode_by_electrode_correlation = 0.2  # ??? # 0.2 for hamming

for channel in range(np.shape(eeg_rip_band.get_data())[1]):  # unpack number of channels
    print(f'Channel #{channel} of {np.shape(eeg_rip_band.get_data())[1]}')
    total_channel_ct += 1  # total channels before artifact removal

    # get data from MNE container
    eeg_rip = eeg_rip_band.get_data()[:, channel, :]
    eeg_ied = eeg_ied_band.get_data()[:, channel, :]

    # select detection algorithm (note that iedlogic is same for both so always run that)
    if filter_type == 'hamming':
        # filter IEDs
        eeg_ied = eeg_ied ** 2  # already rectified now square
        eeg_ied = filtfilt(kaiser_40lp_filter, 1., eeg_ied)  # low pass filter
        mean1 = np.mean(eeg_ied)
        std1 = np.std(eeg_ied)
        iedlogic = eeg_ied >= mean1 + 4 * std1  # Norman et al 2019
        # detect ripples
        ripplelogic = detect_ripples_hamming(eeg_rip, trans_width, sr, iedlogic)
    elif filter_type == 'butter':
        eeg_mne = eeg_raw.get_data()[:, channel, :]
        # detect ripples
        ripplelogic, iedlogic = detect_ripples_butter(eeg_rip, eeg_ied, eeg_mne, sr)
    elif filter_type == 'staresina':
        ripplelogic = detect_ripples_staresina(eeg_rip, sr)

    if filter_type == 'butter':
        desired_sample_rate = 1000  # for Vaz algo

    if sr > desired_sample_rate:  # downsampling here for anything greater than 500 (hamming) or 1000 (butter)
        ripplelogic = downsampleBinary(ripplelogic, sr / desired_sample_rate)

    # ripples are detected, so can remove buffers now #
    # if ripplelogic is just 1d (because it only has 1 "trial") it bugs out programs below
    if len(ripplelogic.shape) == 1:  # if just detecting within a single vector
        ripplelogic = np.expand_dims(ripplelogic, axis=0)

    # skip this electrode if the ripple rate is below threshold
    temp_start_array, _ = getStartEndArrays(ripplelogic)
    elec_ripple_rate = np.sum(temp_start_array) / temp_start_array.shape[0] / time_length

    if elec_ripple_rate < min_ripple_rate:
        print(f'skipped b/c {elec_ripple_rate} below ripple rate thresh {min_ripple_rate} for channel: {channel}')
        continue
    elif elec_ripple_rate > max_ripple_rate:
        print(f'skipped b/c {elec_ripple_rate} ABOVE ripple rate thresh {max_ripple_rate} for channel: {channel}')
        continue

        # check the ripples for this electrode and make sure they're not super correlated across trials
    # first, bin the array so can get more realistic correlation not dependent on ms timing
    binned_ripplelogic = downsampleBinary(ripplelogic[:, :ripplelogic.size - ripplelogic.size % 10],
                                          10)  # downsample by 10x so 10 ms bins
    trial_ripple_df = pd.DataFrame(data=np.transpose(binned_ripplelogic))
    num_cols = len(list(trial_ripple_df))
    temp_tbt_corr = 0
    if num_cols > 1:  # if more than 1 trial
        trial_ripple_df.columns = ['col_' + str(i) for i in range(num_cols)]  # generate range of ints for suffixes
        if sum(sum(trial_ripple_df)) > 1:
            temp_tbt_corr = np.mean(pg.pairwise_corr(trial_ripple_df, method='spearman').r)
        else:
            temp_tbt_corr = 1
        if temp_tbt_corr > max_trial_by_trial_correlation:
            print('skipped b/c above trial-by-trial correlation for ch.: ' + str(channel))
            continue

    ## if this electrode passes SAVE data ##
    trial_by_trial_correlation.append(temp_tbt_corr)  # corr b/w trials
    elec_ripple_rate_array.append(elec_ripple_rate)  # record the average ripple rate for this electrode

    # append arrays across electrodes
    ripple_array = superVstack(ripple_array, ripplelogic)  # append across electrodes