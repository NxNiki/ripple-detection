import traceback

import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from copy import copy
import functools
import datetime
import scipy
from scipy.signal import firwin, filtfilt, kaiserord, convolve2d

# from ptsa.data.filters import morlet
# from ptsa.data.filters import ButterworthFilter
from ripple_detection.general import get_logical_chunks, superVstack


def write_log(s, log_name):
    date = datetime.datetime.now().strftime('%F_%H-%M-%S')
    output = date + ': ' + str(s)
    with open(log_name, 'a') as logfile:
        print(output)
        logfile.write(output+'\n')


def log_df_exception_line(row, e, log_name):
    rd = row._asdict()
    if type(e) is str: # if it's just a string then this was not an Exception I just wanted to print my own error
        write_log('DF Exception: Sub: ' + str(rd['subject']) + ', Sess: ' + str(rd['session']) + \
                  ', Manual error, ' + e + ', file: , line no: XXX', log_name)
    else: # if e is an exception then normal print to .txt log
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_num = exc_tb.tb_lineno
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        write_log('DF Exception: Sub: ' + str(rd['subject']) + ', Sess: ' + str(rd['session']) + \
                  ', ' + e.__class__.__name__ + ', ' + str(e) + ', file: ' + fname + ', line no: ' + str(line_num),
                  log_name)


def log_df_exception(row, e, log_name):
    rd = row._asdict()
    write_log('DF Exception: Sub: ' + str(rd['subject']) + ', Exp: ' + str(rd['experiment']) + ', Sess: ' + \
              str(rd['session']) + ', ' + e.__class__.__name__ + ', ' + str(e), log_name)


def log_exception(e, log_name):
    write_log(e.__class__.__name__ + ', ' + str(e) + '\n' +
              ''.join(traceback.format_exception(type(e), e, e.__traceback__)), log_name)


def norm_fft(eeg):
    from scipy import fft
    # gets you the frequency spectrum after the fft by removing mirrored signal and taking modulus
    N = len(eeg)
    fft_eeg = 1/N*np.abs(fft(eeg)[:N//2]) # should really normalize by Time/sample rate (e.g. 4 s of eeg/500 hz sampling=8)
    return fft_eeg


# PYBEH implementation for temporal clustering. This code applies the df to pybeh
def pd_temp_fact(df, skip_first_n=0):
    
    import pybeh
    from pybeh.temp_fact import temp_fact
    from pybeh.make_recalls_matrix import make_recalls_matrix
    
    """Expects as input a dataframe (df) for one subject"""
    pres_itemnos, rec_itemnos, _, _ = get_itemno_matrices(df)
    recalls = pybeh.make_recalls_matrix.make_recalls_matrix(pres_itemnos, rec_itemnos)

    temp_fact = pybeh.temp_fact.temp_fact(recalls=recalls, 
                  subjects=np.array(['a'] * recalls.shape[0]),
                  listLength=pres_itemnos.shape[1],
                  skip_first_n=skip_first_n)
    return temp_fact[0]


def pd_semantic_fact(df, dist_mat, skip_first_n=0):
    # this doesn't work yet...not sure if I set up the apply with two inputs correctly
    # in any case, the way this uses item_num doesn't make sense with my 461 (not 300) 
    # word list used across ALL catFR1. Would have to reindex the 300 words in each pool
    # to 461 or recalculate the word2vec for every session which is a pain
    import pybeh
    from pybeh.dist_fact import dist_fact
    from pybeh.make_recalls_matrix import make_recalls_matrix
    
    """Expects as input a dataframe (df) for one subject"""
    pres_itemnos, rec_itemnos, _, _ = get_itemno_matrices(df)
#     recalls = pybeh.make_recalls_matrix.make_recalls_matrix(pres_itemnos, rec_itemnos)

    temp_fact = pybeh.dist_fact.dist_fact(rec_itemnos=rec_itemnos, pres_itemnos=pres_itemnos,
                  subjects=np.array(['a'] * rec_itemnos.shape[0]),
                  dist_mat=dist_mat,
                  listLength=pres_itemnos.shape[1],
                  skip_first_n=skip_first_n)
    return temp_fact[0]
    

def get_bp_tal_struct(sub, montage, localization):
    
    # inputs: subject name, montage, localization
    # outputs: 
    
    from ptsa.data.readers import TalReader    
   
    #Get electrode information -- bipolar
    tal_path = '/protocols/r1/subjects/'+sub+'/localizations/'+str(localization)+'/montages/'+str(montage)+'/neuroradiology/current_processed/pairs.json'
    tal_reader = TalReader(filename=tal_path)
    tal_struct = tal_reader.read()
    monopolar_channels = tal_reader.get_monopolar_channels()
    bipolar_pairs = tal_reader.get_bipolar_pairs()
    
    return tal_struct, bipolar_pairs, monopolar_channels


def loc_to_pairs_translation(pairs,localizations):
    # localizations is all the possible contacts and bipolar pairs locations
    # pairs is the actual bipolar pairs recorded (plugged in to a certain montage of the localization)
    # this finds the indices that translate the localization pairs to the pairs/tal_struct

    loc_pairs = localizations.type.pairs
    loc_pairs = np.array(loc_pairs.index)
    split_pairs = [pair.upper().split('-') for pair in pairs.label] # pairs.json is usually upper anyway but things like "micro" are not
    pairs_to_loc_idxs = []
    for loc_pair in loc_pairs:
        loc_pair = [loc.upper() for loc in loc_pair] # pairs.json is always capitalized so capitalize location.pairs to match (e.g. Li was changed to an LI)
        loc_pair = list(loc_pair)
        idx = (np.where([loc_pair==split_pair for split_pair in split_pairs])[0])
        if len(idx) == 0:
            loc_pair.reverse() # check for the reverse since sometimes the electrodes are listed the other way
            idx = (np.where([loc_pair==split_pair for split_pair in split_pairs])[0])
            if len(idx) == 0:
                idx = ' '
        pairs_to_loc_idxs.extend(idx)

    return pairs_to_loc_idxs # these numbers you see are the index in PAIRS frame that the localization.pairs region will get put


def get_elec_regions(localizations,pairs): 
    # 2020-08-13 new version after consulting with Paul 
    # suggested order to use regions is: stein->das->MTL->wb->mni
    
    # 2020-08-26 previous version input tal_struct (pairs.json as a recArray). Now input pairs.json and localizations.json like this:
    # pairs = reader.load('pairs')
    # localizations = reader.load('localization')
    # read about details here: https://memory-int.psych.upenn.edu/InternalWiki/index.php/RAM_data
    
    regs = []    
    atlas_type = []
    pair_number = []
    has_stein_das = 0
    
    # if localization.json exists get the names from each atlas
    if len(localizations) > 1: 
        # pairs that were recorded and possible pairs from the localization are typically not the same.
        # so need to translate the localization region names to the pairs...which I think is easiest to just do here

        # get an index for every pair in pairs
        loc_translation = loc_to_pairs_translation(pairs, localizations)
        loc_dk_names = ['' for _ in range(len(pairs))]
        loc_mtl_names = copy(loc_dk_names)
        loc_wb_names = copy(loc_dk_names)
        for i,loc in enumerate(loc_translation):
            if loc != ' ': # set it to this when there was no localization.pairs
                if 'atlases.mtl' in localizations: # a few (like 5) of the localization.pairs don't have the MTL atlas
                    loc_mtl_names[loc] = localizations['atlases.mtl']['pairs'][i] # MTL field from pairs in localization.json
                    has_mtl = 1
                else:
                    has_mtl = 0 # so can skip in below
                loc_dk_names[loc] = localizations['atlases.dk']['pairs'][i]
                loc_wb_names[loc] = localizations['atlases.whole_brain']['pairs'][i]   
    for pair_ct in range(len(pairs)):
        try:
            pair_number.append(pair_ct) # just to keep track of what pair this was in subject
            pair_atlases = pairs.iloc[pair_ct] #tal_struct[pair_ct].atlases
            if 'stein.region' in pair_atlases: # if 'stein' in pair_atlases.dtype.names:
                test_region = str(pair_atlases['stein.region'])
                if (test_region is not None) and (len(test_region)>1) and \
                   (test_region not in 'None') and (test_region != 'nan'):
                    regs.append(test_region.lower())
#             if 'stein' in pair_atlases.dtype.names:  ### OLD WAY FROM TAL_STRUCT...leaving as example
#                 if (pair_atlases['stein']['region'] is not None) and (len(pair_atlases['stein']['region'])>1) and \
#                    (pair_atlases['stein']['region'] not in 'None') and (pair_atlases['stein']['region'] != 'nan'):
#                     regs.append(pair_atlases['stein']['region'].lower())
                    atlas_type.append('stein')
                    has_stein_das = 1 # temporary thing just to see where stein/das stopped annotating
                    continue # back to top of for loop
                else:
                    pass # keep going in loop
            if 'das.region' in pair_atlases:
                test_region = str(pair_atlases['das.region'])
                if (test_region is not None) and (len(test_region)>1) and \
                   (test_region not in 'None') and (test_region != 'nan'):
                    regs.append(test_region.lower())
                    atlas_type.append('das')
                    has_stein_das = 1
                    continue
                else:
                    pass
            if len(localizations) > 1 and has_mtl==1:             # 'MTL' from localization.json
                if loc_mtl_names[pair_ct] != '' and loc_mtl_names[pair_ct] != ' ':
                    if str(loc_mtl_names[pair_ct]) != 'nan': # looking for "MTL" field in localizations.json
                        regs.append(loc_mtl_names[pair_ct].lower())
                        atlas_type.append('MTL_localization')
                        continue
                    else:
                        pass
                else:
                    pass
            if len(localizations) > 1:             # 'whole_brain' from localization.json
                if loc_wb_names[pair_ct] != '' and loc_wb_names[pair_ct] != ' ':
                    if str(loc_wb_names[pair_ct]) != 'nan': # looking for "MTL" field in localizations.json
                        regs.append(loc_wb_names[pair_ct].lower())
                        atlas_type.append('wb_localization')
                        continue
                    else:
                        pass
                else:
                    pass
            if 'wb.region' in pair_atlases:
                test_region = str(pair_atlases['wb.region'])
                if (test_region is not None) and (len(test_region)>1) and \
                   (test_region not in 'None') and (test_region != 'nan'):
                    regs.append(test_region.lower())
                    atlas_type.append('wb')
                    continue
                else:
                    pass
            if len(localizations) > 1:             # 'dk' from localization.json
                if loc_dk_names[pair_ct] != '' and loc_dk_names[pair_ct] != ' ':
                    if str(loc_dk_names[pair_ct]) != 'nan': # looking for "dk" field in localizations.json
                        regs.append(loc_dk_names[pair_ct].lower())
                        atlas_type.append('dk_localization')
                        continue
                    else:
                        pass
                else:
                    pass
            if 'dk.region' in pair_atlases:
                test_region = str(pair_atlases['dk.region'])
                if (test_region is not None) and (len(test_region)>1) and \
                   (test_region not in 'None') and (test_region != 'nan'):
                    regs.append(test_region.lower())
                    atlas_type.append('dk')
                    continue
                else:
                    pass
            if 'ind.corrected.region' in pair_atlases: # I don't think this ever has a region label but just in case
                test_region = str(pair_atlases['ind.corrected.region'])
                if (test_region is not None) and (len(test_region)>1) and \
                   (test_region not in 'None') and (test_region not in 'nan'):
                    regs.append(test_region.lower())
                    atlas_type.append('ind.corrected')
                    continue
                else:
                    pass  
            if 'ind.region' in pair_atlases:
                test_region = str(pair_atlases['ind.region'])
                if (test_region is not None) and (len(test_region)>1) and \
                   (test_region not in 'None') and (test_region != 'nan'):
                    regs.append(test_region.lower())
                    atlas_type.append('ind')
                    # [tal_struct[i].atlases.ind.region for i in range(len(tal_struct))] # if you want to see ind atlases for comparison to above
                    # have to run this first though to work in pdb dubugger: globals().update(locals())                  
                    continue
                else:
                    regs.append('No atlas')
                    atlas_type.append('No atlas')
            else: 
                regs.append('No atlas')
                atlas_type.append('No atlas')
        except AttributeError:
            regs.append('error')
            atlas_type.append('error')
    return np.array(regs), np.array(atlas_type), np.array(pair_number), has_stein_das

def get_localization_to_pairs_translation(pairs,localizations):
    # this does the opposite of above...don't think I'll use this one but accidentally made it first so keep it JIC
    loc_pairs = localizations.type.pairs
    loc_pair_labels = np.array(loc_pairs.index)

    loc_to_pairs_idxs = []
    for pair in pairs.label:
        split_pair = pair.split('-') # split into list of 2
        idx = np.where([split_pair==list(lp) for lp in loc_pair_labels])[0]
        if loc_pair_labels[idx].size==0: # if didn't find it, check for reverse
            split_pair.reverse()
            temp_mask = [split_pair==list(lp) for lp in loc_pair_labels]
            idx = np.where(temp_mask)[0]
            if len(idx) == 0:
                idx = ' '
        loc_to_pairs_idxs.extend(idx)

    return loc_to_pairs_idxs

def get_tal_distmat(tal_struct):
        
    #Get distance matrix
    pos = []
    for ts in tal_struct:
        x = ts['atlases']['ind']['x']
        y = ts['atlases']['ind']['y']
        z = ts['atlases']['ind']['z']
        pos.append((x, y, z))
    pos = np.array(pos)
    dist_mat = np.empty((len(pos), len(pos))) # iterate over electrode pairs and build the adjacency matrix
    dist_mat.fill(np.nan)
    for i, e1 in enumerate(pos):
        for j, e2 in enumerate(pos):
            if (i <= j):
                dist_mat[i,j] = np.linalg.norm(e1 - e2, axis=0)
                dist_mat[j,i] = np.linalg.norm(e1 - e2, axis=0)    
    distmat = 1./np.exp(dist_mat/120.)
    
    return distmat


def correct_eeg_offset(sub,session,exp,reader,events):
    # The EEG for recall times for many FR subjects (FR1 and catFR1 in particular) does not align with the events since the 
    # implementation of Unity. This is a temporary fix for the EEG alignment for these subjects before
    # the data is corrected in Rhino. Subject-by-subject details are here:
    # https://docs.google.com/spreadsheets/d/1co5f7-dPOktGIXZJ7uptv0SwBJhf36TuhVSMFqRC0X8/edit?usp=sharing
    # jjsakon 2020-09-22
    # Update 2020-09-29 accounting for sampling rate and raising error if eeg events don't exist
    
    ## Inputs ##
    # sub: subject name (type str)
    # session: session number (type int)
    # exp: experiment, typically 'FR1' or 'catFR1' (type str)
    # reader: typical output from CMLReader function (see cmlreaders documentation)
    # events: dataFrame from reader.load('task_events') for your *RETRIEVAL* events of choice; therefore
    #         aligning eeg to recalls or retrieval_start (although see below program if you want to try to 
    #         align retrieval_start to the end of the beep in addition to fixing the EEG alignment issue). 
    #         Correction is *NOT* needed for encoding alignment
    
    ## Output ## 
    # events: events with eegoffset correctly aligned to the events
    
    import re
    
    sub_num = [int(s) for s in re.findall(r'\d+',sub)] # extract number for sub   
    
    temp_eeg = reader.load_eeg(events=events, rel_start=0, rel_stop=100) # just to get sampling rate

    sr = temp_eeg.samplerate
    sr_factor = 1000/sr
    
    if sum(events.eegoffset==-1)>0:
        raise Exception('Events without EEG (those with events.eegoffset=-1) should be removed before calling correctEEGoffset')
 
    if (sub in ['R1379E','R1385E','R1387E','R1394E','R1402E']) or \
        (sub=='R1404E' and session==0 and exp=='catFR1'): 
        # first 5 true for catFR1 and FR1. R1404E only one catFR1 session has partial beep 
        # for these subs there is a partial beep and 500 ms of eeg lag (see "History of issues 2020-09-08" for examples)
        
        # add time (in units of samples) since the events are already ahead of the eeg
        events.eegoffset = events.eegoffset+int(np.round(500/sr_factor)) # add 500 ms of lag
        
    # subs where unity was implemented for some sessions but not others
    elif (sub=='R1396T' and exp=='catFR1') or (sub=='R1396T' and session==1) or \
         (sub=='R1395M' and exp=='catFR1') or (sub=='R1395M' and exp=='FR1' and session>0):
        events.eegoffset = events.eegoffset+int(np.round(1000/sr_factor))
    
    # do nothing since these sessions were pyEPL so the offset is okay
    elif (sub=='R1406M' and session==0) or (sub=='R1415T' and session==0 and exp=='FR1') or (sub=='R1422T' and exp=='FR1') or \
         sub_num[0]>=1525: # and any sub after R1525J is when we caught he mistake and realigned asterisks_off/beep_off/RET_START
        pass
 
    # remaining unity subs
    elif sub_num[0]>=1397 or sub == 'R1389J': 
        events.eegoffset = events.eegoffset+int(np.round(1000/sr_factor))
        
    return events


def get_bad_channels(tal_struct,elecs_cat,remove_soz_ictal):
    # get the bad channels and soz/ictal/lesion channels from electrode_categories.txt files
    
    # 2021-03-15 rewriting this to put 0 for good electrode, 1 for SOZ, and 2 for bad_electrodes (bad leads or the like)
    # 2022-02-04 this makes more sense to use remove_soz_ictal with 1 for bad_electrodes or SOZ
    # 2022-05-09 fixed the logic so can just keep those channels with bad_bp_mask[channel] == 0
    
    if remove_soz_ictal == 2:
        bad_bp_mask = np.ones(len(tal_struct)) # for this one want to keep ONLY SOZ sites
    else:
        bad_bp_mask = np.zeros(len(tal_struct))
        
    if elecs_cat != []:

        bad_elecs = elecs_cat['bad_channel']
        soz_elecs = elecs_cat['soz'] # + elecs_cat['interictal'] only removing SOZ 2021-03-15

        for idx,tal_row in enumerate(tal_struct):
            elec_labels = tal_row['tagName'].split('-')
            # if there are dashes in the monopolar elec names, need to fix that
            if (len(elec_labels) > 2) & (len(elec_labels) % 2 == 0): # apparently only one of these so don't need an else
                n2 = int(len(elec_labels)/2)
                elec_labels = ['-'.join(elec_labels[0:n2]), '-'.join(elec_labels[n2:])]

            if elec_labels[0] in bad_elecs or elec_labels[1] in bad_elecs:
                bad_bp_mask[idx] = 2 # 2 for bad elecs/bad leads
            if elec_labels[0] in soz_elecs or elec_labels[1] in soz_elecs:
                if remove_soz_ictal == 1:
                    bad_bp_mask[idx] = 1 # if not removing SOZ then just leave as 0
                elif remove_soz_ictal == 2:
                    bad_bp_mask[idx] = 0 # for this one want to keep ONLY SOZ sites
            
    return bad_bp_mask

# def get_start_end_arrays(ripple_array):
#     """
#     Get separate arrays of SWR starts and SWR ends from the full binary array
#     """
#
#     # Shift the ripple array to the right and left by one position
#     shifted_right = np.roll(ripple_array, shift=1, axis=1)
#     shifted_left = np.roll(ripple_array, shift=-1, axis=1)
#
#     # Find the start by looking for a transition from 0 to 1
#     start_array = (ripple_array == 1) & (shifted_right == 0)
#     start_array[:, 0] = ripple_array[:, 0]  # Handle the edge case for the first column
#
#     # Find the end by looking for a transition from 1 to 0
#     end_array = (ripple_array == 1) & (shifted_left == 0)
#     end_array[:, -1] = ripple_array[:, -1]  # Handle the edge case for the last column
#
#     return start_array.astype('uint8'), end_array.astype('uint8')


def get_start_end_arrays(ripple_array):
    start_array = np.zeros((ripple_array.shape), dtype='uint8')
    end_array = np.zeros((ripple_array.shape), dtype='uint8')

    num_trials = ripple_array.shape[0]
    for trial in range(num_trials):
        ripplelogictrial = ripple_array[trial]
        starts, ends = get_logical_chunks(ripplelogictrial)
        temp_row = np.zeros(len(ripplelogictrial))
        temp_row[starts] = 1
        start_array[trial] = temp_row  # time when each SWR starts
        temp_row = np.zeros(len(ripplelogictrial))
        temp_row[ends] = 1
        end_array[trial] = temp_row
    return start_array, end_array


def detect_ripples_hamming(eeg_rip, trans_width, sr, iedlogic):
    """
    detect ripples similar to with Butterworth, but using Norman et al. 2019 algo (based on Stark 2014 algo). Description:
    Then Hilbert, clip extreme to 4 SD, square this clipped, smooth w/ Kaiser FIR low-pass filter with 40 Hz cutoff,
    mean and SD computed across entire experimental duration to define the threshold for event detection
    Events from original (squared but un-clipped) signal >4 SD above baseline were selected as candidate SWR events.
    Duration expanded until ripple power <2 SD. Events <20 ms or >200 ms excluded. Adjacent events <30 ms separation (peak-to-peak) merged.

    :param array eeg_rip: hilbert transformed ieeg data
    """

    candidate_sd = 3
    artifact_buffer = 100 # ms around IED events to remove SWRs
    sr_factor = 1000/sr
    ripple_min = 20/sr_factor # convert each to ms
    ripple_max = 250/sr_factor #200/sr_factor
    min_separation = 30/sr_factor # peak to peak
    orig_eeg_rip = copy(eeg_rip)
    clip_SD = np.mean(eeg_rip) + candidate_sd * np.std(eeg_rip)
    eeg_rip[eeg_rip>clip_SD] = clip_SD # clip at 3SD since detecting at 3 SD now
    eeg_rip = eeg_rip**2 # square
    
    # FIR lowpass 40 hz filter for Norman dtection algo
    nyquist = sr/2
    ntaps40, beta40 = kaiserord(40, trans_width/nyquist)
    kaiser_40lp_filter = firwin(ntaps40, cutoff=40, window=('kaiser', beta40), scale=False, pass_zero='lowpass', fs=sr)
    
    eeg_rip = filtfilt(kaiser_40lp_filter,1.,eeg_rip)
    mean_detection_thresh = np.mean(eeg_rip)
    std_detection_thresh = np.std(eeg_rip)
    
    # now, find candidate events (>mean+3SD) 
    orig_eeg_rip = orig_eeg_rip**2
    candidate_thresh = mean_detection_thresh + candidate_sd * std_detection_thresh
    expansion_thresh = mean_detection_thresh+2*std_detection_thresh
    ripplelogic = orig_eeg_rip >= candidate_thresh # EF1208, will evaluate to squared signal is above threshold
    # remove IEDs detected from Norman 25-60 algo...maybe should do this after expansion to 2SD??
    iedlogic = convolve2d(iedlogic,np.ones((1,artifact_buffer)),'same')>0 # expand to +/- 50 ms from each ied point
    ripplelogic[iedlogic==1] = 0 # EF1208, convert above threshold events to 0 if they are an IED
    
    # expand out to 2SD around surviving events
    num_trials = ripplelogic.shape[0]
    trial_length = ripplelogic.shape[1]
    for trial in range(num_trials):
        ripplelogictrial = ripplelogic[trial]
        starts,ends = get_logical_chunks(ripplelogictrial)
        data_trial = orig_eeg_rip[trial]
        for i,start in enumerate(starts):
            current_time = 0
            while data_trial[start+current_time]>=expansion_thresh:
                if (start+current_time)==-1:
                    break
                else:
                    current_time -=1
            starts[i] = start+current_time+1
        for i,end in enumerate(ends):
            current_time = 0
            while data_trial[end+current_time]>=expansion_thresh:
                if (end+current_time)==trial_length-1:
                    break
                else:
                    current_time += 1
            ends[i] = end+current_time
            
        # remove any duplicates from starts and ends
        starts = np.array(starts); ends = np.array(ends)
        _,start_idxs = np.unique(starts, return_index=True)
        _,end_idxs = np.unique(ends, return_index=True)
        starts = starts[start_idxs & end_idxs]
        ends = ends[start_idxs & end_idxs]

        # remove ripples <min or >max length
        lengths = ends-starts
        ripple_keep = (lengths > ripple_min) & (lengths < ripple_max)
        starts = starts[ripple_keep]; ends = ends[ripple_keep]

        # get peak times of each ripple to combine those < 30 ms separated peak-to-peak
        if len(starts)>1:
            max_idxs = np.zeros(len(starts))
            for ripple in range(len(starts)):
                max_idxs[ripple] = starts[ripple] + np.argmax(data_trial[starts[ripple]:ends[ripple]])                    
            overlappers = np.where(np.diff(max_idxs)<min_separation)

            if len(overlappers[0])>0:
                ct = 0
                for overlap in overlappers:
                    ends = np.delete(ends,overlap-ct)
                    starts = np.delete(starts,overlap+1-ct)
                    ct+=1 # so each time one is removed can still remove the next overlap
                
        # turn starts/ends into a logical array and replace the trial in ripplelogic
        temp_trial = np.zeros(trial_length)
        for i in range(len(starts)):
            temp_trial[starts[i]:ends[i]]=1
        ripplelogic[trial] = temp_trial # place it back in
    return ripplelogic


def detect_ripples_butter(eeg_rip, eeg_ied, eeg_mne, sr):  # ,mstimes):
    ## detect ripples ##
    # input: hilbert amp from 80-120 Hz, hilbert amp from 250-500 Hz, raw eeg. All trials X duration (ms),mstime of each FR event
    # output: ripplelogic and iedlogic, which are trials X duration masks of ripple presence 
    # note: can get all ripple starts/ends using getLogicalChunks custom function
    from scipy import signal,stats

    sr_factor = 1000/sr # have to account for sampling rate since using ms times 
    ripplewidth = 25/sr_factor # ms
    ripthresh = 2 # threshold detection
    ripmaxthresh = 3 # ripple event must meet this maximum
    ied_thresh = 5 # from Staresina, NN 2015 IED rejection
    ripple_separation = 15/sr_factor # from Roux, NN 2017
    artifact_buffer = 100 # per Vaz et al 2019 

    num_trials = eeg_mne.shape[0]
    eeg_rip_z = stats.zscore(eeg_rip,axis=None) # note that Vaz et al averaged over time bins too, so axis=None instead of 0
    eeg_ied_z = stats.zscore(eeg_ied,axis=None)
    eeg_diff = np.diff(eeg_mne) # measure eeg gradient and zscore too
    eeg_diff = np.column_stack((eeg_diff,eeg_diff[:,-1]))# make logical arrays same size
    eeg_diff = stats.zscore(eeg_diff,axis=None)

    # convert to logicals and remove IEDs
    ripplelogic = eeg_rip_z>ripthresh
    broadlogic = eeg_ied_z>ied_thresh 
    difflogic = abs(eeg_diff)>ied_thresh
    iedlogic = broadlogic | difflogic # combine artifactual ripples
    iedlogic = signal.convolve2d(iedlogic,np.ones((1,artifact_buffer)),'same')>0 # expand to +/- 100 ms
    ripplelogic[iedlogic==1] = 0 # remove IEDs

    # loop through trials and remove ripples
    for trial in range(num_trials):
        ripplelogictrial = ripplelogic[trial]        
        if np.sum(ripplelogictrial)==0:
            continue
        hilbamptrial = eeg_rip_z[trial]

        starts,ends = get_logical_chunks(ripplelogictrial)  # get chunks of 1s that are putative SWRs
        for ripple in range(len(starts)):
            if ends[ripple]+1-starts[ripple] < ripplewidth or \
            max(abs(hilbamptrial[starts[ripple]:ends[ripple]+1])) < ripmaxthresh:
                ripplelogictrial[starts[ripple]:ends[ripple]+1] = 0
        ripplelogic[trial] = ripplelogictrial # reassign trial with ripples removed

    # join ripples less than 15 ms separated 
    for trial in range(num_trials):
        ripplelogictrial = ripplelogic[trial]
        if np.sum(ripplelogictrial)==0:
            continue
        starts,ends = get_logical_chunks(ripplelogictrial)
        if len(starts)<=1:
            continue
        for ripple in range(len(starts)-1): # loop through ripples before last
            if (starts[ripple+1]-ends[ripple]) < ripple_separation:            
                ripplelogictrial[ends[ripple]+1:starts[ripple+1]] = 1
        ripplelogic[trial] = ripplelogictrial # reassign trial with ripples removed      
    
    return ripplelogic,iedlogic  # ripple_mstimes


def detect_ripples_staresina(eeg_rip, sr):
    # detect ripples using Staresina et al 2015 NatNeuro algo
    window_size = 20 # in ms
    min_duration = 38 # 38 ms

    sr_factor = 1000/sr

    rip2 = np.power(eeg_rip,2)
    window = np.ones(int(window_size/sr_factor))/float(window_size/sr_factor)
    rms_values = []
    # get rms for 20 ms moving avg across all trials (confirmed this conv method is same as moving window)
    for eeg_tr in rip2:
        # from https://stackoverflow.com/questions/8245687/numpy-root-mean-squared-rms-smoothing-of-a-signal
        rms_values.append(np.sqrt(np.convolve(eeg_tr, window, 'same')))  # same means it pads at ends, but doesn't matter with buffers anyway
    rms_thresh = np.percentile(rms_values, 99)  # 99th %ile threshold
    binary_array = rms_values >= rms_thresh

    # now find those with minimum duration between start/end for each trial and if they have 3 peaks/troughs keep them

    ripplelogic = np.zeros((np.shape(binary_array)[0],np.shape(binary_array)[1]))
    for i_trial in range(len(binary_array)):
        binary_trial = binary_array[i_trial]
        starts,ends = get_logical_chunks(binary_trial)
        candidate_events = (np.array(ends)-np.array(starts)+1)>=(min_duration/sr_factor)
        starts = np.array(starts)[candidate_events]
        ends = np.array(ends)[candidate_events]
        ripple_trial = np.zeros(len(binary_trial))
        for i_cand in range(len(starts)):
            # get raw eeg plus half of moving window. idx shouldn't get past end since ripplelogic is smaller than eeg_rip
            eeg_segment = eeg_rip[i_trial].values[int(starts[i_cand]+window_size/sr_factor/2-1):int(ends[i_cand]+window_size/sr_factor/2+1)] # add point on either side for 3 MA filter 
            peaks,_ = lmax(eeg_segment,3) # Matlab function suggested by Bernhard I rewrote for Python. Basically a moving average 3 filter to find local maxes
            troughs,_ = lmin(eeg_segment,3)
            if ((len(peaks)>=3) | (len(troughs)>=3)):
                ripplelogic[i_trial,starts[i_cand]:ends[i_cand]] = 1
    return ripplelogic


def downsample_binary(array, factor):
    # input should be trial X time binary matrix
    array_save = np.array([])
    if factor-int(factor) == 0:  # if an integer
        for t in range(array.shape[0]):  # from https://stackoverflow.com/questions/20322079/downsample-a-1d-numpy-array
            array_save = superVstack(array_save, array[t].reshape(-1, int(factor)).mean(axis=1))
    else:
        # when dividing by non-integer, can just use FFT and round to get new sampling
        from scipy.signal import resample
        if array.shape[1]/factor-int(array.shape[1]/factor) != 0:
            print('Did not get whole number array for down sampling...rounding to nearest 100')
        new_sampling = int(round((array.shape[1]/factor)/100))*100
        for t in range(array.shape[0]):
            array_save = superVstack(array_save,np.round(resample(array[t],new_sampling)))
    return array_save


def ptsa_to_mne(eegs, time_length):  # in ms
    # convert ptsa to mne    
    import mne

    sr = int(np.round(eegs.samplerate))  # get samplerate...round 1st since get like 499.7 sometimes
    eegs = eegs[:, :, :].transpose('event', 'channel', 'time') # make sure right order of names

    time = [x/1000 for x in time_length]  # convert to s for MNE
    clips = np.array(eegs[:, :, int(np.round(sr*time[0])):int(np.round(sr*time[1]))])
    mne_evs = np.empty([clips.shape[0], 3]).astype(int)
    mne_evs[:, 0] = np.arange(clips.shape[0]) # at each timepoint
    mne_evs[:, 1] = clips.shape[2] # 0
    mne_evs[:, 2] = list(np.zeros(clips.shape[0]))
    event_id = dict(resting=0)
    tmin=0.0
    info = mne.create_info([str(i) for i in range(eegs.shape[1])], sr, ch_types='eeg',verbose=False)  
    
    arr = mne.EpochsArray(np.array(clips), info, mne_evs, tmin, event_id, verbose=False)
    return arr


def fastSmooth(a,window_size): # I ended up not using this one. It's what Norman/Malach use (a python
     # implementation of matlab nanfastsmooth, but isn't actually triangular like it says in paper)
    
    # a: NumPy 1-D array containing the data to be smoothed
    # window_size: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    if np.mod(window_size,2)==0:
        print('sliding window must be odd!!')
        print('See https://stackoverflow.com/questions/40443020/matlabs-smooth-implementation-n-point-moving-average-in-numpy-python')
    out0 = np.convolve(a,np.ones(window_size,dtype=int),'valid')/window_size    
    r = np.arange(1,window_size-1,2)
    start = np.cumsum(a[:window_size-1])[::2]/r
    stop = (np.cumsum(a[:-window_size:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def triangleSmooth(data,smoothing_triangle): # smooth data with triangle filter using padded edges
    
    # problem with this smoothing is when there's a point on the edge it gives too much weight to 
    # first 2 points (3rd is okay). E.g. for a 1 on the edge of all 0s it gives 0.66, 0.33, 0.11
    # while for a 1 in the 2nd position of all 0s it gives 0.22, 0.33, 0.22, 0.11 (as you'd want)
    # so make sure you don't use data in first 2 or last 2 positions since that 0.66/0.33 is overweighted
    
    factor = smoothing_triangle-3 # factor is how many points from middle does triangle go?
    # this all just gets the triangle for given smoothing_triangle length
    f = np.zeros((1+2*factor))
    for i in range(factor):
        f[i] = i+1
        f[-i-1] = i+1
    f[factor] = factor + 1
    triangle_filter = f / np.sum(f)

    padded = np.pad(data, factor, mode='edge') # pads same value either side
    smoothed_data = np.convolve(padded, triangle_filter, mode='valid')
    return smoothed_data

def fullPSTH(point_array,binsize,smoothing_triangle,sr,start_offset):
    # point_array is binary point time (spikes or SWRs) v. trial
    # binsize in ms, smoothing_triangle is how many points in triangle kernel moving average
    sr_factor = (1000/sr)
    num_trials = point_array.shape[0]
    xtimes = np.where(point_array)[1]*sr_factor # going to do histogram so don't need to know trial #s
    
    nsamples = point_array.shape[1]
    ms_length = nsamples*sr_factor
    last_bin = binsize*np.ceil(ms_length/binsize)

    edges = np.arange(0,last_bin+binsize,binsize)
    bin_centers = edges[0:-1]+binsize/2+start_offset

    count = np.histogram(xtimes,bins=edges);
    norm_count = count[0]/np.array((num_trials*binsize/1000))

    #smoothed = fastSmooth(norm_count[0],5) # use triangular instead, although this gives similar answer
    if smoothing_triangle==1:
        PSTH = norm_count
    else:
        PSTH = triangleSmooth(norm_count,smoothing_triangle)
    return PSTH,bin_centers

def binBinaryArray(start_array,bin_size,sr_factor):
    # instead of PSTH, get a binned binary array that keeps the trials but bins over time
    bin_in_sr = bin_size/sr_factor
    bins = np.arange(0,start_array.shape[1],bin_in_sr) #start_array.shape[1]/bin_size*sr_factor
    
    # only need to do this for ripples (where bin_size is 100s of ms). For z-scores (which is already averaged) don't convert
    if bin_size > 50: # this is just a dumb way to make sure it's not a z-score
        bin_to_hz = 1000/bin_size*bin_in_sr # factor that converts binned matrix to Hz
    else:
        bin_to_hz = 1
    binned_array = np.zeros((start_array.shape[0], num_bins))
    # note this will be at instantaeous rate bin_in_sr multiple lower (e.g. 100 ms bin/2 sr = 50x)

    for i, row in enumerate(start_array):
        for j, start in enumerate(bins[:-1]):
            binned_array[i, j] = bin_to_hz * np.mean(row[int(start):int(start + bin_in_sr)])

    return binned_array

def getSubSessPredictors(sub_names,sub_sess_names,trial_nums,electrode_labels,channel_coords):
    '''
     get arrays of predictors for each trial so can set up ME model
    2020-08-31 get electrode labels too
    '''
    
    subject_name_array = []
    session_name_array = []
    electrode_array = []
    channel_coords_array = []
    trial_ct = 0
    for ct,subject in enumerate(sub_names):    
        trials_this_loop = int(trial_nums[ct])
        trial_ct = trial_ct + trials_this_loop 
        # update each array with subjects, sessions, and other prdictors
        subject_name_array.extend(np.tile(subject,trials_this_loop))
        session_name_array.extend(np.tile(sub_sess_names[ct],trials_this_loop))
        electrode_array.extend(np.tile(electrode_labels[ct],trials_this_loop))
        channel_coords_array.extend(np.tile(channel_coords[ct],(trials_this_loop,1))) # have to tile trials X 1 or it extends into a vector
        
    return subject_name_array,session_name_array,electrode_array,channel_coords_array

def getSubSessPredictorsWithChannelNums(sub_names,sub_sess_names,trial_nums,electrode_labels,channel_coords,
                                        channel_nums=[]):
    # get arrays of predictors for each trial so can set up ME model
    # 2020-08-31 get electrode labels too
    # EF1208, I'm modifying this function so we don't need the non channel nums version 
    
    subject_name_array = []
    session_name_array = []
    electrode_array = []
    channel_coords_array = []
    channel_nums_array = []

    trial_ct = 0
    for ct,subject in enumerate(sub_names):    
        trials_this_loop = int(trial_nums[ct])
        trial_ct = trial_ct + trials_this_loop 
        # update each array with subjects, sessions, and other prdictors
        subject_name_array.extend(np.tile(subject,trials_this_loop))
        session_name_array.extend(np.tile(sub_sess_names[ct],trials_this_loop))
        electrode_array.extend(np.tile(electrode_labels[ct],trials_this_loop))
        channel_coords_array.extend(np.tile(channel_coords[ct],(trials_this_loop,1))) # have to tile trials X 1 or it extends into a vector
        if len(channel_nums) > 0: 
            channel_nums_array.extend(np.tile(channel_nums[ct],trials_this_loop))
        
    return subject_name_array,session_name_array,electrode_array,channel_coords_array,channel_nums_array

def getMixedEffectCIs(binned_start_array,subject_name_array,session_name_array):
    # take a binned array of ripples and find the mixed effect confidence intervals
    # note that output is the net ± distance from mean
    import statsmodels.formula.api as smf

    # now, to set up ME regression, append each time_bin to bottom and duplicate
    mean_values = []
    CIs = []
    for time_bin in range(np.shape(binned_start_array)[1]):
        ripple_rates = binned_start_array[:,time_bin]
        CI_df = pd.DataFrame(data={'session':session_name_array,'subject':subject_name_array,'ripple_rates':ripple_rates})
        # now get the CIs JUST for this time bin
        vc = {'session':'0+session'} # EF1208, adding 0+ excludes random intercept 
        get_bin_CI_model = smf.mixedlm("ripple_rates ~ 1", CI_df, groups="subject", vc_formula=vc)
        bin_model = get_bin_CI_model.fit(reml=False, method='nm',maxiter=2000)
        mean_values.append(bin_model.params.Intercept)
        CIs = superVstack(CIs,bin_model.conf_int().iloc[0].values)
    # get CI distances at each bin by subtracting from means
    CI_plot = np.array(CIs.T)
    CI_plot[0,:] = mean_values - CI_plot[0,:] # - difference to subtract from PSTH
    CI_plot[1,:] = CI_plot[1,:] - mean_values # + difference to add to PSTH
    
    return CI_plot

def getMixedEffectSEs(binned_start_array,subject_name_array,session_name_array):
    # take a binned array of ripples and find the mixed effect SEs at each bin
    # note that output is the net ± distance from mean
    import statsmodels.formula.api as smf

    # now, to set up ME regression, append each time_bin to bottom and duplicate
    mean_values = []
    SEs = [] #CIs = []
    for time_bin in range(np.shape(binned_start_array)[1]):
        ripple_rates = binned_start_array[:,time_bin]
        SE_df = pd.DataFrame(data={'session':session_name_array,'subject':subject_name_array,'ripple_rates':ripple_rates})
        # now get the SEs JUST for this time bin
        vc = {'session':'0+session'}
        get_bin_CI_model = smf.mixedlm("ripple_rates ~ 1", SE_df, groups="subject", vc_formula=vc)
        bin_model = get_bin_CI_model.fit(reml=True, method='nm',maxiter=2000)
        mean_values.append(bin_model.params.Intercept)
        SEs.append(bin_model.bse_fe)
    # get SE distances at each bin
    SE_plot = superVstack(np.array(SEs).T,np.array(SEs).T)
    
    return SE_plot

def getMixedEffectMeanSEs(binned_start_array,subject_name_array,session_name_array,elec_name_array = []):
    # take a binned array of ripples and find the mixed effect SEs at each bin
    # note that output is the net ± distance from mean
    import statsmodels.formula.api as smf

    # now, to set up ME regression, append each time_bin to bottom and duplicate
    mean_values = []
    SEs = [] #CIs = []
    for time_bin in range(np.shape(binned_start_array)[1]):
        ripple_rates = binned_start_array[:,time_bin]
        if elec_name_array == []: # if not defined just do session in subs
            SE_df = pd.DataFrame(data={'session':session_name_array,'subject':subject_name_array,'ripple_rates':ripple_rates})
            # now get the CIs JUST for this time bin
            vc = {'session':'0+session'}
            get_bin_CI_model = smf.mixedlm("ripple_rates ~ 1", SE_df, groups="subject", vc_formula=vc)
        else:
            SE_df = pd.DataFrame(data={'session':session_name_array,'subject':subject_name_array,'ripple_rates':ripple_rates,
                                       'elec':elec_name_array})
            # now get the SEs JUST for this time bin
            vc = {'session':'0+session','elec':'0+elec'}
            get_bin_CI_model = smf.mixedlm("ripple_rates ~ 1", SE_df, groups="subject", vc_formula=vc)            
        bin_model = get_bin_CI_model.fit(reml=True, method='nm',maxiter=2000)
        mean_values.append(bin_model.params.Intercept)
        SEs.append(bin_model.bse_fe.Intercept)

    # get SE distances at each bin
    SE_plot = superVstack(np.array(SEs).T,np.array(SEs).T)
    
    return mean_values,SE_plot

def fixSEgaps(SE_plot):
    # fill in places where ME model for that bin didn't converge
    # shouldn't be an issue once we move from 40% to 100% of the data!
    for i,tbin in enumerate(SE_plot[0]):
        if isNaN(tbin):
            if (i>0) and (i<len(SE_plot[0])):
                SE_plot[0][i] = np.mean([SE_plot[0][i-1],SE_plot[0][i+1]])
                SE_plot[1][i] = np.mean([SE_plot[1][i-1],SE_plot[1][i+1]])
            elif i>0:
                SE_plot[0][i] = SE_plot[0][i-1]
                SE_plot[1][i] = SE_plot[1][i-1]
            elif i<len(SE_plot[0]):
                SE_plot[0][i] = SE_plot[0][i+1]
                SE_plot[1][i] = SE_plot[1][i+1]
    return SE_plot

def MEstatsAcrossBins(binned_start_array,subject_name_array,session_name_array):
    # returns mixed effect model for the given trial X bins array by comparing bins
    import statsmodels.formula.api as smf
    
    bin_label = []
    session_name = []
    subject_name = []
    ripple_rates = []
    # now, to set up ME pairwise stats, append each time_bin to bottom and duplicate
    for time_bin in range(np.shape(binned_start_array)[1]): 
        session_name.extend(session_name_array)
        subject_name.extend(subject_name_array)
        bin_label.extend(np.tile(str(time_bin),binned_start_array.shape[0]))
        ripple_rates.extend(binned_start_array[:,time_bin])

    bin_df = pd.DataFrame(data={'session':session_name,'subject':subject_name,
                               'bin':bin_label,'ripple_rates':ripple_rates})
    vc = {'session':'0+session'}
    # note, even if there's only one subject being used here, as I do for the t-score histograms
    # this format will still use sessions as random grouping (in other words, same as if I used
    # "session" instead of "subject" below for a single patient)
    sig_bin_model = smf.mixedlm("ripple_rates ~ bin", bin_df, groups="subject", vc_formula=vc) #, re_formula="bin") # adding this really screwed up the model even though it claims it converged...since it's only two bins the intercept must mess thigns up 2022-05-02
    bin_model = sig_bin_model.fit(reml=True, method='nm',maxiter=2000)
    return bin_model

def MEstatsAcrossCategories(binned_recalled_array,sub_recalled,sess_recalled,binned_forgot_array,sub_forgot,sess_forgot):
    # here looking at only a single bin but now comparing across categories (e.g. remembered v. forgotten)
    import statsmodels.formula.api as smf
    
    category_label = []
    session_name = []
    subject_name = []
    ripple_rates = []
    # now, to set up ME pairwise stats, append each time_bin to bottom and duplicate
    for category in range(2): 
        if category == 0: # remembered then forgot
            binned_start_array = binned_recalled_array
            session_name_array = sess_recalled
            subject_name_array = sub_recalled
        else:
            binned_start_array = binned_forgot_array
            session_name_array = sess_forgot
            subject_name_array = sub_forgot        
        session_name.extend(session_name_array)
        subject_name.extend(subject_name_array)
        category_label.extend(np.tile(str(category),binned_start_array.shape[0]))
        ripple_rates.extend(binned_start_array[:,0]) # even though only a vector gotta call it out of this list so it's hashable 

    bin_df = pd.DataFrame(data={'session':session_name,'subject':subject_name,
                               'category':category_label,'ripple_rates':ripple_rates})
    vc = {'session':'0+session'}
    sig_bin_model = smf.mixedlm("ripple_rates ~ category", bin_df, groups="subject", vc_formula=vc,re_formula="category")
    bin_model = sig_bin_model.fit(reml=True, method='nm',maxiter=2000)
    return bin_model

def getCategoryRepeatIndicator(sess,electrode_array,session_name_array,category_array):
    # get an array indicating if each word is from the 1st, 2nd, or 3rd use of a given category in a session
    
    first_elec = np.unique(electrode_array[session_name_array == sess])[0] # just take 1st electrode since it's the same categories for each
    elec_category_array = category_array[( (session_name_array == sess) & (electrode_array == first_elec) )]

    # how many words from each category?
    num_each_cat = []
    for word in np.unique(elec_category_array):
        num_each_cat.append(sum(elec_category_array==word))
    # print('Words presented per category:')
    # np.array(num_each_cat)
    # sum(num_each_cat)

    # create empty list of right size
    category_repeat_array = np.zeros(sum(num_each_cat)) # is this the 1st, 2nd, or 3rd time this category has been used in the session?

    idx_sort = np.argsort(elec_category_array)
    sorted_elec_category_array = elec_category_array[idx_sort]
    vals, idx_start, count = np.unique(sorted_elec_category_array, return_counts=True, return_index=True)
    # splits the indices into separate arrays for each category
    separate_category_arrays = np.split(idx_sort, idx_start[1:])

    # now take each separate category, sort it by idx number, and indicate if it's the 1st, 2nd, or 3rd time used in a given session

    for cats in separate_category_arrays:
        ct = 0
        cat_usage_ct = 1 # start at 1 and go to 3
        sorted_cats = np.sort(cats)
        for word in sorted_cats:
            category_repeat_array[word] = cat_usage_ct # place the usage of this category in the right index
            ct+=1
            if ct % 4 == 0:
                cat_usage_ct+=1 # if went through 4 words already, can bump up usage count
                
    return category_repeat_array

def getRepFRPresentationArray(session_name_array,list_num_key,session_events):
    # for RepFR create an array of whether word presentations are 1st, 2nd, or 3rd of their type (type being 1p, 2p, or 3p)

    session_names = np.unique(session_name_array)
    presentation_array = []

    for sess in session_names:
        
        sess_chs = np.unique(session_events.channel_num)
        
        for ch in sess_chs: # for each elec pair separately to mimic clusterRun order
        
            sess_ch_list_nums = np.unique(list_num_key[( (session_name_array==sess) & (session_events.channel_num==ch) )])

            for ln in sess_ch_list_nums:
                # for each list in each session, figure out which encoding events are 1p/2p/3p
                list_item_nums = session_events[( (session_name_array==sess) & (list_num_key==ln) & (session_events.channel_num==ch) )].item_num 

                if len(list_item_nums) % 27 == 0: # make sure all the words were presented for this list
                    unique_item_nums,item_counts = np.unique(list_item_nums,return_counts=True)
                    item_counter = np.zeros(len(unique_item_nums))
                    list_pres_nums = []
                    for i_num,item in enumerate(list_item_nums):
                        item_counter[findInd(unique_item_nums==item)]+=1
                        list_pres_nums.append(int(item_counter[findInd(unique_item_nums==item)]))
                        # reset if reached end of a full list's worth of presentations (6 3p(which would be 3+2+1), 3 2p (2+1), and 3 1p adds to 48)
                        if ( ((i_num+1) % 27 == 0) & (sum(item_counter) == 27) ): # this shouldn't happen now
                            item_counter = np.zeros(len(unique_item_nums))
                else:
                    # I ran this for patients up until 2021-10-27 and only R1579T-1, list_num=16 had this issue (it's accounted for in updated_recalls)
                    print('ONE OF YOUR LISTS DID NOT HAVE 27 WORDS!!!')
                    print(sess)
                    print(ln)
                presentation_array.extend(list_pres_nums)
    presentation_array = np.array(presentation_array)

    return presentation_array

def getRepFRRepeatArray(session_name_array,list_num_key,session_events):
    # for RepFR create an array of whether word presentations are 1p (presented only once, 2p, or 3p
    
    # cannot use session_events.repeats because I have rejiggered the indices 
    # (those are saved electrode->list while I'm loading all these arrays as sub_sess->ln. 
    # So get repeat_array via sub_sess->ln so it matches

    session_names = np.unique(session_name_array)
    repeat_array = []

    for sess in session_names:
        
        sess_chs = np.unique(session_events.channel_num)
        
        for ch in sess_chs: # for each elec pair separately to mimic clusterRun order
            
            sess_ch_list_nums = np.unique(list_num_key[( (session_name_array==sess) & (session_events.channel_num==ch) )])

            for ln in sess_ch_list_nums:
                # for each list in each session, figure out which encoding events are 1p/2p/3p
                list_item_nums = session_events[((session_name_array==sess) & (list_num_key==ln) & (session_events.channel_num==ch) )].item_num

                if len(list_item_nums) % 27 == 0: # make sure all the words were presented for this list
                    unique_item_nums,item_counts = np.unique(list_item_nums,return_counts=True)
                    pres_nums = item_counts/(sum(item_counts)/27) # this divides by number of electrodes to get back to 1p/2p/3p instead of multiples of it
                    list_pres_nums = []
                    for item_num in list_item_nums:
                        list_pres_nums.append(int(pres_nums[findInd(unique_item_nums==item_num)]))
                else:
                    # I ran this for patients up until 2021-10-27 and only R1579T-1, list_num=16 had this issue (it's accounted for in updated_recalls)
                    print('ONE OF YOUR LISTS DID NOT HAVE 27 WORDS!!!')
                    print(sess)
                    print(ln)
                repeat_array.extend(list_pres_nums)
    repeat_array = np.array(repeat_array)

    return repeat_array

def bootPSTH(point_array,binsize,smoothing_triangle,sr,start_offset): # same as above, but single output so can bootstrap
    # point_array is binary point time (spikes or SWRs) v. trial
    # binsize in ms, smoothing_triangle is how many points in triangle kernel moving average
    sr_factor = (1000/sr)
    num_trials = point_array.shape[0]
    xtimes = np.where(point_array)[1]*sr_factor # going to do histogram so don't need to know trial #s
    
    nsamples = point_array.shape[1]
    ms_length = nsamples*sr_factor
    last_bin = binsize*np.ceil(ms_length/binsize)

    edges = np.arange(0,last_bin+binsize,binsize)
    bin_centers = edges[0:-1]+binsize/2+start_offset

    count = np.histogram(xtimes,bins=edges);
    norm_count = count/np.array((num_trials*binsize/1000))
    #smoothed = fastSmooth(norm_count[0],5) # use triangular instead, although this gives similar answer
    PSTH = triangleSmooth(norm_count[0],smoothing_triangle)
    return PSTH

def makePairwiseComparisonPlot(comp_data,comp_names,col_names,figsize=(7,5)):
    # make a pairwise comparison errorbar plot with swarm and FDR significance overlaid
    # comp_data: list of vectors of pairwise comparison data
    # comp_names: list of labels for each pairwise comparison
    # col_names: list of 2 names: 1st is what is in data, 2nd is what the grouping of the labels 
    
    import pandas as pd
    from scipy.stats import ttest_1samp
    from statsmodels.stats.multitest import fdrcorrection
    import matplotlib.pyplot as plt
    import seaborn as sb

    # make dataframe
    comp_df = pd.DataFrame(columns=col_names)
    for i in range(len(comp_data)):
        # remove NaNs
        comp_data[i] = np.array(comp_data[i])[~np.isnan(comp_data[i])]
        
        temp = pd.DataFrame(columns=col_names)
        temp['pairwise_data'] = comp_data[i]
        temp['grouping'] = np.tile(comp_names[i],len(comp_data[i]))
        comp_df = comp_df.append(temp,ignore_index=False, sort=True)

    figSub,axSub = plt.subplots(1,1, figsize=figsize)
    axSub.bar( range(len(comp_names)), [np.mean(i) for i in comp_data], 
              yerr = [2*np.std(i)/np.sqrt(len(i)) for i in comp_data],
              color = (0.5,0.5,0.5), error_kw={'elinewidth':18, 'ecolor':(0.7,0.7,0.7)} )
    sb.swarmplot(x='grouping', y='pairwise_data', data=comp_df, ax=axSub, color=(0.8,0,0.8), alpha=0.3)
    axSub.plot([axSub.get_xlim()[0],axSub.get_xlim()[1]],[0,0],linewidth=2,linestyle='--',color=(0,0,0),label='_nolegend_')
    for i in range(len(comp_names)):
        plt.text(i-0.2,-4,'N='+str(len(comp_data[i])))
    # put *s for FDR-corrected significance
    p_values = []
    for i in range(len(comp_data)):
        p_values.append(ttest_1samp(comp_data[i],0)[1])
    sig_after_correction = fdrcorrection(p_values)[0]
    for i in range(len(sig_after_correction)):
        if sig_after_correction[i]==True:
            plt.text(i-0.07,4.15,'*',size=20)
    print('FDR-corrected p-values for each:')
    fdr_pvalues = fdrcorrection(p_values)[1]

    # axSub.set(xticks=[],xticklabels=comp_names)
    axSub.set_ylim(-4.5,4.5)
    plt.xlabel(col_names[0])
    plt.ylabel(col_names[1])
    figSub.tight_layout()
    
    print(fdr_pvalues)
    return fdr_pvalues


def SubjectDataFrames(sub_list):
    if isinstance(sub_list, str):
        sub_list = [sub_list]
    
    df = get_data_index('all')
    indices_list = [df['subject']==sub for sub in sub_list]
    indices = functools.reduce(lambda x,y: x|y, indices_list)
    df_matched = df[indices]
    return df_matched


def GetElectrodes(sub,start,stop):
    df_sub = SubjectDataFrames(sub)
    reader = CMLReadDFRow(next(df_sub.itertuples()))
    evs = reader.load('events')
    enc_evs = evs[evs.type=='WORD']
    eeg = reader.load_eeg(events=enc_evs, rel_start=start, rel_stop=stop, clean=True)
    return eeg.to_ptsa().channel.values

def MakeLocationFilter(scheme, location):
    return [location in s for s in [s if s else '' for s in scheme.iloc()[:]['ind.region']]]


def lmax(x,filt):
    """
    Find local maxima in vector X,where
	LMVAL is the output vector with maxima values, INDD  is the
	corresponding indexes, FILT is the number of passes of the small
	running average filter in order to get rid of small peaks.  Default
	value FILT =0 (no filtering). FILT in the range from 1 to 3 is
	usially sufficient to remove most of a small peaks
	For example:
	xx=0:0.01:35; y=sin(xx) + cos(xx ./3);
	plot(xx,y); grid; hold on;
	[b,a]=lmax(y,2)
	 plot(xx(a),y(a),'r+')
	see also LMIN
    translated from Matlab by J. Sakon 2021-11-16
    """
    
    x_orig = copy(x)
    num_pts = len(x)
    fltr = np.array([1, 1, 1])/3
    x1 = x[0]
    x2 = x[-1]
    for jj in range(filt):
        c = np.convolve(fltr,x)
        x = c[1:num_pts+2]
        x[0] = x1
        x[-1] = x2

    lmval = []; indd = []
    i=1 # start at 2nd point
    while i < num_pts-1:
        if x[i] > x[i-1]:
            if x[i] > x[i+1]:
                lmval.append(x[i])
                indd.append(i)
            elif (x[i] == x[i + 1]) & (x[i] == x[i + 2]):
                i = i+2 # skip 2 points
            elif x[i] == x[i+1]:
                i = i+1 # skip 1 point
        i = i+1
    if (filt > 0) & (len(indd) > 0):
        if (indd[0] <= 3) | ((indd[-1] + 2) > num_pts):
            rng = 1
        else:
            rng = 2
        temp_val = []
        temp_ind = []
        for ii in range(len(indd)):
            temp_val.append(np.max(x_orig[indd[ii]-rng:indd[ii]+rng]))
            max_idx = np.argmax(x_orig[indd[ii]-rng:indd[ii]+rng])
            temp_ind.append(indd[ii]+max_idx-rng-1)  
        lmval = temp_val
        indd = temp_ind

    return lmval,indd

def lmin(x,filt):
    # translated from Matlab by J. Sakon 2021-11-16. See lmax above for description
    
    x_orig = copy(x)
    num_pts = len(x)
    fltr = np.array([1, 1, 1])/3
    x1 = x[0]
    x2 = x[-1]
    for jj in range(filt):
        c = np.convolve(fltr,x)
        x = c[1:num_pts+2]
        x[0] = x1
        x[-1] = x2

    lmval = []; indd = []
    i=1 # start at 2nd point
    while i < num_pts-1:
        if x[i] < x[i-1]:
            if x[i] < x[i+1]:
                lmval.append(x[i])
                indd.append(i)
            elif ( (x[i] == x[i+1]) & (x[i]==x[i+2]) ):
                i = i+2 # skip 2 points
            elif x[i] == x[i+1]:
                i = i+1 # skip 1 point
        i = i+1
    if ( (filt > 0) & (len(indd)>0 ) ):
        if ( (indd[0] <= 3) | ((indd[-1]+2) > num_pts) ):
            rng = 1
        else:
            rng = 2
        temp_val = []
        temp_ind = []
        for ii in range(len(indd)):
            temp_val.append(np.min(x_orig[indd[ii]-rng:indd[ii]+rng]))
            max_idx = np.argmin(x_orig[indd[ii]-rng:indd[ii]+rng])
            temp_ind.append(indd[ii]+max_idx-rng-1)  
        lmval = temp_val
        indd = temp_ind

    return lmval,indd

class SubjectStats():
    def __init__(self):
        self.sessions = 0
        self.lists = []
        self.recalled = []
        self.intrusions_prior = []
        self.intrusions_extra = []
        self.repeats = []
        self.num_words_presented = []
    
    def Add(self, evs):
        enc_evs = evs[evs.type=='WORD']
        rec_evs = evs[evs.type=='REC_WORD']
        
        # Trigger exceptions before data collection happens
        enc_evs.recalled
        enc_evs.intrusion
        enc_evs.item_name
        if 'trial' in enc_evs.columns:
            enc_evs.trial
        else:
            enc_evs.list

        self.sessions += 1
        if 'trial' in enc_evs.columns:
            self.lists.append(len(enc_evs.trial.unique()))
        else:
            self.lists.append(len(enc_evs.list.unique()))
        self.recalled.append(sum(enc_evs.recalled))
        self.intrusions_prior.append(sum(rec_evs.intrusion > 0))
        self.intrusions_extra.append(sum(rec_evs.intrusion < 0))
        words = rec_evs.item_name
        self.repeats.append(len(words) - len(words.unique()))
        self.num_words_presented.append(len(enc_evs.item_name))
        
    def ListAvg(self, arr):
        return np.sum(arr)/np.sum(self.lists)
    
    def RecallFraction(self):
        return np.sum(self.recalled)/np.sum(self.num_words_presented)
    
def SubjectStatTable(subjects):
    """ Prepare LaTeX table of subject stats """

    table = ''
    try:
        table += '\\begin{tabular}{lrrrrrr}\n'
        table += ' & '.join('\\textbf{{{0}}}'.format(h) for h in [
            'Subject',
            '\\# Sessions',
            '\\# Lists',
            'Avg Recalled',
            'Prior Intrusions',
            'Extra Intrusions',
            'Repeats'
        ])
        table += ' \\\\\n'

        for sub in subjects:
            df_sub = SubjectDataFrames(sub) # at this level 
            stats = SubjectStats()
            for row in df_sub.itertuples():
                reader = CMLReadDFRow(row) # "row" is whole row of dataframe (1 session in this case)
                evs = reader.load('task_events')
                stats.Add(evs)
            table += ' & '.join([sub, str(stats.sessions)] +
                ['{:.2f}'.format(x) for x in [
                    np.mean(stats.lists),
                    stats.ListAvg(stats.recalled),
                    stats.ListAvg(stats.intrusions_prior),
                    stats.ListAvg(stats.intrusions_extra),
                    stats.ListAvg(stats.repeats)
                ]]) + ' \\\\\n'
        
        table += '\\end{tabular}\n'
    except Exception as e:
        print (table)
        
    
    return table  

def get_power(eeg, tstart, tend, freqs=[5,8]):

    eeg = eeg.resample(500)  #downsample to ___ Hz

    #Use MNE to get multitaper power in theta and HFA bands
    from mne.time_frequency import psd_multitaper
    theta_pow, fdone = psd_multitaper(eeg, fmin=freqs[0], fmax=freqs[1], tmin=tstart, 
                                      tmax=tend, verbose=False)  #power is (events, elecs, freqs)
    theta_pow_avg = np.mean(theta_pow, -1)  #Average across freq_bands
    return theta_pow_avg.squeeze(), fdone

def ClusterRun(function, parameter_list, max_cores=1000):

    """
    function: The routine run in parallel, which must contain all necessary
       imports internally.

       parameter_list: should be an iterable of elements, for which each element
       will be passed as the parameter to function for each parallel execution.

       max_cores: Standard Rhino cluster etiquette is to stay within 100 cores
       at a time.  Please ask for permission before using more.

       In jupyterlab, the number of engines reported as initially running may
       be smaller than the number actually running.  Check usage from an ssh
       terminal using:  qstat -f | egrep "$USER|node" | less

       Undesired running jobs can be killed by reading the JOBID at the left
       of that qstat command, then doing:  qdel JOBID
    """

    import cluster_helper.cluster
    from pathlib import Path

    num_cores = len(parameter_list)
    num_cores = min(num_cores, max_cores)

    myhomedir = str(Path.home())
    # can add in 'mem':Num where Num is # of GB to allow for memory into extra_params
    #...Nora said it doesn't work tho and no sign it does
    # can also try increasing cores_per_job to >1, but should also reduce num_jobs to not hog
    # so like 2 and 50 instead of 1 and 100 etc. Went up to 5/20 for encoding at points
    # ...actually now went up to 10/10 which seems to stop memory errors 2020-08-12
    with cluster_helper.cluster.cluster_view(scheduler="sge", queue="RAM.q", \
        num_jobs=5, cores_per_job=10, \
        extra_params={'resources':'pename=python-round-robin'}, \
        profile=myhomedir + '/.ipython/') \
        as view:
        # 'map' applies a function to each value within an interable
        res = view.map(function, parameter_list)
        
    return res
# 20 cores_per_job is enough for catFR1 SWRclustering. 7 missed 5-10
# AMY encoding and surrounding_recall no issues with 10 cores/job
# 10 works for ENTPHC with encoding
# 10 works for all surrounding_recall regardless of region (with HFA too)
# 30 works for most of FR1 encoding...40 works for all
# 25 didn't work for a few...made a list of the 15 or so in SWRanalysis 2022-03-09
# 25 didn't work for a few catFR1 too. Made list of 20 and will try running with 50 cores/job

def z_score_epochs(power):
    # power should be a 3d array of shape num_trials x num_channels x num_timesteps
    return (power - np.mean(power, axis=(0,2), keepdims=True)) / np.std(np.mean(power, axis=2, keepdims=True),axis=0, keepdims=True) 

def compute_morlet(eeg, freqs, sr, desired_sr, n_jobs, tmin, tmax, mode='power', split_power_idx=None):
    """
    :param mne array eeg: eeg data
    :param ndarray freqs: frequencies to compute morlet wavelets over
    :param int sr: sampling rate of eeg data
    :param int desired_sr: desired sr of power/phase data
    :param int n_jobs: how many threads to run on
    :param float tmin, tmax: where to select signal from (to remove buffer), in ms
    :param str mode: power or phase
    :param int/None split_power_idx: compute two freq bands, return both and their union
    if int this is where freqs is divided
    """
    
    from mne.time_frequency import tfr_morlet
    from mne.filter import resample
    morlet_output = tfr_morlet(eeg, freqs, n_cycles=5, return_itc=False, average=False, n_jobs=n_jobs, output=mode)
    morlet_output.crop(tmin=tmin/1000, tmax=tmax/1000, include_tmax=False) # crop out the buffer
    
    if mode == 'power':
        # log transform, mean across wavelet frequencies, and z-score
        #morlet_output.data = np.log10(morlet_output.data)
        
        # split frequency bands into two groups 
        if split_power_idx is not None:
            morlet_output_data_1 = np.mean(morlet_output.data[:, :, :split_power_idx], axis=2)
            morlet_output_data_2 = np.mean(morlet_output.data[:, :, split_power_idx:], axis=2)
            
        morlet_output.data = np.mean(morlet_output.data, axis=2)
        
    if mode == 'phase':
        # take circular mean across wavelet phase values
        morlet_output.data = scipy.stats.circmean(morlet_output.data, high=np.pi, low=-np.pi, axis=2)
        
    if sr > desired_sr: 
        morlet_output.data = resample(morlet_output.data, down=sr/desired_sr)
        if split_power_idx is not None:
            morlet_output_data_1 = resample(morlet_output_data_1, down=sr/desired_sr)
            morlet_output_data_2 = resample(morlet_output_data_2, down=sr/desired_sr)
    
    if split_power_idx is not None:
        return morlet_output.data, morlet_output_data_1, morlet_output_data_2
    else:
        return morlet_output.data
    
    
    
    
    