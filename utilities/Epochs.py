#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Several function that help to get data structure in eppchs
@author: vpeterson, Sep 2021
"""
import numpy as np
import mne


def get_epochs_pad(data_file, events, twindow=90.0):
    """
    Extract epochs from file.

    This function segments data tmin sec before target onset and tmax sec
    after target onset

    Parameters
    ----------
    data : array, shape(n_channels, n_samples)
        either cortex of subcortex data to be epoched.
    events : array, shape(n_events,2)
        All events that were found by the function
        'get_events'.
        The first column contains the event time seconds and the second column
        contains the event id.
    tmin : float
        Start time before event (in sec).
        If nothing is provided, defaults to 1.
    tmax : float
        Stop time after  event (in sec).
        If nothing is provided, defaults to 1.

    Returns
    -------
    X : array, shape(n_events, n_channels, n_samples)
        epoched data
    Y : array, shape(n_events, n_samples)
        label information, no_sz=0, sz_on=1, sz=2.
        See 'get_events' for more information regarding labels names.
    T : array, shape(n_events, n_samples)
        time of the sz_on.
    """
    raw = mne.io.read_raw_edf(data_file)
    sf = raw.info['sfreq']
    data = raw.get_data()
    n_channels, time_rec = np.shape(data)
    # labels
    idx_eof = np.where(events[:, 1] == 0)[0]
    eof_event_time = events[idx_eof, 0]  # in seconds
    # add zero for first eof
    eof_event_time = np.hstack((0.0, eof_event_time))
    # # check recording lenght is at least 90s
    rec_len = [t - s for s, t in zip(eof_event_time, eof_event_time[1:])]
    idx_shorttrials = np.where(np.asarray(rec_len) < 60)[0]
    # get final idx_eof and eof event time
    #Do not delete short trials if they contain a seizure
    # keep_inds = []
    # for i in range(len(idx_shorttrials)):
    #     if events[idx_eof[idx_shorttrials[i]] - 1, 1] != 0:
    #         keep_inds.append(i)
    # idx_shorttrials = np.delete(idx_shorttrials,keep_inds)
    # delete remaining short trials:
    idx_eof_new = np.delete(idx_eof, idx_shorttrials)
    eof_event_time_new = events[idx_eof_new, 0]
    rec_len_new = np.delete(rec_len, idx_shorttrials)

    # check if annot time is not longer than file time
    idx_stop = np.where(np.dot(eof_event_time_new, sf) > time_rec)

    # avoid trying to read those trials
    idx_eof_ = np.delete(idx_eof_new, idx_stop)
    nTrials = len(idx_eof_)

    Y = np.zeros((nTrials,)).astype(int)  # labels
    Z = np.zeros((nTrials,))  # time
    n_samples = int(round((twindow)*sf))
    X = np.zeros((nTrials, n_channels, n_samples))
    
    # extract epochs starts
    for i in range(nTrials):
        len_archive = rec_len_new[i]
        time_sof = eof_event_time_new[i] - len_archive
        start_epoch = int(round(time_sof * sf))
        if len_archive >= twindow:
            stop_epoch = int(round((time_sof + twindow) * sf))
            final_epoch = stop_epoch
            epoch = data[:, start_epoch:stop_epoch]
        else:
            stop_epoch = int(round((time_sof + len_archive) * sf))
            final_epoch = int(round((time_sof + twindow) * sf))
            epoch = data[:, start_epoch:stop_epoch]
            # pad the time that is not part of the shortened epoch to reach twindow
            time_to_pad = final_epoch - stop_epoch # in samples
            # using mirrow to padding
            pad_length = int(round(1 * sf))
            pad_data = data[:,(stop_epoch - pad_length):stop_epoch]
            num_pad_reps = int(np.floor(time_to_pad/pad_length) + 1)
            total_pad_data = np.tile(pad_data,num_pad_reps)
            epoch = np.concatenate((epoch,total_pad_data),1)
            # now extract only the data fit to the frame
            epoch = epoch[:, :n_samples]
        # check lenght epoch
        if (final_epoch - start_epoch) != n_samples:
            raise ValueError('epoch lengths does not represent time window')
        
        X[i] = epoch

        if idx_eof_[i] != len(events)-1:
            if events[idx_eof_[i] - 1, 1] != 0 and (idx_eof_[i] - 1) != -1:  # then is sz_on
                # label
                Y[i] = events[idx_eof_[i] - 1, 1]
                # time
                time_sz = events[idx_eof_[i] - 1, 0]
                time = time_sz - time_sof
                time_eof = events[idx_eof_[i], 0]
                if len_archive > twindow:  # large archive
                    if time > twindow:  # sz happens and epoch didn't capture it
                        n_90 = int(len_archive / twindow)  # n of 90 s in axive
                        t_90 = time / twindow  # in which n time happens
                        if n_90 < t_90:  # sz is happening closer the end
                            # redefine epoch from eof to - 90
                            start_epoch = int(round((time_eof - twindow) * sf))
                            stop_epoch = int(round(time_eof * sf))
                            epoch = data[:, start_epoch:stop_epoch]
                            X[i] = epoch

                            # time
                            time = twindow - (time_eof - time_sz)
                        else:
                            # make sure we pick up the correct 90 s
                            start_epoch = int(round((time_sof + int(t_90)*twindow) * sf))
                            stop_epoch = int(round((time_sof + int(t_90)*twindow + twindow) * sf))
                            epoch = data[:, start_epoch:stop_epoch]
                            X[i] = epoch

                            # time
                            time = time_sz - time_sof - int(t_90)*twindow 
                if np.sign(time) == -1:
                    raise ValueError('Time cannot be negative')
                Z[i] = time
    return X, Y, Z

def get_epochs_nopad_all(data_file, events, twindow=90.0):
    """
    Extract epochs from file and use zeropadding

    This function segments data tmin sec before target onset and tmax sec
    after target onset

    Parameters
    ----------
    data : array, shape(n_channels, n_samples)
        either cortex of subcortex data to be epoched.
    events : array, shape(n_events,2)
        All events that were found by the function
        'get_events'.
        The first column contains the event time seconds and the second column
        contains the event id.
    tmin : float
        Start time before event (in sec).
        If nothing is provided, defaults to 1.
    tmax : float
        Stop time after  event (in sec).
        If nothing is provided, defaults to 1.

    Returns
    -------
    X : array, shape(n_events, n_channels, n_samples)
        epoched data
    Y : array, shape(n_events, n_samples)
        label information, no_sz=0, sz_on=1, sz=2.
        See 'get_events' for more information regarding labels names.
    T : array, shape(n_events, n_samples)
        time of the sz_on.
    """
    raw = mne.io.read_raw_edf(data_file)
    sf = raw.info['sfreq']
    data = raw.get_data()
    n_channels, time_rec = np.shape(data)
    # labels
    idx_eof = np.where(events[:, 1] == 0)[0]
    eof_event_time = events[idx_eof, 0]  # in seconds
    # add zero for first eof
    eof_event_time = np.hstack((0.0, eof_event_time))
    # # check recording lenght is at least 90s
    rec_len = [t - s for s, t in zip(eof_event_time, eof_event_time[1:])]
    idx_shorttrials = np.where(np.asarray(rec_len) < 60)[0]
    # # get final idx_eof and eof event time
    # #Do not delete short trials if they contain a seizure
    # keep_inds = []
    # for i in range(len(idx_shorttrials)):
    #     if events[idx_eof[idx_shorttrials[i]] - 1, 1] != 0:
    #         keep_inds.append(i)
    # idx_shorttrials = np.delete(idx_shorttrials,keep_inds)
    # delete remaining short trials:
    idx_eof_new = np.delete(idx_eof, idx_shorttrials)
    eof_event_time_new = events[idx_eof_new, 0]
    rec_len_new = np.delete(rec_len, idx_shorttrials)

    # check if annot time is not longer than file time
    idx_stop = np.where(np.dot(eof_event_time_new, sf) > time_rec)

    # avoid trying to read those trials
    idx_eof_ = np.delete(idx_eof_new, idx_stop)
    nTrials = len(idx_eof_)

    Y = np.zeros((nTrials,)).astype(int)  # labels
    Z = np.zeros((nTrials,))  # time
    n_samples = int(round((twindow)*sf))
    X = []
    # extract epochs starts
    for i in range(nTrials):
        len_archive = rec_len_new[i]
        time_sof = eof_event_time_new[i] - len_archive
        start_epoch = int(round(time_sof * sf))
        if len_archive >= twindow:
            stop_epoch = int(round((time_sof + twindow) * sf))
            epoch = data[:, start_epoch:stop_epoch]
        else:
            stop_epoch = int(round((time_sof + len_archive) * sf))
            epoch = data[:, start_epoch:stop_epoch]
            # pad the time that is not part of the shortened epoch to reach twindow      
    
        if idx_eof_[i] != len(events)-1:
            if events[idx_eof_[i] - 1, 1] != 0 and (idx_eof_[i] - 1) != -1:  # then is sz_on
                # label
                Y[i] = events[idx_eof_[i] - 1, 1]
                # time
                time_sz = events[idx_eof_[i] - 1, 0]
                time = time_sz - time_sof
                time_eof = events[idx_eof_[i], 0]
                if len_archive > twindow:  # large archive
                    if time > twindow:  # sz happens and epoch didn't capture it
                        n_90 = int(len_archive / twindow)  # n of 90 s in axive
                        t_90 = time / twindow  # in which n time happens
                        if n_90 < t_90:  # sz is happening closer the end
                            # redefine epoch from eof to - 90
                            start_epoch = int(round((time_eof - twindow) * sf))
                            stop_epoch = int(round(time_eof * sf))
                            epoch = data[:, start_epoch:stop_epoch]
                            # time
                            time = twindow - (time_eof - time_sz)
                        else:
                            # make sure we pick up the correct 90 s
                            start_epoch = int(round((time_sof + int(t_90)*twindow) * sf))
                            stop_epoch = int(round((time_sof + int(t_90)*twindow + twindow) * sf))
                            epoch = data[:, start_epoch:stop_epoch]
                            # time
                            time = time_sz - time_sof - int(t_90)*twindow 
                if np.sign(time) == -1:
                    raise ValueError('Time cannot be negative')
                Z[i] = time
        X.append(epoch)

    return X, Y, Z

def get_epochs_zeropad_all(data_file, events, twindow=90.0):
    """
    Extract epochs from file and use zeropadding

    This function segments data tmin sec before target onset and tmax sec
    after target onset

    Parameters
    ----------
    data : array, shape(n_channels, n_samples)
        either cortex of subcortex data to be epoched.
    events : array, shape(n_events,2)
        All events that were found by the function
        'get_events'.
        The first column contains the event time seconds and the second column
        contains the event id.
    tmin : float
        Start time before event (in sec).
        If nothing is provided, defaults to 1.
    tmax : float
        Stop time after  event (in sec).
        If nothing is provided, defaults to 1.

    Returns
    -------
    X : array, shape(n_events, n_channels, n_samples)
        epoched data
    Y : array, shape(n_events, n_samples)
        label information, no_sz=0, sz_on=1, sz=2.
        See 'get_events' for more information regarding labels names.
    T : array, shape(n_events, n_samples)
        time of the sz_on.
    """
    raw = mne.io.read_raw_edf(data_file)
    sf = raw.info['sfreq']
    data = raw.get_data()
    n_channels, time_rec = np.shape(data)
    # labels
    idx_eof = np.where(events[:, 1] == 0)[0]
    eof_event_time = events[idx_eof, 0]  # in seconds
    # add zero for first eof
    eof_event_time = np.hstack((0.0, eof_event_time))
    # # check recording lenght is at least 90s
    rec_len = [t - s for s, t in zip(eof_event_time, eof_event_time[1:])]
    idx_shorttrials = np.where(np.asarray(rec_len) < 60)[0]
   
    # delete short trials:
    idx_eof_new = np.delete(idx_eof, idx_shorttrials)
    eof_event_time_new = events[idx_eof_new, 0]
    rec_len_new = np.delete(rec_len, idx_shorttrials)

    # check if annot time is not longer than file time
    idx_stop = np.where(np.dot(eof_event_time_new, sf) > time_rec)

    # avoid trying to read those trials
    idx_eof_ = np.delete(idx_eof_new, idx_stop)
    nTrials = len(idx_eof_)

    Y = np.zeros((nTrials,)).astype(int)  # labels
    Z = np.zeros((nTrials,))  # time
    n_samples = int(round((twindow)*sf))
    X = np.zeros((nTrials, n_channels, n_samples))
    
    # extract epochs starts
    for i in range(nTrials):
        len_archive = rec_len_new[i]
        time_sof = eof_event_time_new[i] - len_archive
        start_epoch = int(round(time_sof * sf))
        if len_archive >= twindow:
            stop_epoch = int(round((time_sof + twindow) * sf))
            final_epoch = stop_epoch
            epoch = data[:, start_epoch:stop_epoch]
        else:
            stop_epoch = int(round((time_sof + len_archive) * sf))
            final_epoch = int(round((time_sof + twindow) * sf))
            epoch = data[:, start_epoch:stop_epoch]
            # pad the time that is not part of the shortened epoch to reach twindow
            time_to_pad = final_epoch - stop_epoch # in samples
        
            
            if time_to_pad > 0:
             
                # using zeros
                pad_data = np.zeros((4, time_to_pad))
                epoch = np.concatenate((epoch,pad_data),1)
                # now extract only the data fit to the frame
                epoch = epoch[:, :n_samples]
            # using 0.5s to pad.
            pad_length = int(round(0.1 * sf))
            pad_data = data[:,(stop_epoch - pad_length):stop_epoch]
            num_pad_reps = int(np.floor(time_to_pad/pad_length) + 1)
            total_pad_data = np.tile(pad_data,num_pad_reps)
            epoch = np.concatenate((epoch,total_pad_data),1)
            # now extract only the data fit to the frame
            epoch = epoch[:, :n_samples]
        # check lenght epoch
        if (final_epoch - start_epoch) != n_samples:
            raise ValueError('epoch lengths does not represent time window')
        
        X[i] = epoch

        if idx_eof_[i] != len(events)-1:
            if events[idx_eof_[i] - 1, 1] != 0 and (idx_eof_[i] - 1) != -1:  # then is sz_on
                # label
                Y[i] = events[idx_eof_[i] - 1, 1]
                # time
                time_sz = events[idx_eof_[i] - 1, 0]
                time = time_sz - time_sof
                time_eof = events[idx_eof_[i], 0]
                if len_archive > twindow:  # large archive
                    if time > twindow:  # sz happens and epoch didn't capture it
                        n_90 = int(len_archive / twindow)  # n of 90 s in axive
                        t_90 = time / twindow  # in which n time happens
                        if n_90 < t_90:  # sz is happening closer the end
                            # redefine epoch from eof to - 90
                            start_epoch = int(round((time_eof - twindow) * sf))
                            stop_epoch = int(round(time_eof * sf))
                            epoch = data[:, start_epoch:stop_epoch]
                            X[i] = epoch

                            # time
                            time = twindow - (time_eof - time_sz)
                        else:
                            # make sure we pick up the correct 90 s
                            start_epoch = int(round((time_sof + int(t_90)*twindow) * sf))
                            stop_epoch = int(round((time_sof + int(t_90)*twindow + twindow) * sf))
                            epoch = data[:, start_epoch:stop_epoch]
                            X[i] = epoch

                            # time
                            time = time_sz - time_sof - int(t_90)*twindow 
                if np.sign(time) == -1:
                    raise ValueError('Time cannot be negative')
                Z[i] = time
    return X, Y, Z
def get_events(annot_file, Verbose=False):
    """
    Clean annotation file and generate event array.

    Parameters
    ----------
    annot_files : str
        annot_file.

    Returns
    -------
    events : array
        DESCRIPTION.

    """

    annot_ = np.loadtxt(annot_file, delimiter=',', skiprows=1,
                        dtype=str)
    # we define here the valid labels
    idx = [idx for idx, s in enumerate(annot_[:, 1]) if "eof" not in s]
    idx2 = [idx for idx, s in enumerate(annot_[idx, 1]) if "sz_" not in s]

    idx_2delete = []
    for i in range(len(idx2)):
        idx_2delete.append(idx[idx2[i]])

    idx_2delete = np.asarray(idx_2delete).astype(np.int)
    if Verbose:
        print(str(len(idx_2delete))+' events deleted')

    events = np.delete(annot_, idx_2delete, 0)
    # change and arrange categorical to discrete labels
    # eof == 0, sz_on_*==1, sz*==2
    idx_eof = [idx for idx, s in enumerate(events[:, 1]) if "eof" in s]
    idx_sz_ = [idx for idx, s in enumerate(events[:, 1]) if "sz" in s]
    idx_sz_on = [idx for idx, s in enumerate(events[:, 1]) if "sz_on" in s]
    # the sz_ get everything with sz. the class "sz with onset" will be get
    # as a difference between the idx_sz and idx_sz_on
    idx_sz = set(idx_sz_) - set(idx_sz_on)
    idx_sz = list(idx_sz)

    events[idx_eof, 1] = 0
    events[idx_sz_on, 1] = 1
    events[idx_sz, 1] = 2

    events = events.astype(np.float)
    return events

