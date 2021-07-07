# -*- coding: utf-8 -*-
"""
Description: By using SVM, classify alpha or beta wave.
Reference: https://mne.tools/dev/auto_examples/time_frequency/time_frequency_global_field_power.html
"""

import os.path as op
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import somato
from mne.baseline import rescale
from mne.stats import bootstrap_confidence_interval


from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
  iter_freqs = [
     ('Theta', 4, 7),
     ('Alpha', 8, 12),
     ('Beta', 13, 25),
     ('Gamma', 30, 45)
  ]
  event_id, tmin, tmax = 1, -1., 3.
  baseline = None

  raw_fname = Path("../../mne_data/MNE-somato-data/sub-01/meg/sub-01_task-somato_meg.fif")
  raw = mne.io.read_raw_fif(raw_fname)
  events = mne.find_events(raw,stim_channel='STI 014')

  raw.pick_types(meg='grad', eeg=True, eog=False)
  raw.load_data()

  # bandpass filter for alpha
  alpha_freq = iter_freqs[1]
  raw.filter(alpha_freq[1], alpha_freq[2], n_jobs=1,
           l_trans_bandwidth=1,
           h_trans_bandwidth=1)
  # epoch with alpha data
  epochs_alpha = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=baseline,
                            reject=dict(grad=4000e-13),
                            preload=True)
  # remove evoked response
  epochs_alpha.subtract_evoked()

  del raw, events

  raw = mne.io.read_raw_fif(raw_fname)
  events = mne.find_events(raw, stim_channel='STI 014')
  raw.pick_types(meg='grad', eeg=True, eog=False)
  raw.load_data()

  # bandpass filter for beta
  beta_freq = iter_freqs[2]
  raw.filter(beta_freq[1], beta_freq[2], n_jobs=1,
             l_trans_bandwidth=1,
             h_trans_bandwidth=1)
  # epoch with beta data
  epochs_beta = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=baseline,
                            reject=dict(grad=4000e-13),
                            preload=True)
  # remove evoked response
  epochs_beta.subtract_evoked()

  alpha_data = epochs_alpha.get_data()  # [n_epochs, n_channels, n_times]
  beta_data = epochs_beta.get_data()
  class0 = np.zeros((alpha_data.shape[0],1))
  class1 = np.ones((beta_data.shape[0], 1))
  alpha_data = alpha_data.reshape(len(class0), -1)
  beta_data = beta_data.reshape(len(class1),-1)

  X = np.concatenate((alpha_data, beta_data), axis=0)
  y = np.concatenate((class0, class1), axis = 0)
  X_train, X_test, y_train, y_test = train_test_split(X,y)

  clf = svm.SVC()
  clf.fit(X_train, y_train)

  y_pred = clf.predict(X_test)
  print(y_pred)

  print(accuracy_score(y_test, y_pred))
