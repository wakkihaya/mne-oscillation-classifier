MNE-somato-data-bids
====================

This dataset contains the MNE-somato-data in BIDS format.

The conversion can be reproduced through the Python script stored in the
`/code` directory of this dataset. See the README in that directory.

The `/derivatives` directory contains the outputs of running the FreeSurfer
pipeline `recon-all` on the MRI data with no additional commandline options
(only defaults were used):

$ recon-all -i sub-01_T1w.nii.gz -s 01 -all

After the `recon-all` call, there were further FreeSurfer calls from the MNE
API:

$ mne make_scalp_surfaces -s 01 --force
$ mne watershed_bem -s 01

The derivatives also contain the forward model `*-fwd.fif`, which was produced
using the source space definition, a `*-trans.fif` file, and the boundary
element model (=conductor model) that lives in
`freesurfer/subjects/01/bem/*-bem-sol.fif`.

The `*-trans.fif` file is not saved, but can be recovered from the anatomical
landmarks in the `sub-01/anat/T1w.json` file and MNE-BIDS' function
`get_head_mri_transform`.

See: https://github.com/mne-tools/mne-bids for more information.

Notes on FreeSurfer
===================
the FreeSurfer pipeline `recon-all` was run new for the sake of converting the
somato data to BIDS format. This needed to be done to change the "somato"
subject name to the BIDS subject label "01". Note, that this is NOT "sub-01",
because in BIDS, the "sub-" is just a prefix, whereas the "01" is the subject
label.


