# Interestingness_ICMR

Code for the work: "Deep Pairwise Classification & Ranking for Predicting Media Interestingness" 
by Jayneel Parekh, Harshvardhan Tibrewal, Sanjeel Parekh

Contains code for training/testing our system on the MediaEval Dataset.

The MediaEval Dataset needs to be acquired by the user himself/herself
We will check if we can upload the feature respresentations used by us.

The Network Weights file (for testing system we trained) exceeds the max file size of git, thus not uploaded here.
Will be uploaded soon, but not on git. Link will be added here

Functionality currently available to user is 
(a) Selecting whether to run experiments for image or video
(b) Training or Testing or making predictions for any set of images/video-shots if
    provided with appropriate feature representation files for images/video (respresented as feature vector of size 4096)
(c) Selection of ranking algorithm for prediction (4 options: 'mih_to', 'mih_ro', 'sp', 'pp')
Command to train image interestingness system: python train_nn.py image train mih_to

Currently the model is fixed and thus so is the input size.

Next possible update is to allow more flexibility in feature respresentation of the input.

