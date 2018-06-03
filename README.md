# Interestingness_ICMR

Code for the work: "Deep Pairwise Classification & Ranking for Predicting Media Interestingness" 
by Jayneel Parekh, Harshvardhan Tibrewal, Sanjeel Parekh
("Mention arXiv link")

### Requirements
- numpy 
- tensorflow >= 1.3.0
- keras >= 2.1.3

### Data
Contains code for training/testing our system on the MediaEval Dataset
(http://www.multimediaeval.org/mediaeval2017/mediainterestingness/index.html).

> The MediaEval Dataset needs to be acquired by the user himself/herself.
> We will check if we can upload the feature respresentations used by us.

The Network Weights file (for testing system we trained) exceeds the max file size of git, thus not uploaded here.
Will be uploaded soon, but not on git. Link will be added here


### Usage
Command to train image interestingness system: 
```
python train_nn.py [data_type] [operation] [ranker]

Functionality currently available to user is 

(a) data_type : Selects whether to run experiments on images or videos. (Options : [image , video] )

(b) operation : Training or Testing or making predictions for any set of images/video-shots if
    provided with appropriate feature representation files for images/video (respresented as feature vector of size 4096)
    (Options : [train ,  test] )
    
(c) ranker: selection of ranking algorithm for prediction (Options: [mih_to, mih_ro, sp, pp] )

Example : python train_nn.py image train mih_to
```

Currently the model is fixed and uses fc7 features (AlexNet) for images and c3d features for videos. 

Next possible update is to allow more flexibility in feature respresentation of the input.

