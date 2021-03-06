# SimpsonsExtractor
Extracting characters from the simpsons with autorotoscoping

## Installation and setup
### Requirements
Graphics card that supports CUDA 10.2 or higher.  The process uses a lot of GPU ram.  I can only verify that it works reliably with 6 Gigs of GRAM

I highly highly highly recomend you install anaconda.  Trying to install pytorch on windows with pip is awful

Not tested on Linux

### STEPS

1.  Install python 3.7 or higher
1.  Install [pytorch](https://pytorch.org/get-started/locally/)  Cuda is mandatory.  If you're computer can't handle cuda, this script will take days to run, AND you'll have to change the source code to tell it to run on the CPU.
1.  run pip install -r requirements.txt
1.  Check out [RAFT](https://github.com/princeton-vl/RAFT) somewhere on your computer  
1. 1. Open a command prompt and browse to the "core" subfolder of RAFT
   1. run "conda develop ."
   1. Per the instructions on RAFT's readme, download the pretrained models
   1. Whever you extracted raft-sintel.pth to, change the initModel(self) method of the RaftMaskMaker class in torchmotion.py.  (currently line 181)
1.  Go to wherever you checked this project out, and run "conda develop ."

## Usage
python -m extractchar [filename]
Produces 3 outputs,
1.  Motion-segment-segments-[name].mp4,  which has a graphical representation of the segmentation of the video and which ones are used in creating the motion mask
2.  Motion-segment-mask-[name].mp4, the mask that is used to cut out the output.  NOTE that this includes BOTH the segments AND the output of RAFT.  Useful if you want to clean it up.
3.  Motion-segment-extracted-[name].mp4.  The input file cut to the shape of the mask.
