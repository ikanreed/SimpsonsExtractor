# SimpsonsExtractor
Extracting characters from the simpsons with autorotoscoping

## Installation and setup

### requirements
Graphics card that supports CUDA 10.2 or higher.  The process uses a lot of GPU ram.  I can only verify that it works reliably with 6 Gigs of GRAM

I highly highly highly recomend you install conda.  Trying to install pytorch on windows with pip is awful

--TODO--

## Usage
python -m extractchar [filename]
Produces 3 outputs,
1.  Motion-segment-segments-[name].mp4,  which has a graphical representation of the segmentation of the video and which ones are used in creating the motion mask
2.  Motion-segment-mask-[name].mp4, the mask that is used to cut out the output.  NOTE that this includes BOTH the segments AND the output of RAFT.  Useful if you want to clean it up.
3.  Motion-segment-extracted-[name].mp4.  The input file cut to the shape of the mask.
