from moviepy.editor import *
from FindMattes import createMatteImg
from torchmotion import *
from PIL import Image
from numpy import array

import sys
import argparse

from segmentation import flood_video, sobel_video



DEBUG = True
block_size=12

def is_delta_block(frameA, frameB, blockx, blocky, pixel_threshold, color_threshold):
    pixel_diffs=0
    for x in range(block_size):
        for y in range(block_size):
            if is_delta_pixel(frameA, frameB, x+blockx*block_size, y+blocky*block_size, color_threshold):
                pixel_diffs+=1
    return pixel_diffs>pixel_threshold
    
def is_delta_pixel(frameA, frameB, pixelx, pixely,color_threshold):
    pixa= frameA[pixely][pixelx]
    pixb= frameB[pixely][pixelx]
    return math.abs(pixa[0]-pixb[0])+math.abs(pixa[1]-pixb[1])+math.abs(pixa[2]-pixb[2])>color_threshold
    
class Frames:
    def __init__(self, imgs, fps):
       self.imgs=imgs
       self.fps=fps
       
    def make_frame(self, t):
        index=int(self.fps*t)%len(self.imgs)
        img=self.imgs[index]
        data=img.tobytes('raw')
        resultStruct=[]
        index=0
        for i in range(img.height):
            row=[]
            for j in range(img.width):
                pixel=[data[index],data[index+1],data[index+2]]
                index+=3
                row.append(pixel)
            resultStruct.append(row)
        return array(resultStruct)
        
def ApplyPanStabilization(filename, fps):
    sourcevideo=VideoFileClip(filename)
    maker=StabilizePanMaker(sourcevideo, 120/fps, fps)
    outname=".".join(filename.split('.')[:-1])
    result=VideoClip(maker.make_frame,duration=sourcevideo.duration)
    result.write_videofile(f'stabilized-{outname}.mp4',fps=fps,codec="libx264", bitrate="3000k")
    
def ApplyResnet(filename):
    
    video=VideoFileClip(filename)
    #clips=[]
    mattes=[]
    frameindex=0
    effectFactors={
    16:1.0, #moe is being found to be a potted plant too often, when jumping, actually just keep the potted plant
    15:2.0,  #and we could do with a bit more person detection
    }
    for frame in video.iter_frames():
        print(f'start frame {frameindex}')
        frameindex+=1
        intermediate=Image.frombytes('RGB',(480,360), frame)
        print('  image made from frame')
        
        matte=createMatteImg(intermediate,480,amplifyFactor=1000, startOver=not mattes)
        if DEBUG:
            matte.save(f'tmp/tmpmask{frameindex}.png')
        print('  matte made from img')
        mattes.append(matte)
    frames=Frames(mattes,video.fps)
    result=VideoClip(frames.make_frame,duration=video.duration)
    result.write_videofile('output.mp4',fps=video.fps,codec='mpeg4')

def ApplyMotion(filename):
    sourcevideo=VideoFileClip(filename)
    motion=MotionDifferenceFrameMaker(sourcevideo, frameReach=3, frameDecay=0.8)
    diffClip=VideoClip(motion.make_frame,duration=sourcevideo.duration)
    maskMaker=LocalityEnhancerMaskMaker(diffClip,pixelReach=18,pixelDecay=0.9)
    
    maskClip=VideoClip(maskMaker.make_frame, ismask=True, duration=sourcevideo.duration)
    maskClip.write_videofile("mask.mp4", fps=sourcevideo.fps, codec="mpeg4")
    
    #timeMask=TemporalMaskEnhancer(maskClip,maxDelta=0.2, samples=4)
    #timeClip=VideoClip(timeMask.make_frame, ismask=True, duration=sourcevideo.duration)
    #timeClip.write_videofile("mask2.mp4",fps=sourcevideo.fps, codec="mpeg4")
    
    masked=sourcevideo.set_mask(maskClip.to_mask())
    composite=CompositeVideoClip([ColorClip((640,480),(0,0,0),duration=sourcevideo.duration),masked])
    composite.write_videofile('fully masked.mp4', fps=sourcevideo.fps, codec="mpeg4" )

def ApplyDepth(filename):
    sourcevideo=VideoFileClip(filename)
    depthClip=VideoClip(MidasDepthFramer(sourcevideo).make_frame, duration=sourcevideo.duration)
    outname=".".join(filename.split('.')[:-1])
    depthClip.write_videofile(f'depth-{outname}.mp4',fps=sourcevideo.fps, codec="mpeg4")
def ApplyRaft(filename):
    
    sourcevideo=VideoFileClip(filename)
    maker=RaftMaskMaker(sourcevideo,target_fps=8, backwards=True)
    raftClip=VideoClip(maker.make_frame, duration=sourcevideo.duration)
    raftMask=VideoClip(maker.make_mask, duration=sourcevideo.duration,ismask=True)
    #sourcevideo.write_gif('source-giffed.gif',fps=8)
    #raftClip.write_videofile('raft.mp4',fps=8, codec="mpeg4")
    outname=".".join(filename.split('.')[:-1])
    raftMask.write_videofile(f'raftmask-{outname}.mp4',fps=8,codec="libx264",bitrate="3000k")
    
def ApplyCombined(filename):
    sourcevideo=VideoFileClip(filename)
    #sourcevideo.duration=0.75
    #sourcevideo.duration=sourcevideo.duration/4
    combinedClip=VideoClip(CombinedDepthAnalysis(sourcevideo, fps=8).make_mask,duration=sourcevideo.duration,ismask=True)
    masked=sourcevideo.set_mask(combinedClip)
    composite=CompositeVideoClip([ColorClip((640,480),(0,0,0),duration=sourcevideo.duration),masked])
    outname=".".join(filename.split('.')[:-1])
    composite.write_videofile(f'extracted-{outname}.mp4',fps=sourcevideo.fps,codec="mpeg4")
    #combinedClip.write_videofile('combined.mp4',fps=8,codec='mpeg4', )
    
def ApplySegmentation(filename):
    sourcevideo=VideoFileClip(filename)
    sourcevideo.fps=8
    data_result, borders, segments,neighbors=flood_video(sourcevideo,flexibility=5, target_segments=350)
    make_frame=lambda T: data_result[:,:,int(T*8),:].detach().cpu().numpy()
    output=VideoClip(make_frame, duration=sourcevideo.duration)
    output.write_videofile('segmented-'+filename,fps=8,codec="mpeg4")
def ApplySobel(filename):
    sourcevideo=VideoFileClip(filename)
    sourcevideo.fps=8
    data_result=sobel_video(sourcevideo)
    make_frame=lambda T: data_result[:,:,int(T*8),:].detach().cpu().numpy()
    output=VideoClip(make_frame, duration=sourcevideo.duration)
    output.write_videofile('sobel3d-'+filename,fps=8,codec="mpeg4")
    
    
def ApplyMotionSegmentation(filename, create_segment_video, target_fps=8, stabilize=False):
    with torch.no_grad():
        sourcevideo=VideoFileClip(filename)
        segmented=sourcevideo
        stabilizer=None
        if stabilize:
            print('stabilizing video')
            stabilizer=StabilizePanMaker(sourcevideo, 60, target_fps)
            segmented=VideoClip(stabilizer.make_frame, duration=sourcevideo.duration)
            print('video stabilized')
        #sourcevideo.duration=12*(1/8)
        #frame rate shouldn't matter for number of segments, rate of motion might increase it, but I'm not sampling that.  Screw you.
        maker=SegmentedFlowMaker(sourcevideo,activityThreshold=0.5,target_segments=550*(int(8*sourcevideo.duration)**0.5), target_fps=target_fps, stabilizer=stabilizer)
        mask=VideoClip(maker.make_mask,duration=sourcevideo.duration,ismask=True)
        just_raft=VideoClip(maker.make_just_raft,duration=sourcevideo.duration,ismask=False)
        colors=VideoClip(maker.make_segment_rgb,duration=sourcevideo.duration)
        #resnet=VideoClip(maker.make_resnet,duration=sourcevideo.duration)
        outname=".".join(filename.split('.')[:-1])
        if create_segment_video:
            colors.write_videofile(f'motion-segment-segments-{outname}.mp4',fps=target_fps,codec="mpeg4")
            mask.write_videofile(f'motion-segment-mask-{outname}.mp4',fps=target_fps,codec="mpeg4")
            just_raft.write_videofile(f'motion-segment-raft-{outname}.mp4',fps=target_fps,codec="mpeg4")
            #resnet.write_videofile(f'motion-segment-resnet-{outname}.mp4',fps=8,codec="mpeg4")
        masked=sourcevideo.set_mask(mask)
        composite=CompositeVideoClip([ColorClip(sourcevideo.size,(0,0,0),duration=sourcevideo.duration), masked])
        composite.write_videofile(f'motion_segment-extracted-{outname}.mp4',fps=target_fps,codec="libx264", bitrate="3000k")
    
    
    
def main():
    parser=argparse.ArgumentParser(description="Make a time-inclusive segmented version of a RAFT flow")
    parser.add_argument("filename", type=str, help="the file to segment")
    parser.add_argument("--fps", type=int, default=8, help="analyze at this frame rate")
    parser.add_argument("--sidefiles", action="store_false", help="Include secondary files like mask and segment map")
    parser.add_argument("--nomotionsegmentation", action="store_false", help="Do the main analysis, set to False if you only want to look at other artifacts")
    parser.add_argument("--depth", action="store_true", help="Produce a depth map, may be used for further filtering later")
    parser.add_argument("--stabilize", action="store_true", help="Stabilize the video")
    args=parser.parse_args()
    #if sys.argv and len(sys.argv)==2:
        #ApplyRaft(sys.argv[1])
        #ApplySobel(sys.argv[1])
    if args.stabilize:
        ApplyPanStabilization(args.filename,args.fps)
    if args.depth:
        ApplyDepth(args.filename)
    if args.nomotionsegmentation:
        ApplyMotionSegmentation(args.filename,args.sidefiles, args.fps, args.stabilize)
    
    else:
        print ("please enter filename")
    
    
if __name__=='__main__':
    main()