import torch
import torch.cuda
from moviepy.editor import VideoClip
import torchvision.transforms as T
from PIL import Image
import numpy as np
from raft import RAFT
#from FindMattes import createMatteFrame

FromFrameTransform=T.Compose([T.ToTensor()])

def torchFrame(frame):
    #result=FromFrameTransform(frame)
    result=torch.from_numpy(frame).float()
    if torch.cuda.is_available():
        result=result.to('cuda')
    return result
    

def CalculateDifferenceField(video):
    framecount=0
    result=None
    for rawframe in video.iter_frames():
        frame=torchFrame(rawframe)
        if result is None:
            result=frame*0
        print(f"outer loop, frame {framecount}")
        innerframecount=0
        for rawinnerframe in video.iter_frames():
            innerframe=torchFrame(rawinnerframe)
            result+=(frame-innerframe).abs()
            innerframecount+=1
        framecount+=1
    
    result[:,:,0]=result[:,:,0]/result[:,:,0].max()
    result[:,:,1]=result[:,:,1]/result[:,:,1].max()
    result[:,:,2]=result[:,:,2]/result[:,:,2].max()
    result=(result*255).floor().detach().cpu().numpy().astype(np.uint8)
    
    return Image.fromarray(result)
    
class MidasDepthFramer:
    def __init__(self, wrapped):
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        self.midas.to('cuda')
        self.midas.eval()
        self.wrapped=wrapped
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.default_transform
    def make_frame(self, t):
        frame=self.wrapped.get_frame(t)
        batch=self.transform(frame).to('cuda')
        prediction = self.midas(batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
            ).squeeze()
        #prediction
        result=torchFrame(frame)*0
        result[:,:,0]=prediction[:,:]*255/prediction.max()
        return result.floor().int().detach().cpu().numpy()
        
    
class MotionDifferenceFrameMaker:
    def __init__(self,wrappedClip,frameReach=4,frameDecay=1.0,motionEnhancement=2):
        self.wrapped=wrappedClip
        print('gathering raw frames')
        self.sourceFrames=[torchFrame(rawframe) for rawframe in wrappedClip.iter_frames()]
        self.differences=[None for i in range(len(self.sourceFrames))]
        for i in range(len(self.sourceFrames)):
            result=self.sourceFrames[i]*0
            midframe=self.sourceFrames[i]
            print(f'gathering differences for frame {i}')
            for frameSteps in range(1,1+frameReach):
                if i-frameSteps>=0:
                    previous=self.sourceFrames[i-frameSteps]
                    result+=((midframe-previous)**motionEnhancement*(frameDecay**frameSteps)).abs()
                if i+frameSteps<len(self.sourceFrames):
                    next=self.sourceFrames[i+frameSteps]
                    result+=((midframe-next)**motionEnhancement*(frameDecay**frameSteps)).abs()
            result[:,:,0]=result[:,:,0]/result[:,:,0].max()
            result[:,:,1]=result[:,:,1]/result[:,:,1].max()
            result[:,:,2]=result[:,:,2]/result[:,:,2].max()
            self.differences[i]=result
    def make_frame(self, t):
        index=int(self.wrapped.fps*t)%len(self.differences)
        return (self.differences[index]*255).floor().int().detach().cpu().numpy()

def rollFill(input, shifts, dim, fillValue=0):
    if shifts==0:
        return input
    result=torch.roll(input, shifts,dim)
    if dim==0:
        if shifts>0:
            result[0:shifts,:]=fillValue
        if shifts<0:
            result[shifts:,:]=fillValue
    if dim==1:
        if shifts>0:
            result[:,0:shifts]=fillValue
        if shifts<0:
            result[:,shifts:]=fillValue
    return result
        
class LocalityEnhancerMaskMaker:
    def __init__(self, wrappedMaskSource, pixelReach=4, pixelDecay=1.0):
        self.wrapped=wrappedMaskSource
        self.pixelReach=pixelReach
        self.pixelDecay=pixelDecay
    def make_frame(self, t):
        frame=torchFrame(self.wrapped.get_frame(t))
        if self.wrapped.ismask:
            masked=frame
        else:
            masked=frame.sum(2)
        masked=masked-masked.mean()
        negativeindexes=masked<0
        masked[negativeindexes]*=2.5
        result=masked*0
        for x in range(self.pixelReach):
            leftShift=rollFill(masked,x,1)
            rightShift=rollFill(masked,-x,1)
            for y in range(self.pixelReach-x):
                multiplier=1.0
                if x==0: multiplier*=0.5
                if y==0: multiplier*=0.5
                upperLeftShift=rollFill(leftShift,y,0)
                upperRightShift=rollFill(rightShift,y,0)
                lowerLeftShift=rollFill(leftShift,-y,0)
                lowerRightShift=rollFill(rightShift,-y,0)
                result+=(upperLeftShift+upperRightShift+lowerLeftShift+lowerRightShift)*multiplier*(self.pixelDecay**(x+y))
        #result=result.clamp(min=0.0)
        result=result*10/result.max()
        result=result.clamp(min=0.0, max=1.0)
        #result=result.repeat(1,1,3)
        return result.detach().cpu().numpy()
import random
class TemporalMaskEnhancer:
    def __init__(self, wrappedMask, maxDelta=3, samples=2):
        self.wrapped=wrappedMask
        self.maxDelta=maxDelta
        self.samples=samples
    def make_frame(self, t):
        frame=torchFrame(self.wrapped.get_frame(t))
        result=frame
        if not self.wrapped.ismask:
            result=result.sum(2)
        for i in range(self.samples):
            otherframe=frame
            targetTime=t+random.uniform(-self.maxDelta,self.maxDelta)
            try: 
                otherframe=torchFrame(self.wrapped.get_frame(targetTime))
            except Exception as e:
                print(f"Couldn't fetch time {targetTime} due to {e}")
                pass
            result=(result*result*result*otherframe)**(1/4.0)
        result=result.clamp(0,255)
        return result.int().detach().cpu().numpy()
        
import torch.nn.functional as F
class RaftMaskMaker:
    def __init__(self, wrappedClip, target_fps=None, backwards=False):
        
        self.backwards=backwards
        self.target_fps=target_fps
        if not self.target_fps:
            self.target_fps=wrappedClip.fps
        self.maximize=True
        self.wrapped=wrappedClip
            
        self.initModel()
        self.frames=[self.format(torchFrame(x)).cpu() for x in self.wrapped.iter_frames(fps=self.target_fps)]
        self.cachedDiffs=[None for x in self.frames]
        self.backwardsCache=[None for x in self.frames]
        self.makeCache()
    def initModel(self):
        from argparse import Namespace
        modelname="D:/git/RAFT/models/raft-sintel.pth"
        
        #self.model = torch.nn.DataParallel(RAFT(Namespace(small=False,mixed_precision=False)))
        self.model = torch.nn.DataParallel(RAFT(Namespace(small=False,mixed_precision=False)))
        self.model.load_state_dict(torch.load(modelname))

        self.model = self.model.module
        self.model.to('cuda')
        self.model.eval()
    def makeCache(self):
        for i in range(len(self.frames)):
            index=i
            print(f'making cache for frame {index}')
            if index==len(self.frames)-1:
                index=index-1    
            frame1=self.frames[index].cuda()
            frame2=self.frames[index+1].cuda()
            try:
                flow_low, flow_up = self.model(frame1, frame2, iters=10, test_mode=True)
                if self.backwards:
                    back_low, back_up=self.model(frame2, frame1, iters=10,test_mode=True)
            except RuntimeError:
                import pdb
                pdb.set_trace()
            if self.backwards:
                self.backwardsCache[i]=back_up[0].permute(1,2,0).abs().detach()
            self.cachedDiffs[i]=flow_up[0].permute(1,2,0).abs().detach()
    #help with slow segments
    def findMissing(self, index, gap):
        threshold=3
        flow_up=self.cachedDiffs[index]
        if index>(gap-1) and index<len(self.cachedDiffs)-gap:
            before=self.cachedDiffs[index-gap]>threshold
            
            after=self.cachedDiffs[index+gap]>threshold
            current=(flow_up>threshold)
            #if we had it before now, and after now, but not now, it's a good chance it's missing
            return (before&after)&(~current)
        else:
            return flow_up.new_full(flow_up.shape,False,dtype=torch.bool)
    
    
    def make_frame_data(self,t):
        threshold=3
        index=int(self.target_fps*t)
        
        if self.backwards:
            if index==len(self.cachedDiffs)-1:
                flow_up=self.backwardsCache[index-1]
            elif index==0:
                flow_up=self.cachedDiffs[index]
            else:
                flow_up=self.backwardsCache[index-1]+self.cachedDiffs[index]
        else:
            flow_up=self.cachedDiffs[index]
            
        
        modified=flow_up*1
        if self.maximize:
            modified[modified >threshold]=255.0
        imgresult=torchFrame(self.wrapped.get_frame(t))*0
        imgresult[:,:,0:2]=modified
        
        
        #missingIndexes=self.findMissing(index,1)
        #missingIndexes=missingIndexes|self.findMissing(index,2)
        #missingIndexes=missingIndexes[:,:,0]|missingIndexes[:,:,1]
        #toModify=imgresult[missingIndexes]
        #toModify[:,2]=255.0
        #imgresult[missingIndexes]=toModify

        
        return imgresult.detach()
    def make_frame(self,t):
        return (self.make_frame_data(t)).int().detach().cpu().numpy()
    def make_mask(self, t):
        result= self.make_frame_data(t)
        result=result.clamp(min=0)
            
        #result[result>0.2]=255
        return (result.max(2).values/255.0).detach().cpu().numpy()
        
    def format(self, frame):
        #return frame.permute(2,0,1)
        return self.pad(frame.permute(2,0,1)[None])
    def pad(self, frame):
        left, right=self.getpad(frame.shape[-2])
        top, bottom=self.getpad(frame.shape[-1])
        return F.pad(frame, [left,right, top,bottom],mode='replicate')
    def getpad(self, size):
        amount=(8-size%8)%8
        return amount//2, amount//2+amount%2
        
        
from datetime import datetime
class PerformanceTracker:
    def __init__(self,parent=None):
        self.parent=parent
        self.startTime=datetime.now()
        self.lastEvent=None
        self.log=[]
    def event(self, name):
        called=datetime.now()
        created_child=PerformanceTracker(self)
        if self.lastEvent:
            oldtime, oldname, child=self.lastEvent
            if(child.log):
                child.event('[[END]]')
            self.log.append( (name, f'{oldname} to {name}', called-oldtime,created_child) )
        else:
            self.log.append( (name,f'time from init until {name}',called-self.startTime,created_child) )
        self.lastEvent=called, name, created_child
        return created_child
    def do_print(self, depth=0,max_depth=None, minSeconds=0.0):
        if depth==0:
            self.event("[[PRINTED]]")
            
        if max_depth is not None and depth>max_depth:
            return
        for name, message, span, child in self.log:
            if(span.total_seconds()>minSeconds):
                print(' '*depth+f'*{message:<30} {span.total_seconds():0>4.3f}')
            child.do_print(depth+1,max_depth, minSeconds)
         
            
        
class CombinedDepthAnalysis:
    def __init__(self, wrapped, fps):
        self.raft=RaftMaskMaker(wrapped,fps)
        self.depth=MidasDepthFramer(wrapped)
        self.wrapped=wrapped
        self.fps=fps
        self.depth_cache={}
        self.motion_cache={}
        self.hue_cache={}
        self.result_cache={}
    def make_border_field(self, depth,max_gradient):
        decay=0.9
        radius=4#zero disables the loop
        
        result=depth.new_empty(depth.shape+(4,))
        result[:,:,2]=depth-rollFill(depth,-1,1)
        result[:,:,3]=depth-rollFill(depth,1,1)
        result[:,:,0]=depth-rollFill(depth,-1,0)
        result[:,:,1]=depth-rollFill(depth,1,0)
        #print(f'result metadata mean: {result.mean()} max: {result.max()} min: {result.min()} median: {result.median()}')
        #combine non-local gradients
        for x in range(-radius+1,radius):
            horizontal=rollFill(result, x,1)
            
            for y in range(-radius+1+abs(x),radius-abs(x)):
                if x==0 and y==0:
                    continue #bad, but who cares
                localdecay=decay**((x*x+y*y)**0.5)
                #local decay represents how much this point matters based on distance
                vertical=rollFill(horizontal,y,0)
                result=result+vertical*localdecay
        result=result.abs()
        print(f'result metadata mean: {result.mean()} max: {result.max()} min: {result.min()} stddev: {result.std()} median: {result.median()}')
        
        field= result.abs()<result.mean()+result.std()
        #seal some gaps
        #vertical=rollFill(field,0,-1) & rollFill(field,0,1)
        #field[:,:,2]&= vertical[:,:,2]
        #field[:,:,3]&= vertical[:,:,3]
        
        #horizontal=rollFill(field,1,-1) & rollFill(field,1,1)
        #field[:,:,0]&=horizontal[:,:,0]
        #field[:,:,1]&=horizontal[:,:,1]
        return field
    
    def alt_grow(self, max_growth,depth, motionPixels, preshrinks, max_gradient, hues, hue_limit):
        toptracker=PerformanceTracker()
        for iteration in range(preshrinks):
           changed=motionPixels.clone().detach()
           for shifts in (-1,1):
               for axis in (0,1):
                   rolled=rollFill(motionPixels,shifts,axis,fillValue=False)
                   changed=changed&rolled
           motionPixels=changed
        sourcetracker=toptracker.event('shrinks')
        
        
        xcoords=torch.arange(0,depth.shape[1]).repeat(depth.shape[0],1).unsqueeze(2).repeat(1,1,depth.shape[2]).float().to('cuda')
        ycoords=torch.arange(0,depth.shape[0]).unsqueeze(1).repeat(1,depth.shape[1]).unsqueeze(2).repeat(1,1,depth.shape[2]).float().to('cuda')
        grown_to=motionPixels.clone().detach()
        depth_sources=depth.new_full(depth.shape+(2,),-1,dtype=torch.float) #near and far that find it
        depth_sources[motionPixels]=depth[motionPixels].unsqueeze(1).repeat(1,2)
        
        sourcetracker.event('A')
        
        hue_sources=hues.new_full(hues.shape+(2,),-1,dtype=torch.float) #near and far that find it
        hue_sources[motionPixels]=hues[motionPixels].unsqueeze(1).repeat(1,2)
        
        sourcetracker.event('B')
        
        horizontal_sources=depth.new_full(motionPixels.shape+(2,),-1,dtype=torch.float) #left and right that find it
        horizontal_sources[motionPixels]=xcoords[motionPixels].unsqueeze(1).repeat(1,2)
        
        sourcetracker.event('C')
        
        vertical_sources=depth.new_full(motionPixels.shape+(2,),-1,dtype=torch.float) #top and bottommost that find it
        vertical_sources[motionPixels]=ycoords[motionPixels].unsqueeze(1).repeat(1,2)
        
        sourcetracker.event('D')
        
        stepSizes=[]
        added=1
        while sum(stepSizes)<max_growth//2:
            stepSizes.append(added)
            added*=2
        stepSizes=stepSizes+stepSizes[::-1]
        
        itertracker=toptracker.event('before iterations')
        for stepSize in stepSizes:
            looptracker=itertracker.event(f'{iteration}')
            iterationGrowth=grown_to.clone().detach()
            
            if stepSize==1:
                axes=(0,1,2)
            else:
                axes=(0,1)
            
            for shifts in (-stepSize,stepSize):
                for axis in axes:
                    
                    loop_hue=hue_limit
                    loop_gradient=max_gradient
                    
                    if axis==2:
                        loop_gradient*=2
                        loop_hue=loop_hue/4
                    looptracker.event('begin rolls')
                    candidates=rollFill(grown_to, shifts, axis, fillValue=False)
                    
                    rolledDepth=rollFill(depth_sources, shifts,axis,fillValue=-1)
                    rolledHue=rollFill(hue_sources, shifts,axis, fillValue=0)
                    
                    looptracker.event('candidate selection')
                    #rolledHor=rollFill(horizontal_sources,shifts,axis,fillValue=-1)
                    #rolledVer=rollFill(vertical_sources,shifts,axis,fillValue=-1)
                    
                    candidates&= (rolledDepth[:,:,:,0] - depth[:] < loop_gradient) & (depth[:]-rolledDepth[:,:,:,1] <loop_gradient)
                    #at some point address that hue is a modulo ring
                    candidates&= (rolledHue[:,:,:,0] - hues[:] < loop_hue) & (hues[:]-rolledHue[:,:,:,1] <loop_hue)
                    
                    newCandidates=candidates&~grown_to
                    newCandidateSources=newCandidates.roll(-shifts,axis)#rollFill(newCandidates,-shifts,axis,fillValue=False)
                    iterationGrowth[newCandidates]=True
                    
                    looptracker.event('update sources')
                    depth_sources[newCandidates]=depth_sources[newCandidateSources]
                    hue_sources[newCandidates]=hue_sources[newCandidateSources]
                    #horizontal_sources[newCandidates]=horizontal_sources[newCandidateSources]
                    #vertical_sources[newCandidates]=vertical_sources[newCandidateSources]
                    #expand minimums
                    #depth_sources[candidates,0]=torch.min(depth_sources[candidates,0],rolledDepth[candidates,0])
                    self.updateMinMaxField(depth_sources,rolledDepth,candidates)
                    self.updateMinMaxField(hue_sources,rolledHue,candidates)
                    #self.updateMinMaxField(horizontal_sources,rolledHor,candidates)
                    #self.updateMinMaxField(vertical_sources,rolledVer,candidates)
            #we've stopped growing
            #diffs=(grown_to!=iterationGrowth)
            #print(f'{iteration} produced {diffs[diffs].numel()}')
            if (grown_to==iterationGrowth).all():
                break
            grown_to=iterationGrowth
        toptracker.do_print(minSeconds=0.1)
        
        
        return grown_to
    def updateMinMaxField(self,source,rolled_source,candidates):
        chunk=source[candidates]
        rolledChunk=rolled_source[candidates]
        chunk[:,0]=torch.min(chunk[:,0],rolledChunk[:,0])
        chunk[:,1]=torch.max(chunk[:,1],rolledChunk[:,1])
        source[candidates]=chunk
        
    def do_grow(self, depth, max_growth, max_gradient, initial_values, preshrinks):
        for iteration in range(preshrinks):
            initial_values=self.shrink(initial_values)
        border=self.make_border_field(depth, max_gradient)
        result=initial_values
        for iteration in range(max_growth):
            result=self.grow(result, border, max_growth)
        return result
        
    def shrink(self, current_values):
        changed=current_values*1
        for shifts in (-1,1):
            for axis in (0,1):
                rolled=current_values.roll(shifts,axis)
                changed=torch.max(changed,rolled)
        return changed
    
    def grow(self,current_values, border_field, max_growth):
        increased=current_values+1
        growth_candidates=current_values<max_growth
        new_values=current_values*1       
        
        
        for shifts in (-2,3):
            for axis in (0,1):
                border_index=(shifts+1)//2 + axis*2
                rolled=increased.roll(shifts,axis)
                indexes=(current_values>rolled) & growth_candidates.roll(shifts, axis) &  ~border_field.any(2)#border_field[:,:,border_index]
                new_values[indexes]=rolled[indexes]
        return new_values
        
        
    
    def get_depth(self,t):
        result=self.depth_cache.get(t,None)
        if result is None:
            result=torchFrame(self.depth.make_frame(t))[:,:,0].squeeze().detach()
            self.depth_cache[t]=result
        return result
    def get_motion(self, t):
        result=self.motion_cache.get(t,None)
        if result is None:
            result=torchFrame(self.raft.make_mask(t))>0.5
            self.motion_cache[t]=result
        return result
        
    def get_hue(self, t):
        result=self.hue_cache.get(t,None)
        if result is None:
            result=self.make_hue_frame(torchFrame(self.wrapped.get_frame(t))).detach()
            self.hue_cache[t]=result
        return result
            
    def make_hue_frame(self, frame):
        normalized=frame/255
        minval,_=normalized.min(2)
        maxval,maxindex=normalized.max(2)
        delta=maxval-minval
        result=frame.new_empty(normalized.shape[:-1])
        
        reds=(normalized[:,:,0]==maxval)&(normalized[:,:,0]!=0)
        redvals=normalized[reds]
        
        result[reds]=((redvals[:,1]-redvals[:,2])/delta[reds])
        
        greens=(normalized[:,:,1]==maxval)&(normalized[:,:,1]!=0)
        greenvals=normalized[greens]
        result[greens]=((greenvals[:,2]-greenvals[:,0])/delta[greens])+2
        
        
        blues=(normalized[:,:,2]==maxval)&(normalized[:,:,2]!=0)
        bluevals=normalized[blues]
        result[blues]=((bluevals[:,0]-bluevals[:,1])/delta[blues])+4
        
        
        return result*42.5 #scale to 255 space
        
    def make_frame(self, t):
        
        max_gradient=300
        #motionflow=torchFrame(self.raft.make_mask(t))>0.5
        #depthdata=torchFrame(self.depth.make_frame(t))[:,:,0].squeeze()
        step=1.0/self.fps
        times=[]
        
        t=t-(t%step)#just generate the same frame over and over
        if t in self.result_cache:
            return self.result_cache[t]
        
        for frameindex in range(-1,2):
            tval=t+frameindex*step
            tval=min(max(tval,0),self.wrapped.duration-step)
            times.append(tval)
        middleT=(len(times)-1)//2
        
        tplus=min(t+step,self.wrapped.duration-step)
        depth4d=torch.stack([self.get_depth(x) for x in times],2) #(self.get_depth(tminus), self.get_depth(t), self.get_depth(tplus)),2)
        motion4d=torch.stack([self.get_motion(x) for x in times], 2) #(self.get_motion(tminus), self.get_motion(t), self.get_motion(tplus)),2)
        hues4d=torch.stack([self.get_hue(x) for x in times],2)
        motionflow=motion4d[:,:,middleT]

        result=torchFrame(self.wrapped.make_frame(t))*0#depthdata.new_empty(depthdata.shape+(3,))
        

        #borders=~self.make_border_field(depthdata,max_gradient).all(2)
        #chunk=result[borders]
        #chunk[:,0]=255
        #result[borders]=chunk
        
        
        
        chunk=result[motionflow]
        chunk[:,1]=255
        result[motionflow]=chunk
        
        #grown=depthdata.new_full(depthdata.shape,1000)
        #grown[motionflow]=0
        #grown=self.do_grow(depthdata, 50, max_gradient, grown, 3)
        
        
        grown=self.alt_grow(max_growth=30,depth=depth4d, motionPixels=motion4d, preshrinks=2, max_gradient=5, hues=hues4d, hue_limit=5)
        #nuke the areas we never moved into
        
        #scale the rest to show how much growing had to happen to fill it with blue
        #grown=grown/20*128.0
        #grown[grown>0]+=127
        grown=grown[:,:,1]#discard the time dimension, focus on the present
        chunk=result[grown]
        chunk[:,2]=255
        result[grown]=chunk
        #result[:,:,2]=grown/3
        
        full_result= result.detach().cpu().numpy()
        self.result_cache[t]=full_result
        return full_result
    def make_mask(self, t):
        source=torchFrame(self.make_frame(t))
        result=source[:,:,0]*0
        result[(source[:,:,0]>1) | (source[:,:,1]>1) | (source[:,:,2]>1)]=1.
        return result.detach().cpu().numpy()
        
class SegmentedFlowMaker:
    def __init__(self, wrapped,activityThreshold=.2, target_segments=500,large_segment_preference=1):
        wrapped.fps=8
        self.wrapped=wrapped
        from segmentation import flood_video
        
        
        print('making segments')
        self.colors,_,segments,_=flood_video(wrapped,flexibility=5, target_segments=target_segments,color_borders=False)
        del _
        seg_ids=segments.unique().detach()
        print(f'number of segment ids: {seg_ids}')
        self.colors=self.colors.detach().cpu()
        segments=segments.detach().cpu()
        import gc
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        
        count=0
        print('Running raft')
        self.raft=RaftMaskMaker(wrapped,8, backwards=True)
        raft_data=torch.stack([(self.raft.make_frame_data(i/8)>1).any(-1) for i in range(int(8*wrapped.duration))],dim=2).cuda()
        frame_pixel_counts=raft_data.sum(0).sum(0)
        average_pixels=frame_pixel_counts.float().mean()
        for frame_index, count in enumerate(frame_pixel_counts):
            searchBackwards=False
            if count<average_pixels/2:
                new_index=frame_index
                while count<average_pixels/2:
                    new_index+=1
                    if new_index>=len(frame_pixel_counts):
                        new_index=frame_index
                        searchBackwards=True
                        break
                    count=frame_pixel_counts[new_index]
                if searchBackwards:
                    while count<average_pixels/2:
                        new_index-=1
                        if new_index<0:
                            print(frame_index, 'was impossible to repair')
                            new_index=frame_index
                            break
                        count=frame_pixel_counts[new_index]
                raft_data[...,frame_index]|=raft_data[...,new_index]
                        
                        
        self.boolMask=raft_data.clone().bool()#be explicit that this a bool
        #print('running resnet')
        #self.resnetMattes=torch.stack([createMatteFrame(torchFrame(x),15) for x in wrapped.iter_frames(fps=8)], dim=2).cpu()
        #print(self.resnetMattes.shape)
        
        seg_ids=seg_ids.cuda()
        segments=segments.cuda()
        areaThreshold=0.9
        self.activatedSegments=[]
        self.colors=self.colors.cuda()
        for index in seg_ids:
            gc.collect(0)
            #print(f'checking index #{count} id: ({index})')
            segment=(segments==index)
            #print(f'color is {self.colors[segment][0]}')
            count+=1
            activityOverlap=segment&raft_data
            
            activityRatio=((activityOverlap).sum().float()-large_segment_preference)/segment.sum().float()
            #print('ratio was ', activityRatio)
            if activityRatio>activityThreshold and index>-1:
                print(f'pass!: size {segment.sum().item()}')
                self.activatedSegments.append(index.item())
                #self.boolMask|=segment
            else:
                if activityOverlap.any():
                    area=(segment.nonzero().max(0).values-segment.nonzero().min(0).values).prod()
                    activityArea=(activityOverlap.nonzero().max(0).values-activityOverlap.nonzero().min(0).values).prod()
                    if activityArea.float()/area.float()>areaThreshold:
                        print('second pass!',activityArea, area, activityOverlap.nonzero().max(0).values, activityOverlap.nonzero().min(0).values,segment.nonzero().max(0).values, segment.nonzero().min(0).values)
                        self.activatedSegments.append(segment)
                        #self.boolMask|=segment
                    else:
                        self.colors[segment]//=4
                else:
                    self.colors[segment]//=4
        for activatedIndex in self.activatedSegments:
            segment=(segments==activatedIndex)
            self.boolMask|=segment
        for frame_index in range(self.boolMask.shape[2]):
            frame=self.boolMask[...,frame_index]
            print('before', self.boolMask.sum())
            mapped=frame.unfold(0,9,1).unfold(1,9,1)
            #reduced=mapped.sum(-1).sum(-1)>25
            #we expect non-trivial cases to have a nearby "real" corner
            #here's why: if you can't go diagonal in any direction a few pixels and find something, you're likely to be a horizontal or vertical line
            #need to address diagonal lines too, though
            
            reduced=mapped[...,0,0]|mapped[...,0,-1]|mapped[...,-1,0]|mapped[...,-1,-1]
            reduced&=mapped[...,1,1]|mapped[...,1,-2]|mapped[...,-2,1]|mapped[...,-2,-2]
            #reduced&=mapped[...,5,0]|mapped[...,5,-1]|mapped[...,0,5]|mapped[...,-1,5] #let's also demand one straight direction
            increased=mapped.sum(-1).sum(-1)>7*7
            
            self.boolMask[4:-4,4:-4,frame_index]&=reduced
            self.boolMask[4:-4,4:-4,frame_index]|=increased
            print('after', self.boolMask.sum())
            
                
    def make_segment_rgb(self, T):
        result= self.colors[:,:,int(8*T)]
        #result[result==self.skipped_segments]//=4
        return result.cpu().numpy()
    def make_mask(self, T):
        T=min(T, self.wrapped.duration-1/8)
        mask_frame=self.boolMask[:,:,int(8*T)]
        result=mask_frame.new_full(mask_frame.shape,fill_value=0,dtype=torch.float)
        result[mask_frame]=1.0
        return result.detach().cpu().numpy()
    def make_resnet(self, T):
        T=min(T, self.wrapped.duration-1/8)
        frame=self.resnetMattes[:,:,int(8*T)]
        result=frame.new_full(frame.shape+(3,),0,dtype=torch.uint8)
        result[frame]=255
        return result.cpu().numpy()
        
        
def count(radius, dimension):
    result=0
    if dimension==0:
        return 1
    for i in range(-radius,radius+1):
        result+=count(radius-abs(i),dimension-1)
    return result