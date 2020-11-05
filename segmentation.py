import torch
import torch.cuda

import torch.nn.functional as F
from torch.nn import Unfold

from torchmotion import rollFill
import gc

from concurrent.futures import ThreadPoolExecutor



import pdb

DEBUG=True

def get_image(name):
    from PIL import Image
    import numpy
    img=Image.open(name)
    return torch.from_numpy(numpy.asarray(img)).cuda()
    
    
def write_img(resultdata,name):
    from PIL import Image
    import numpy
    Image.fromarray(resultdata.cpu().numpy().astype(numpy.uint8)).save(name)
    
"""
import segmentation
from moviepy.editor import *
from segmentation import write_img, get_image, get_neighborhood
def r():
    import importlib
    importlib.reload(segmentation)
    
#r(); result, borders, segments,neighbors=segmentation.flood_one_frame(get_image('fademe.png'), flexibility=3)
#r(); result, borders, segments,neighbors=segmentation.flood_video(VideoFileClip('dance.mp4'), flexibility=0, color_borders=False)
"""



    

def flood_one_frame(frame, flexibility=0, target_segments=75,color_borders=True):
    datasource=make_hsv(torch.tensor(frame).cuda())[...,2]
    mins,_=identify_local_reduction(datasource, (13,13), lambda a, dim: torch.min(a,dim=dim).values, 255)
    maxs,_=identify_local_reduction(datasource, (13,13), lambda a, dim: torch.max(a,dim=dim).values, 0)
    
    results=flood_anything(datasource,mins,maxs,flexibility,target_segments,color_borders)
    write_img(results[0], 'flooded3.png')
    return results
    
maximum_segments=10000
def flood_video(video, flexibility=0, target_segments=120,color_borders=False, viewer=None):
    with torch.no_grad():
        video.fps=8
        frames=torch.stack([torch.from_numpy(frame).float().cuda() for frame in video.iter_frames()],dim=-2)
        datasource=make_hsv(frames)[...,2] #switch from value to average of hsv
        #datasource=(sobel3d(datasource[...,2])+datasource[...,0])/2
        #sample_size=int((datasource.numel()//target_segments)**(1/3))+1
        #sample_size+=(sample_size+1)%2
        sample_size=9
        
        del frames
        mins,_=identify_local_reduction(datasource, (sample_size,sample_size,max(3,sample_size)), lambda a, dim: torch.min(a,dim=dim).values, 255)
        maxs,_=identify_local_reduction(datasource, (sample_size,sample_size,max(3,sample_size)), lambda a, dim: torch.max(a,dim=dim).values, 0)
        
        target_segments=min(target_segments, maximum_segments)
        print('before culling',(mins|maxs).sum())
        while (mins|maxs).sum()>target_segments:
            culling=torch.rand(datasource.shape).cuda()>0.25
            mins&=culling
            maxs&=culling
            del culling
        print('after culling',(mins|maxs).sum())
        results=flood_anything(datasource,mins,maxs,flexibility,target_segments,color_borders, False, viewer)
        return results
    
    #gradients=torch.zeroes(datasource.shape).float()
    #for index,dimension in enumerate(datasource.shape)
    #    gradients+=(datasource-datasource.roll(dimension, 1)).abs()
    #we now have a uniform difference field

def sobel_video(video):
    frames=torch.stack([torch.from_numpy(frame).float().cuda() for frame in video.iter_frames()],dim=-2)
    hsv=make_hsv(frames)
    return torch.stack([sobel3d(hsv[...,2]) for i in range(3)],3)
def sobel3d(datasource):
    xkern=torch.tensor([
        [
        [1,0,-1],
        [2,0,-2],
        [1,0,-1]
        ],[
        [2,0,-2],
        [4,0,-4],
        [2,0,-2]
        ],[
        [1,0,-1],
        [2,0,-2],
        [1,0,-1]
        ]]).float().cuda()
    ykern=xkern.transpose(1,2)
    zkern=xkern.transpose(0,2)
    result=datasource.clone()
    datasource=F.pad(datasource,pad=[1,1,1,1,1,1],value=0)
    datasource[0,:,:]=datasource[1,:,:]
    datasource[-1,:,:]=datasource[-2,:,:]
    datasource[:,0,:]=datasource[:,1,:]
    datasource[:,-1,:]=datasource[:,-2,:]
    datasource[:,:,0]=datasource[:,:,1]
    datasource[:,:,-1]=datasource[:,:,-2]
    
    
    view=datasource.unfold(0,3,1).unfold(1,3,1).unfold(2,3,1)
    for resultchunk,viewchunk in zip(result.chunk(27,2),view.chunk(27,2)):
        gx=(xkern*viewchunk).sum(-1).sum(-1).sum(-1)
        gy=(ykern*viewchunk).sum(-1).sum(-1).sum(-1)
        gz=(zkern*viewchunk).sum(-1).sum(-1).sum(-1)
        resultchunk[:]=(gx**2+gy**2+gz**2)**0.5
    #result=(result*result)/result.mean()#amplify the high, deamp the low
    return result.clamp(0,255)
    
    
def flood_anything(datasource,mins,maxs,flexibility,target_segments,color_borders=True, combine_segments=False, viewer=None):
    
    gc.collect(0)#free our GPU RAM
    segment_id_field, segment_ids=flood_pixels(mins, maxs,datasource, flexibility)
    neighbor_matrix=None
    if combine_segments:
        gc.collect(0)
        highest_id=segment_ids.max().item()
        neighbor_matrix=torch.zeros((highest_id+2,highest_id+2),dtype=torch.bool).cuda()
        print('identifying segment neighbors')
        for dimension,_ in enumerate(segment_id_field.shape):
            for direction in (-1,1):
                rolled_seg_id=rollFill(segment_id_field,direction,dimension,-1)
                pairs=torch.stack((segment_id_field, rolled_seg_id),-1).reshape(-1,2).unique(dim=0)
                neighbor_matrix[pairs[:,0],pairs[:,1]]=True #will write -1(i.e. last) for image boundaries,
        neighbor_matrix=neighbor_matrix[:-1,:-1]#cull boundaries we don't want to loop on edges
        
        averages=datasource.new_empty(segment_ids.shape)
        print(f'averaging {averages.numel()} segments')
        for i in segment_ids:
            averages[i]=datasource[segment_id_field==i].mean()
            if(neighbor_matrix[i].sum()<2):
                print(f'index {i} is isolated.  Size: {(segment_id_field==i).sum()}')
            #sizes[i]=(datasource[segment_id_field==i]**0).sum()
        #quadtratic formula solve: 
        ratio=(highest_id+1)/target_segments
        distance= (((2*ratio-1)**0.5)-1)/len(datasource.shape)
        
        print('generating neighborhood matrix')
        distance=abs(distance)
        #we get absurd impossible complex distances when we want more segments than we have
        hood=get_neighborhood(neighbor_matrix, max_distance=distance+1)
        

        print('identifying extreme segments')
        #pdb.set_trace()
        max_segments=identify_segment_reduction(hood, averages,lambda a: torch.max(a),distance)
        min_segments=identify_segment_reduction(hood, averages,lambda a: torch.min(a),distance)
        del hood
        gc.collect(0) #clean up that huge thousands by thousands matrix
        both=torch.cat((max_segments,min_segments),0)
        flood_segments(both,averages, segment_id_field, neighbor_matrix)
        
        
    #somehow fix the border
    print('recomputing borders for expanded segments')
    resborders=segment_id_field.new_full(segment_id_field.shape, False, dtype=torch.bool)
    resborders[:]=False
    for dimension,_ in enumerate(resborders.shape):
        for direction in (-1,):#only go one direction for border check
            rolled_seg_id=rollFill(segment_id_field,direction,dimension,-1)
            resborders|=(segment_id_field!=rolled_seg_id) & (rolled_seg_id>-1)
    
    
    #colors[0]=torch.tensor([255,0,0]).cuda()
    print('final segmentation result completion')
    remaining=segment_id_field.unique()
    remainingHsv=make_colors(remaining.numel())
    remainingColors=make_rgb(remainingHsv)
    colors=segment_ids.new_full(segment_ids.shape+(3,),0,dtype=torch.int)
    colors[remaining]=remainingColors
    
    result=colors[(segment_id_field)].detach()
    if color_borders:
        result[resborders]=torch.tensor([255,0,0]).int().cuda()
    
    return result, resborders, segment_id_field, neighbor_matrix

def make_colors(color_count,loops=3):
    return torch.stack(((torch.arange(color_count)*loops*255//(color_count))%255, (torch.arange(color_count)%2)*128+127,torch.ones(color_count)*255),-1).cuda()
    
    

from datetime import datetime


def timeall(block, context):
    alltimes=[]
    statements=block.split('\n')
    for statement in statements:
        statement=statement.strip()
        if statement:
            alltimes.append(time_statement(statement,context))
        
    print ('longest statements')
    for time, statement in sorted(alltimes)[-5:]:
        print(time,'::', statement)
        

def time_statement(statement,context):
    start=datetime.now()
    exec(statement,globals(), context)
    return (datetime.now()-start).total_seconds(),statement

def flood_pixels(mins, maxs,datasource, flexibility,base_increment=2, acceleration=4):
    
    padding=sum([[1,1] for x in datasource.shape],[])
    realContentViewSlices=[slice(1,-1) for x in datasource.shape]
    
    alive_padded=F.pad(mins|maxs,pad=padding, value=False)
    alive=alive_padded[realContentViewSlices]
    #dead_padded=F.pad(alive.new_full(alive.shape, False), pad=padding, value=True)
    #dead=dead_padded[realContentViewSlices] #kinda a stupid way to do this
    unborn=~alive
    newborns=alive.new_full(alive.shape,False)
    #newdead=alive.new_full(alive.shape,False)
    borders=alive.new_full(alive.shape,False)
    
    segment_id_field_padded=F.pad(datasource.clone().long().cpu(),pad=padding).pin_memory().cuda(non_blocking=True)
    segment_id_field=segment_id_field_padded[realContentViewSlices]
    segment_id_field[unborn]=-1
    segment_ids=torch.arange(alive.sum()).cuda()
    #origins=alive.nonzero()
    segment_id_field[alive]=segment_ids
    
    segment_rolledviews={}
    alive_rolledviews={}
    #dead_rolledviews={}
    
    for dimension,_ in enumerate(alive.shape):
        
        segunfold=segment_id_field_padded.unfold(dimension,3,1)
        alive_unfold=alive_padded.unfold(dimension,3,1)
        #dead_unfold=dead_padded.unfold(dimension,3,1)
        for direction in (-1,1):
            slices=realContentViewSlices[:dimension]+[slice(None,None)]+realContentViewSlices[dimension+1:]+[1+direction]
            key=(dimension,direction)
            segment_rolledviews[key]=segunfold[slices]
            assert(segment_rolledviews[key].shape==segment_id_field.shape), ('shapes are bad', segment_rolledviews[key].shape, segment_id_field.shape)
            alive_rolledviews[key]=alive_unfold[slices]
            #dead_rolledviews[key]=dead_unfold[slices]
            
    base_depth=datasource[alive]
    
    allkeys=sorted(segment_rolledviews.keys())
   
    #actually an int, not a tensor
    flood_level=0
    dimensional_cost=[2,2,1]
    #pool=ThreadPoolExecutor(4)
    from operator import or_
    from itertools import accumulate
    while unborn.any() and alive.any():
        #update alive state
        newborns[:]=False
        #newdead[:]=alive[:]
        #findl=lambda key: find_new_in_dir(key,segment_id_field,dimensional_cost,alive_rolledviews,segment_rolledviews,base_depth,unborn,datasource,flood_level)
        #for newborn in pool.map(findl,allkeys):
        #    newborns|=newborn
        
        
        for dimension,_ in enumerate(alive.shape):
            cost=dimensional_cost[dimension]
            for direction in (-1,1):
                key=(dimension,direction)
                rolledalive=alive_rolledviews[key]
                rolled_seg_id=segment_rolledviews[key]
                flood_base=base_depth.take(rolled_seg_id)
                newthisdir=unborn&rolledalive&(((flood_base-datasource).abs())<(flood_level/cost))
                segment_id_field[newthisdir]=rolled_seg_id[newthisdir]
                newborns|=newthisdir
                
        alive|=newborns
        #dead|=newdead
        #alive&=~dead
        unborn=~alive   
        if (newborns).sum()<=flexibility and (alive).sum()>flexibility:
            print(f'increasing flood, {alive.sum().item()/alive.numel()*100:2.2f}% done')
            if flood_level<10:
                flood_level+=base_increment
            elif flood_level<50:
                flood_level+=base_increment+acceleration
            else:
                flood_level+=base_increment+acceleration*2
        else:
            print(f'keeping flood level at {flood_level} due to {newborns.sum()} new pixels')
    return segment_id_field.clone().detach(),segment_ids #it doesn't matter what our original source for this stuff is
    
def find_new_in_dir(key,segment_id_field,dimensional_cost,alive_rolledviews,segment_rolledviews,base_depth,unborn,datasource,flood_level):
    dimension, direction=key
    cost=dimensional_cost[dimension]
    rolledalive=alive_rolledviews[key]
    rolled_seg_id=segment_rolledviews[key]
    flood_base=base_depth.take(rolled_seg_id)
    newthisdir=unborn&rolledalive&((flood_base-datasource).abs()<(flood_level/cost))
    segment_id_field[newthisdir]=rolled_seg_id[newthisdir]
    return newthisdir
    
def flood_segments(extreme_segments,values, segment_id_field, neighbors):
    alive=values.bool()
    alive[:]=False
    indexes=torch.arange(values.shape[0]).long().cuda()
    #indexes[:]=-1
    
    assigned_values=values.new_full(values.shape,2000)
    assigned_values[extreme_segments]=values[extreme_segments]
    penalties=values.new_full(values.shape,0)
    
    assigned_indexes=torch.zeros(values.shape, dtype=torch.long).cuda()
    assigned_indexes[:]=-1
    assigned_indexes[extreme_segments]=extreme_segments
    #differences=values.new_full(liveconnections.shape,2000)
    floodable=values.new_full(neighbors.shape, False, dtype=torch.bool)
    alive[extreme_segments]=True
    flood_level=0
    new_alive=alive.clone()
    #assignments_full=assigned_indexes.new_full(neighbors.shape,30000)
    while (~alive).any() and flood_level<255:
        new_alive[:]=False
        
        liveassigned=assigned_indexes[alive]
        
        #Alive X All grid of connections
        liveconnections=neighbors[alive]
        
        #base_values=assigned_values[alive].expand(liveconnections.t().shape).t()
        #differences=
        
        
        
        #Alive-x-All grid of base values that can spread
        
        numchunks=20#this seems excessive 
            #chunk x all     chunk x all     chunk x 1   chunk x 1     chunk x 1         chunk x 1         
            #All X All                      All X All (output)               All x 1                 All x 1                    All x 1                             all x 1
        for neighbor_chunk, floodable_chunk, alive_chunk,value_chunk, assigned_chunk, penalty_chunk in\
            zip(neighbors.chunk(numchunks,0),floodable.chunk(numchunks,0), alive.chunk(numchunks,0),values.chunk(numchunks,0), assigned_values.chunk(numchunks,0), penalties.chunk(numchunks)):
            #chunkalive x all
            connection_chunk=neighbor_chunk.new_full(neighbor_chunk.shape, False)
            connection_chunk[alive_chunk]=neighbor_chunk[alive_chunk]
            
            livevalues=assigned_chunk.new_zeros(assigned_chunk.shape)
            livevalues[alive_chunk]=assigned_chunk[alive_chunk]
            
            
            
                
            #alive x chunk                     
            value_assignments=values.new_full(connection_chunk.shape,2000)#impossible to have a sane difference at 2000
                                                #chunkalive x 1   #t()=all x chunkalive 
            #each row(all) represents the depth value that the chunk will receive from each alive, connected, source in the chunk(column)
            value_assignments[connection_chunk]=livevalues.expand(connection_chunk.t().shape).t()[connection_chunk]
            expanded_penalties=penalty_chunk.expand(connection_chunk.t().shape).t()
            
            
                             #expand()=
            difference_chunk=(value_chunk.expand(connection_chunk.t().shape).t()-value_assignments).abs()+expanded_penalties
            
            floodable_chunk[alive_chunk]=(difference_chunk<flood_level)[alive_chunk]
          #alive x all
        #livevalues=assigned_values[alive]
                            #alive x all                         t=()all x alive
        #value_assignments[liveconnections]=livevalues.expand(liveconnections.t().shape).t()[liveconnections]
        #differences=(values.expand(liveconnections.shape)-value_assignments).abs()
        #Alive-x-all grid of values that are deep enough to spread
        #floodable=differences<flood_level
        
        #valid_locations=~alive & (differences<flood_level)
        
        #alive-x-all grid of indexes being assigned
        gc.collect(0)
        #assignments=assignments_full[alive]
        #assignments[:]=30000
        #hey wait, we didn't clean up the old assignments variable
        expanded=liveassigned.expand(liveconnections.t().shape)
        #chunks are views and write underlying data
                    #all x 1-> chunk x 1         all x 1-> chunk x 1       Alive x all -> alive x chunk    alive x all -> alive x chunk
                    #all x alive -> chunk x all  all x 1 -> chunk x 1             all x 1 -> chunk x 1                  
        for         new_alive_chunk,             alive_chunk,             connectionchunk,                   floodable_chunk,\
                    expanded_chunk,              index_chunk,                       values_chunk,                        in \
                zip(new_alive.chunk(numchunks,0),alive.chunk(numchunks,0),liveconnections.chunk(numchunks,1),floodable[alive].chunk(numchunks,1),\
                    expanded.chunk(numchunks,0), assigned_indexes.chunk(numchunks,0),assigned_values.chunk(numchunks,0)) :
            mask_chunk=(connectionchunk&floodable_chunk).t()
            assignments=expanded_chunk.new_full(mask_chunk.shape,30000)
            assignments[mask_chunk]=expanded_chunk[mask_chunk]
            new_alive_chunk[~alive_chunk]=mask_chunk.any(1)[~alive_chunk]
            index_chunk[~alive_chunk]=assignments.min(1).values[~alive_chunk]
            notOutOfRange=index_chunk<30000
            values_chunk[~alive_chunk&notOutOfRange]=assigned_values[index_chunk[~alive_chunk&notOutOfRange]]
            
        indexes,counts=assigned_indexes.unique(return_counts=True)
        penalty_factor=0.3
        
        #penalties[indexes]=counts*penalty_factor
        #assignments[liveconnections&floodable]=expanded[liveconnections&floodable]
        
        #make all the dead squares that have connections and depth alive
        #new_alive[~alive]=(liveconnections&floodable[alive]).any(0)[~alive]
        #make all those newly alive squares have the lowest key that's spreading to them(arbitrary)
        #assigned_indexes[~alive]=assignments.min(0).values[~alive]
        
        #value_assignments=values.new_full(liveconnections.shape,0)
        #value_assignments[liveconnections]=livevalues.expand(liveconnections.t().shape).t()[liveconnections]
        
        #notOutOfRange=assigned_indexes<5000
        
        #for values_chunk, alive_chunk, range_chunk, index_chunk in zip(assigned_values.chunk(numchunks,0),alive.chunk(numchunks,0),notOutOfRange.chunk(numchunks,0),assigned_indexes.chunk(numchunks,0)):
            #note on the right side that we're plucking from ALL assigned values(not the chunk) because the indexes we come up with might fall outside
        #    values_chunk[~alive_chunk&range_chunk]=assigned_values[index_chunk[~alive_chunk&range_chunk]]
        #assigned_values[~alive&notOutOfRange]=assigned_values[assigned_indexes[~alive&notOutOfRange]]
        
        
        # for index, row in zip(indexes[alive],neighbors[alive]):
            # assigned=assigned_indexes[index]
            # differences=(values-values[assigned]).abs()
            # newdead[index]=(~row|alive|dead).all()
            # candidates=row & unborn & ~new_alive & (differences<flood_level)
            # new_alive|=candidates
            # assigned_indexes[new_alive]=assigned
            # for changed in indexes[candidates]:
                # segment_id_field[segment_id_field==changed]=assigned
        alive|=new_alive
        if not new_alive.any():
            print(f'increasing segment flood {flood_level}')
            if(flood_level>20):
                flood_level+=2
            else:
                flood_level+=0.125
        else:
            print('added', new_alive.sum().item(), 'segments')
    print('matching index field to segments')
    for index,assigned in enumerate(assigned_indexes):
        segment_id_field[segment_id_field==index]=assigned
            
    
def old_segment_linker():
    joins={}
    for own_id,row in enumerate(neighbor_matrix):
        neighbors=segment_ids[row]
        myjoins=[]
        for neighbor in neighbors:
            #only search bottom triangle
            if neighbor.item()>=own_id:
                continue
            if (averages[neighbor]-averages[own_id]).abs()<averages.std():
                myjoins.append(neighbor.item())
        if myjoins:
            joins[own_id]=myjoins
            
    while joins:
        key=sorted(joins.keys())[0]
        descendants=linkup(joins,key)
        joins.remove(key)
        for descendant in descendants:
            if descendant in joins:
                joins.remove(descendant)
                segment_ids[segment_ids==descendant]=key
    
    
def get_neighborhood(neighbors, max_distance=3):
    splits=max(neighbors.numel()//(10**7),1)
    #we lose a fair amount of connectivity, 
    neighborhood=neighbors.float()
    
    max_distance=max(max_distance,5)#even if we don't need it, at least get a few steps
    #if splits>1:
    for distance in range(2,int(max_distance)+1):
        distten=torch.Tensor([distance])[0]             #3 x X                       X x 3
        for vertical_chunk, outview_h in zip(neighbors.chunk(splits,0), neighborhood.chunk(splits,0)):
                                                    #X x 5                          3 x 5
            for horizontal_chunk,outview in zip(neighborhood.chunk(splits,1), outview_h.chunk(splits,1)):
                mathresult=vertical_chunk.float() @ horizontal_chunk
                has_value=mathresult!=0
                had_no_value=outview==0
                changed=has_value&had_no_value
                outview[changed]=distance
                #outview[:]=torch.min(outview,mathresult)
                #outview[:]=torch.min(outview, distten)
                #outview=torch.min(torch.min(vertical_chunk.float() @ horizontal_chunk,outview),distten)
    for chunk in neighborhood.chunk(splits):
        chunk[chunk==0]==1000
    #neighborhood[neighborhood==0]=1000
    neighborhood.fill_diagonal_(0)
    return neighborhood
                
    """neighborhood[neighbors]=1
    onestep=neighborhood.float()
    distance=1
    while (neighborhood==0).any() and distance<max_distance:
        distance=distance+1
        new_neighborhood=torch.matmul(neighborhood, onestep)
        #new_neighborhood[new_neighborhood!=0]=step
        #print(new_neighborhood, neighborhood)
        if not ((new_neighborhood!=0) & (neighborhood==0)).any():
            break
        neighborhood[(new_neighborhood!=0) & (neighborhood==0)]=distance
    neighborhood=neighborhood.float()
    neighborhood[neighborhood==0]=1000#we're never going to have segments thousnads away from each other
    return neighborhood.float()"""
    
    
def linkup(graph, key):
    result=set(key)
    if key not in graph:
        return result
    for subitem in graph[key]:
        result.union_update(linkup(subitem))
    return result
    
"""algorithm credit to https://www.rapidtables.com/convert/color/hsv-to-rgb.html"""
def make_rgb(hsv):
    shrunk=hsv/255.0
    C=shrunk[...,1]*shrunk[...,2]
    Hs=shrunk[...,0]*6
    X=C*(1-((Hs%2)-1).abs())
    m=shrunk[...,2]-C
    primes=shrunk.new_full(shrunk.shape,0)
    Z=C.new_full(C.shape, 0)#zero
    primes[(Hs>=0) & (Hs<1)]=torch.stack((C,X,Z),-1)[(Hs>=0) & (Hs<1)]
    primes[(Hs>=1) & (Hs<2)]=torch.stack((X,C,Z),-1)[(Hs>=1) & (Hs<2)]
    primes[(Hs>=2) & (Hs<3)]=torch.stack((Z,C,X),-1)[(Hs>=2) & (Hs<3)]
    primes[(Hs>=3) & (Hs<4)]=torch.stack((Z,X,C),-1)[(Hs>=3) & (Hs<4)]
    primes[(Hs>=4) & (Hs<5)]=torch.stack((X,Z,C),-1)[(Hs>=4) & (Hs<5)]
    primes[(Hs>=5) & (Hs<6)]=torch.stack((C,Z,X),-1)[(Hs>=5) & (Hs<6)]
    rgb=primes.clone()
    rgb[...,0]+=m
    rgb[...,1]+=m
    rgb[...,2]+=m
    return (rgb*255).int()
    
    
def make_hsv(rgb):
    #we don't need no stinkin alpha channel
    if rgb.shape[-1]==4:
        rgb=rgb[...,:3]
    normalized=(rgb.float()/255)
    minval,_=normalized.min(-1)
    maxval,_=normalized.max(-1)
    delta=maxval-minval
    hue=normalized.new_empty(maxval.shape)
    saturation=normalized.new_empty(maxval.shape)
    nulls=(maxval==0)
    hue[nulls]=0
    saturation[nulls]=0
    saturation[~nulls]=delta[~nulls]/maxval[~nulls]
    
    
    reds=(normalized[...,0]==maxval)&~nulls
    redvals=normalized[reds]
    hue[reds]=((redvals[:,1]-redvals[:,2])/delta[reds])
    
    greens=(normalized[...,1]==maxval)&~nulls
    greenvals=normalized[greens]
    hue[greens]=((greenvals[:,2]-greenvals[:,0])/delta[greens])+2
    
    
    blues=(normalized[...,2]==maxval)&~nulls
    bluevals=normalized[blues]
    hue[blues]=((bluevals[:,0]-bluevals[:,1])/delta[blues])+4
    
    hue[delta==0]=0
    
    return torch.stack((hue*42.5,saturation*255,maxval*255),-1) #scale to 255 space
    
def identify_segment_reduction(neighborhood, values, reducer_function, distance):
    #indexes=torch.arange(values.shape[0]).cuda()
    result=[]
    for index, row in enumerate(neighborhood):
        nearby=(row<distance)
        
        if nearby.any():
            reduced=reducer_function(values[nearby])
            if reduced==values[index]:
                result.append(index)
    return torch.tensor(result).long().cuda()
        

def identify_local_reduction(value_tensor,sample_shape, reducer_function, pad_value):
    assert all([x%2==1 for x in sample_shape]), 'sample shape must be all odd dimensions'
    sample_shape=torch.tensor(sample_shape).int()
    # value_tensor=value_tensor.cuda()
    noisy_tensor=value_tensor+torch.rand(value_tensor.shape).cuda()
    stepsize=1
    mem_usage=noisy_tensor.numel()*sample_shape.prod()//(stepsize**sample_shape.numel())
    while mem_usage>2*10**8:
        stepsize+=1
        mem_usage=noisy_tensor.numel()*sample_shape.prod()//(stepsize**sample_shape.numel())
    
    if DEBUG:
        index=0
        symbols=['bytes','k','m','g','t']
        while mem_usage>1024:
            mem_usage=mem_usage//1024
            index+=1
        symbol=symbols[index]
        print(f'Estimated reducer function GPU memory use: {mem_usage}{symbol}')
    # coordinates=make_coordinate_grid(noisy_tensor.shape)
    # #aranger=make_aranger(sample_shape)
    # aranger=make_coordinate_grid(tuple(sample_shape))#if we pad, we can use the middle as the middle instead of zero
    
    
    # #1=0(no need for padding at all), 3=1 5=2 etc
    padding=(sample_shape//2).unsqueeze(1).repeat(1,2).reshape(-1)
    print('padding is ',padding)
    padded_data=F.pad(noisy_tensor, list(reversed(padding.tolist())), value=pad_value).cuda()

    allshifts=[[]]
    print('shape was', sample_shape)
    for dimension,_ in enumerate(sample_shape):
        
        newshifts=[]
        for step in range(stepsize):
            for existingShift in allshifts:
                newshifts.append(existingShift+[step])
        allshifts=newshifts

    result=noisy_tensor.new_full(noisy_tensor.shape,False, dtype=torch.bool)
    for index,shift in enumerate(allshifts):
        print(f'sample chunking {index} of {len(allshifts)}')    
        reduced=padded_data
        slices=()
        for shiftedDimension, amount in enumerate(shift):
            reduced=reduced.roll(-amount, shiftedDimension)
            slices=slices+(slice(amount, None, stepsize),)
            
        for dim,size in enumerate(sample_shape):
            reduced=reduced.unfold(dim, size, stepsize)
        for dimension in sample_shape:
            reduced=reducer_function(reduced,dim=-1)
        reduced=reduced[shapeToSlices(noisy_tensor[slices].shape)]
            
            
        assert reduced.shape == noisy_tensor[slices].shape, (reduced.shape, noisy_tensor[slices].shape, shift, slices)
        subresult=(reduced==noisy_tensor[slices])  
        result[slices]|=subresult

    assert result.any(), (reduced, noisy_tensor)
    
    return result.detach(), None
    
    
def shapeToSlices(shape):
    return [slice(0,x,None) for x in shape]

    
def make_aranger(sample_shape):
    assert all([x%2==1 for x in sample_shape])
    aranger=make_coordinate_grid(sample_shape)
    median=aranger
    for _ in sample_shape:
        median=median.median(0).values
    aranger=aranger-median
    return aranger
    
    
    
def make_coordinate_grid(shape):
    if hasattr(shape,'shape'):
        shape=shape.shape
    result=torch.zeros(shape+(len(shape),)).long().cuda()
    allSlice=slice(None,None)
    for dimension in range(len(shape)):
        arangement=torch.arange(shape[dimension]).long().cuda()
        slices=[allSlice]*len(shape)+[dimension]
        for higherDimension in range(len(shape)-dimension-1):
            arangement=arangement.unsqueeze(higherDimension+1)
        
        result[slices]=arangement
    return result.cuda()