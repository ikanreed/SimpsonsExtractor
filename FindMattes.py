from PIL import Image
import torch
import torch.cuda
import torchvision.transforms as T
from torchvision import models
import numpy as np

fcn = None

def getRotoModel():
    global fcn
    fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
    if torch.cuda.is_available():
        print('enable cuda')
        fcn.to('cuda')

# Define the helper function
def decode_segmap(image, nc=21):

    label_colors = np.array([(0, 0, 0),  # 0=background
                           # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor(disabled)
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 0, 0)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb
def decode_certainty(image, squeeze,maxval,minval):
   r=np.zeros_like(image).astype(np.uint8)
   idx= image!=40
   r[idx]=(squeeze[idx]-minval)*255/(maxval-minval)
   return np.stack([r,r,r], axis=2)


def createMatteFrame(frame, target_type):
    trf=T.Compose([T.Normalize(mean = [0.485, 0.456, 0.406], 
                                 std = [0.229, 0.224, 0.225])])
    frame=torch.stack([frame[...,0],frame[...,1],frame[...,2]],dim=0)/255.0
    
    normalized=trf(frame)
    input=normalized.unsqueeze(0).cuda()
    if (fcn == None): getRotoModel()
    output=fcn(input)['out'].squeeze()
    output[target_type]*=2
    return torch.argmax(output,dim=0)==target_type

def createMatte(filename, matteName, size):
    img = Image.open(filename)
    im=createMatteImg(img,size)
    im.save(matteName)
def createMatteImg(img, size, amplifyFactor=1,selectionFactors={}, startOver=False, target=15):
    global previousSqueeze
    trf = T.Compose([T.Resize(size),
                     T.ToTensor(), 
                     #T.Normalize(mean = [0.485, 0.456, 0.406], 
                     #            std = [0.229, 0.224, 0.225])])
                     T.Normalize(mean = [0.485, 0.456, 0.406], 
                                 std = [0.229, 0.224, 0.225])])
    if startOver:
        previousSqueeze=None
    inp = trf(img).unsqueeze(0)
    if torch.cuda.is_available():
        inp=inp.to('cuda')
    if (fcn == None): getRotoModel()
    out = fcn(inp)['out']
    squeeze=out.squeeze()
    #squeeze[15]=squeeze[15]*2
    
    squeeze[0]=squeeze[0]/amplifyFactor #let's really crush the candidate data down to nothing
    
    
    
    for key in selectionFactors:
        squeeze[key]=squeeze[key]*selectionFactors[key]
    
    #print(squeeze[15].size())
    predetached=torch.argmax(squeeze,dim=0)
    om = predetached.detach().cpu().numpy()
    rgb = decode_segmap(om)
    
    #print('max val', torch.max(squeeze[15]))
    #rgb=decode_certainty(om, squeeze[15].detach().cpu().numpy(), torch.max(squeeze[15]).detach().cpu().numpy(), torch.min(squeeze[15]).detach().cpu().numpy())
    
    
    im = Image.fromarray(rgb)
    
    return im
    