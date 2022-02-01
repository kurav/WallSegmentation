import os, torch, PIL, torchvision.transforms
import numpy as np
import cv2
# Function for visualizing wall prediction (original image, segmentation mask and original image with the segmented wall)
def visualize_wall(img, pred):
    img_green = img.copy()
    black_white = img.copy()
    img_green[pred == 0] = [6,255,115]
    black_white[pred == 0] = [6,255,115]
    black_white[pred != 0] = [0,0,0]


    result = cv2.addWeighted(img_green, 0.6, img, 0.4, 0, img_green)
    result1 = cv2.resize(result,(400,300))
    cv2.imshow('result',result1)
    cv2.waitKey(0)
    

# Function for segmenting wall in the input image
def segment_image( segmentation_module, img ):
    
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.495, 0.505, 0.486], # These are RGB mean+std values
            std=[0.229, 0.220, 0.225])  # across a large photo dataset.
        ])
    
    img_original = np.array(img)
    img_data = pil_to_tensor(img)
    singleton_batch = {'img_data': img_data[None]}#.cuda()
    segSize = img_original.shape[:2]
    
    with torch.no_grad():
        scores = segmentation_module( singleton_batch, segSize = segSize )
    _, pred = torch.max( scores, dim = 1 )
    pred = pred.cpu()[0].numpy()
    
    visualize_wall(img_original, pred)

def find(p):
    distances = np.zeros(p.shape[1], dtype=np.uint32)
    nothing=True
    for col in range(p.shape[1]):
        if p[0][col]==1:
            for row in range(p.shape[0]):
                if p[row][col]==0:
                    distances[col]=row-1
                    nothing=False
                    break

    # print(np.argmax(distances), np.max(distances), nothing)
    if np.max(distances) != 0:
        all_max_distance_idx = np.argwhere(distances == np.amax(distances))
        if len(all_max_distance_idx)>1:
            if distances[0]<distances[-1]:
                return np.max(distances), all_max_distance_idx.flat[0]
            else:
                return np.max(distances), all_max_distance_idx.flat[-1]
        else:
            return np.max(distances), np.argmax(distances)
    return None
