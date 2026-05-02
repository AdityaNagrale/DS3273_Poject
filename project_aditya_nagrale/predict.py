import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy import ndimage
import os
from .model import MorphoNet
from .config import resize_x, resize_y, CLASSES

def detect_merger_heuristic(img_path):
    """
    Advanced adaptive override for merger detection.
    Detects interacting cores based on brightness peaks and proximity.
    """
    try:
        # Load image and convert to grayscale
        img_raw = Image.open(img_path).convert('RGB')
        # We simulate the normalization to match the CNN's view
        img_arr = np.array(img_raw).mean(axis=2) / 255.0
        
        # 1. Adaptive Thresholding
        p98 = np.percentile(img_arr, 98)
        med = np.median(img_arr)
        max_val = np.max(img_arr)
        # Threshold must be at least halfway between median and max
        threshold = max(p98, med + (max_val - med) * 0.5)
        
        binary_img = img_arr > threshold
        
        # 2. Noise Cleanup (Remove single pixels/tiny specks)
        binary_img = ndimage.binary_opening(binary_img, structure=np.ones((2,2)))
        
        # 3. Label distinct blobs
        labeled_array, num_features = ndimage.label(binary_img)
        
        if num_features >= 2:
            slices = ndimage.find_objects(labeled_array)
            valid_blobs = []
            for i, s in enumerate(slices):
                size = np.sum(labeled_array[s] == i+1)
                # Core must be at least 40 pixels (in a 224x224 context)
                if size > 40: 
                    center = np.array([(s[0].start + s[0].stop)/2, (s[1].start + s[1].stop)/2])
                    valid_blobs.append({'size': size, 'center': center})
            
            # Sort by size (biggest first)
            valid_blobs = sorted(valid_blobs, key=lambda x: x['size'], reverse=True)
            
            if len(valid_blobs) >= 2:
                c1 = valid_blobs[0]['center']
                c2 = valid_blobs[1]['center']
                # Calculate distance in normalized 224x224 space
                # (ndimage uses original image size, so we scale it)
                h, w = img_arr.shape
                dist_x = (c1[0] - c2[0]) * (224.0 / w)
                dist_y = (c1[1] - c2[1]) * (224.0 / h)
                dist = np.sqrt(dist_x**2 + dist_y**2)
                
                # Distance must be between 15 and 156 pixels (approx 0.07 to 0.7 of 224)
                if 15 < dist < 156:
                    return True
    except Exception as e:
        pass # Fallback to CNN only
    return False

def classify_galaxies(list_of_img_paths, model_path=None):
    if model_path is None:
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'checkpoints', 'final_weights.pth')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = MorphoNet(num_classes=len(CLASSES))
    try:
        state_dict = torch.load(model_path, map_location=device)
        # Fix DataParallel "module." prefix
        if any(k.startswith('module.') for k in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Model Load Error: {e}")
        
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = []
    with torch.no_grad():
        for path in list_of_img_paths:
            # 1. CNN Prediction
            try:
                img = Image.open(path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)
                
                # 2. Apply Adaptive Heuristic
                is_merger = detect_merger_heuristic(path)
                
                # Override only if CNN is very unsure (<60%) and heuristic sees two cores
                if is_merger and conf.item() < 0.60:
                    # Generic Merger capture (Class 4)
                    pred_idx = 4
                
                results.append(CLASSES[pred_idx])
            except:
                results.append(CLASSES[0]) # Fallback
                
    return results
