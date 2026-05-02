import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import base64
import numpy as np
from scipy import ndimage

# --- 1. MODEL ARCHITECTURE (7-Layer Betterment) ---
class MorphoNet(nn.Module):
    def __init__(self, num_classes=5):
        super(MorphoNet, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 5: 14 -> 7
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 6: 7 -> 3
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 7: 3 -> 1
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)

# --- 2. HEURISTIC ENGINE ---
def heuristic_merger_check(image_tensor):
    """
    Advanced adaptive override for merger detection.
    image_tensor: (1, 3, 224, 224) normalized tensor
    """
    # Convert to grayscale
    img = image_tensor[0].mean(dim=0).cpu().numpy()
    
    # 1. Adaptive Thresholding
    # Use 98th percentile for peaks, but ensure it's significantly above the median
    p98 = np.percentile(img, 98)
    med = np.median(img)
    max_val = np.max(img)
    # Threshold must be at least halfway between median and max to be a real 'core'
    threshold = max(p98, med + (max_val - med) * 0.5)
    
    binary_img = img > threshold
    
    # 2. Noise Cleanup (Remove single pixels/tiny specks)
    binary_img = ndimage.binary_opening(binary_img, structure=np.ones((2,2)))
    
    # 3. Label distinct blobs
    labeled_array, num_features = ndimage.label(binary_img)
    
    if num_features >= 2:
        slices = ndimage.find_objects(labeled_array)
        valid_blobs = []
        for i, s in enumerate(slices):
            size = np.sum(labeled_array[s] == i+1)
            # Core must be at least 40 pixels to be significant
            if size > 40: 
                center = np.array([(s[0].start + s[0].stop)/2, (s[1].start + s[1].stop)/2])
                valid_blobs.append({'size': size, 'center': center})
        
        # Sort by size (biggest first)
        valid_blobs = sorted(valid_blobs, key=lambda x: x['size'], reverse=True)
        
        if len(valid_blobs) >= 2:
            # Check the interaction between the two primary cores
            c1 = valid_blobs[0]['center']
            c2 = valid_blobs[1]['center']
            dist = np.linalg.norm(c1 - c2)
            
            # Distance must be between 15 and 156 pixels (approx 0.07 to 0.7 of image)
            # This prevents triggering on a single core split by noise
            if (0.07 * 224) < dist < (0.7 * 224):
                return True
    return False

# --- 3. SETTINGS & PREPROCESSING ---
CLASSES = ["Smooth", "Spiral", "Edge-on", "Barred", "Merger"]
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_morphonet(path):
    model = MorphoNet(num_classes=5)
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        # Fix potential DataParallel "module." prefix issue
        if any(k.startswith('module.') for k in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        model.eval()
        return model
    else:
        st.error(f"Brain File Missing! Please place '{path}' in the same folder as this script.")
        return None

# --- UI Helper Functions ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(bin_file):
    bin_str = get_base64_of_bin_file(bin_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .block-container {{
        background: rgba(0, 0, 0, 0.75);
        border-radius: 25px;
        padding: 3rem 2rem 5rem 2rem !important;
        margin-top: 2rem;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    h1, h2, h3, p, span, label {{ color: white !important; }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="MorphoNet", page_icon="🌌", layout="wide")

bg_path = "STScI-01EVSVGGW4CYHJSE3D7PEBM1C6.png"
if os.path.exists(bg_path):
    set_png_as_page_bg(bg_path)

st.title("🌌 MorphoNet: Automated Galaxy Classifier")
st.markdown("Automated classification of galactic morphologies using deep learning.")

st.sidebar.header("Control Panel")
default_weights = "project_aditya_nagrale/checkpoints/final_weights.pth"
weights_file = st.sidebar.text_input("Model Weights (.pth)", default_weights)

# Load Model
model = load_morphonet(weights_file)
if model:
    st.sidebar.success("Model Online")
    uploaded_file = st.file_uploader("Upload Galaxy Observation", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        image = Image.open(uploaded_file).convert("RGB")
        
        with col1:
            st.image(image, caption="Target Galaxy", use_container_width=True)
        
        with st.spinner("Analyzing spectral and spatial features..."):
            img_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                conf, pred = torch.max(probs, 0)
                
                # --- HEURISTIC OVERRIDE ---
                is_merger = heuristic_merger_check(img_tensor)
                if is_merger and conf < 0.85:
                    # Overriding to Class 4 (Merger)
                    final_idx = 4 
                    st.sidebar.info("Heuristic: Dual-core detected. Biasing toward Merger.")
                else:
                    final_idx = pred.item()

        with col2:
            st.subheader("Analysis Breakdown")
            for i, prob in enumerate(probs):
                st.write(f"**{CLASSES[i]}**")
                st.progress(float(prob))
            
            if final_idx != pred.item():
                st.warning(f"**CNN Prediction:** {CLASSES[pred.item()]} ({conf:.2%})")
                st.success(f"**Scientific Override:** {CLASSES[final_idx]} (Detected Dual Cores)")
            else:
                st.success(f"**Final Classification:** {CLASSES[final_idx]} ({conf:.2%})")

st.sidebar.markdown("---")
st.sidebar.caption("Aditya Nagrale | IISER Pune")
