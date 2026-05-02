# MorphoNet: Automated Galaxy Morphology Classification

This repository contains the **MorphoNet** project—a Deep Learning system designed to automatically classify galaxy images from the Galaxy Zoo 2 dataset into six morphological categories.

## 🌌 Project Overview
Galaxy morphology classification is essential for understanding the evolution of the universe. With millions of images captured by modern telescopes, manual classification is no longer feasible. MorphoNet provides a high-performance, automated solution using a custom Convolutional Neural Network (CNN) architecture.

### **Classification Task (5-Way)**
1. **Smooth** (Elliptical)
2. **Spiral** (Featured/Disk)
3. **Edge-on Disk**
4. **Barred Spiral**
5. **Merger** (Consolidated)

---

## 🛠️ Technical Implementation

### **1. Custom 7-Layer Architecture**
Unlike standard models that use aggressive max-pooling (reducing spatial detail), MorphoNet uses a **Spatial Preservation Strategy**:
* **Depth:** 7 Convolutional Blocks (scaling from 32 to 1024 filters).
* **Resolution:** Max-pooling is stopped after Layer 4, maintaining a **14x14 spatial grid** for high-level features.
* **Global Average Pooling (GAP):** Summarizes spatial features at the very end to prevent overfitting and ensure translation invariance.

### **2. Adaptive Scientific Heuristic**
To address the rarity of galactic mergers, the `predict.py` module includes an **Adaptive Heuristic engine** using `scipy.ndimage`:
* **Dual-Core Detection:** Scans images for interacting bright cores.
* **Proximity Logic:** Automatically biases the classification toward "Merger" if two significant, proximate cores are identified, providing a scientific safety net for the CNN.

---

## 📈 Results
* **Final Validation Accuracy:** **72.19%** (Epoch 16)
* **Performance:** Surpassed standard ResNet-18 baseline (67.95%) on the same dataset.
* **Feature Localization:** Visualized using "X-Ray" activation mapping, proving the model correctly identifies spiral arms and galactic cores.

---

## 📂 Repository Structure
* **`project_aditya_nagrale/`**: Official submission folder.
    * `model.py`: MorphoNet architecture.
    * `predict.py`: Hybrid CNN + Heuristic predictor.
    * `interface.py`: Standardized grading interface.
    * `checkpoints/`: Optimized model weights (`final_weights.pth`).
    * `data/`: 60 sample images for verification.
* **`main.py`**: Streamlit-based interactive UI.
* **`temp.py`**: Model visualization and diagnostic toolkit.

## 🚀 How to Run
1. Install dependencies: `pip install torch torchvision streamlit scipy matplotlib pillow pandas`
2. Run the interactive UI: `streamlit run main.py`
3. For grading: Use `project_aditya_nagrale/interface.py` to call `the_predictor`.

---
**Developed by:** Aditya Nagrale  
**Course:** DS3273 - Image and Video Processing with Deep Learning  
**Institution:** IISER Pune
