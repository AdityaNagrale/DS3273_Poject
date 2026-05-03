# MorphoNet: A Spatially-Preserved CNN for Automated Galaxy Morphology Classification

## Abstract
MorphoNet is a specialized Deep Learning framework designed for the multi-class morphological classification of galaxies using the Galaxy Zoo 2 dataset. By employing a custom seven-layer convolutional architecture that prioritizes spatial feature preservation, the system effectively distinguishes between complex galactic structures. The implementation further integrates a deterministic "Scientific Heuristic" engine to enhance classification reliability in rare merger events, achieving a final validation accuracy of **72.19%**.

---

## 🌌 Classification Taxonomy
The system performs a 5-way classification task across the following morphological categories:
1. **Smooth:** Elliptical systems lacking disk or featured structures.
2. **Spiral:** Featured systems exhibiting distinct disk components and spiral patterns.
3. **Edge-on Disk:** Disk-dominated systems viewed from a lateral orientation.
4. **Barred Spiral:** Spiral galaxies possessing a central linear bar structure.
5. **Merger:** Interacting systems characterized by tidal tails, dual cores, or significant structural distortion.

---

## 🛠️ Technical Methodology

### **1. Spatially-Preserved Convolutional Architecture**
Traditional CNN architectures (e.g., VGG, ResNet) utilize aggressive pooling operations that often deteriorate fine-grained spatial features. MorphoNet addresses this via a **Spatial Preservation Strategy**:
* **Resolution Maintenance:** Max-pooling operations are restricted to the initial four convolutional blocks.
* **Feature Grid:** Blocks 5 through 7 maintain a constant **14×14 spatial resolution**, allowing the network to retain structural integrity for higher-order feature extraction.
* **Global Average Pooling (GAP):** The final 1024-channel feature map is compressed into a 1×1 vector, ensuring translation invariance while preserving the global context learned in earlier layers.

### **2. Adaptive Heuristic Engine (Dual-Core Detection)**
To mitigate the challenges associated with class imbalance and the visual subtlety of galactic mergers, `predict.py` implements a hybrid logic engine:
* **Interacting Peak Analysis:** Utilizing `scipy.ndimage`, the system performs adaptive thresholding and binary opening to isolate high-intensity galactic cores.
* **Proximity Logic:** If multiple significant centroids are detected within a defined interaction radius (0.07–0.7 of image width), the system provides a scientific bias toward the "Merger" class, specifically in cases of CNN predictive uncertainty.

---

## 📈 Empirical Results & Analysis
* **Peak Accuracy:** 72.19% (Consolidated 5-way classification).
* **Baseline Comparison:** Outperformed the standard ResNet-18 benchmark (67.95%) by approximately 4.2%.
* **Interpretability:** Model reasoning was verified through Grad-CAM activation mapping and Max-Projection filter analysis, confirming that the network correctly localizes spiral arms and galactic nuclei.

---

## 📂 Repository Organization
* **`project_aditya_nagrale/`**: Core submission package containing modular implementations for training, inference, and dataset management.
* **`main.py`**: Streamlit-based interactive dashboard for real-time galactic analysis.
* **`temp.py`**: Diagnostic suite for generating scientific audit visualizations and heatmaps.
* **`checkpoints/`**: Optimized weights (`final_weights.pth`) derived from a 30-epoch training cycle on Kaggle.

---

## 🤖 Use of AI
This project was developed in professional collaboration with **Gemini CLI**, an interactive AI agent. The partnership involved:
* **Architectural Engineering:** Co-designing the 7-layer spatially-preserved pipeline to prevent feature squashing.
* **Strategic Optimization:** Pivot from 6-way to 5-way classification and the design of the validation-centric `ReduceLROnPlateau` training logic.
* **Heuristic Design:** Implementation of the deterministic image-processing safety net for merger detection.
* **Technical Auditing:** Systematic debugging and refinement of the cross-module prediction interface.

---
**Lead Developer:** Aditya Nagrale  
**Academic Institution:** IISER Pune  
**Course:** DS3273 - Image and Video Processing with Deep Learning
