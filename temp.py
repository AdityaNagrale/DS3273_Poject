# =============================================================
# 1. PREPROCESSING: PIVOT TO 5-WAY LABELS
# =============================================================
# Run this on Kaggle to update your dataset to the 5-way structure
import pandas as pd

def update_to_5way_labels(csv_path):
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Original: 0:Smooth, 1:Spiral, 2:Edge-on, 3:Barred, 4:Smooth-Merger, 5:Spiral-Merger
    # New: 0:Smooth, 1:Spiral, 2:Edge-on, 3:Barred, 4:Merger (Both 4 and 5)
    
    old_merger_count = len(df[df['label'].isin([4, 5])])
    df.loc[df['label'] == 5, 'label'] = 4
    new_merger_count = len(df[df['label'] == 4])
    
    print(f"Consolidated {old_merger_count} merger samples into a single Class 4.")
    print("New Label Distribution (5-way):")
    print(df['label'].value_counts().sort_index())
    
    output_path = "gz2_5way_metrics.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

# USE IN KAGGLE: 
# update_to_5way_labels("/kaggle/input/datasets/adityanagrale86/morphonet-data/gz2_scientific_metrics.csv")

# =============================================================
# 2. 5-WAY SPATIALLY PRESERVED ARCHITECTURE (CELL 8)
# =============================================================
import torch
import torch.nn as nn

class MorphoNet(nn.Module):
    def __init__(self, num_classes=5): # SET TO 5
        super(MorphoNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
            
            # Spatial Preservation Blocks (NO MAXPOOL)
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.BatchNorm2d(1024), nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes) # num_classes is now 5
        )
    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)))

# =============================================================
# 3. KAGGLE TRAINING SETUP (5-WAY WEIGHTS)
# =============================================================
# class_counts must be updated for 5 classes
# Smooth: 93156, Spiral: 67863, Edge-on: 19834, Barred: 49324, Merger: 4355 + 5163 = 9518
counts_5way = torch.tensor([93156, 67863, 19834, 49324, 9518], dtype=torch.float)
weights = (1.0 / torch.sqrt(counts_5way))
weights = weights / weights.sum() * 5 # Scale to number of classes
criterion = nn.CrossEntropyLoss(weight=weights)
