# Configuration for MorphoNet
batch_size = 32
total_epochs = 20
learning_rate = 0.0001
resize_x = 224
resize_y = 224
input_channels = 3
num_classes = 5

# Class mapping (Simplified to 5-way as per proposal)
CLASSES = ["Smooth", "Spiral", "Edge-on", "Barred", "Merger"]
