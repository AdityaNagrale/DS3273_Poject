# Standardized Interface for Grading
from .model import MorphoNet as TheModel
from .train import train_model as the_trainer
from .predict import classify_galaxies as the_predictor
from .dataset import GalaxyDataset as TheDataset
from .dataset import get_dataloader as the_dataloader
from .config import batch_size as the_batch_size
from .config import total_epochs as total_epochs
