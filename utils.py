import torch
import numpy as np
import math
import random

def get_train_val_test_size(total_size, train_val_test_ratio):
    train_scale, validation_scale, test_scale = train_val_test_ratio 
    training_size = math.ceil(total_size*(train_scale/(train_scale + validation_scale + test_scale)))
    validation_size = math.floor(total_size*(validation_scale/(train_scale+ validation_scale+ test_scale)))
    return training_size, validation_size, total_size - training_size - validation_size

def set_seed(random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

