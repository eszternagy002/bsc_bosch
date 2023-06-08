import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Required constants.
ROOT_DIR = '../input/rendezett_festmenyek_'
VALID_SPLIT = 0.1
#IMAGE_SIZE = 224 # Image size of resize when applying transforms.
BATCH_SIZE = 1 
NUM_WORKERS = 4 # Number of parallel processes for data preparation.

# Training transforms
def get_transform(IMAGE_SIZE, pretrained):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize_transform(pretrained)])
    return train_transform

def mean_and_std():
    norm = open('norm.txt', 'r')
    mean_std = ''
    for row in norm:
        mean_std = row
    norm.close()
    mean_std = mean_std.split(",")
    mean_std.pop(-1)
    for i in range(len(mean_std)):
        mean_std[i] = mean_std[i][:-1]
        mean_std[i] = float(mean_std[i][7:])
    return mean_std

# Image normalization transforms.
def normalize_transform(pretrained):
    ms = mean_and_std()
    if pretrained: # Normalization for pre-trained weights.
        normalize = transforms.Normalize(
            mean=ms[0:3],
            std=ms[3:])
    else: # Normalization when training from scratch.
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])
    return normalize

def get_datasets(IMAGE_SIZE, pretrained, version):
    """
    Function to prepare the Datasets.

    :param pretrained: Boolean, True or False.

    Returns the training and validation datasets along 
    with the class names.
    """
    dataset_train = datasets.ImageFolder(
        ROOT_DIR +version + '/train', 
        transform=(get_transform(IMAGE_SIZE, pretrained)))
    
    dataset_val = datasets.ImageFolder(
        ROOT_DIR + version + '/val', 
        transform=(get_transform(IMAGE_SIZE, pretrained)))

    dataset_test = datasets.ImageFolder(
        ROOT_DIR  + version+ '/test', 
        transform=(get_transform(IMAGE_SIZE, pretrained)))

    return dataset_train, dataset_val, dataset_train.classes

def get_data_loaders(dataset_train, dataset_valid):
    """
    Prepares the training and validation data loaders.

    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.

    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, valid_loader