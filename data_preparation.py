import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np # Need numpy for data filtering

# CIFAR-10 statistics for normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# Define the original classes we want to keep and their new mappings
TARGET_CLASSES_ORIGINAL = [3, 5, 8, 9] # Original CIFAR-10 labels for cat, dog, ship, truck
NEW_CLASS_MAPPING = {
    3: 0, # cat -> 0
    5: 1, # dog -> 1
    8: 2, # ship -> 2
    9: 3  # truck -> 3
}

# Update the global class names list to reflect the new mapping
CIFAR10_CLASSES = ['cat', 'dog', 'ship', 'truck']
NUM_CLASSES = len(CIFAR10_CLASSES) # The new number of classes

print(f"Using only classes: {CIFAR10_CLASSES}")
print(f"New number of classes: {NUM_CLASSES}")


def filter_and_remap_dataset(dataset):
    """
    Filters the dataset to keep only specified original classes and
    remaps their labels according to NEW_CLASS_MAPPING.
    """
    original_data = dataset.data # numpy array
    original_targets = dataset.targets # list

    # Find indices corresponding to the target original classes
    indices_to_keep = [
        i for i, target in enumerate(original_targets)
        if target in TARGET_CLASSES_ORIGINAL
    ]

    # Filter data and targets
    new_data = original_data[indices_to_keep]
    new_targets = [NEW_CLASS_MAPPING[original_targets[i]] for i in indices_to_keep]

    # Update dataset attributes in place
    dataset.data = new_data
    dataset.targets = new_targets

    print(f"Dataset filtered. New number of samples: {len(dataset)}")
    # Update classes attribute if needed by DataLoader/Dataset (CIFAR10 already has it)
    # dataset.classes = CIFAR10_CLASSES # CIFAR10 uses the original class names, we rely on the remapped targets

    return dataset


def get_cifar10_dataset(data_dir='./data'):
    """
    Prepares, filters, remaps, and returns data loaders for
    a subset of CIFAR-10 training and testing.

    Args:
        batch_size (int): The batch size for the data loaders.
        data_dir (str): The directory to download/load the dataset from.

    Returns:
        tuple: A tuple containing (train_loader, test_loader).
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Transformations for the training set with data augmentation
    train_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Transformations for the test set (no augmentation, just normalization)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Load the training and test datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # --- Filter and Remap Datasets ---
    print("Filtering and remapping training dataset...")
    train_dataset = filter_and_remap_dataset(train_dataset)

    print("Filtering and remapping testing dataset...")
    test_dataset = filter_and_remap_dataset(test_dataset)
    # --- End Filtering ---
    print(f"Using only classes: {CIFAR10_CLASSES}")
    print(f"New number of classes: {NUM_CLASSES}")

    return train_dataset, test_dataset


def get_dataloader(train_dataset, test_dataset, batch_size=128, num_worders=4):
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worders
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worders
    )

    print(f"Filtered CIFAR-10 dataset loaders created. train_dataset:{len(train_dataset)} test_dataset: {len(test_dataset)}.")

    return train_loader, test_loader