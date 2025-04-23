import torch.nn as nn

def get_loss_function():
    """
    Returns the loss function for training the model.
    We use Cross-Entropy Loss for classification.
    """
    return nn.CrossEntropyLoss()
