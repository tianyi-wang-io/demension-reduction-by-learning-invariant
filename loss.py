import torch.nn as nn
import torch

def get_loss_function():
    """
    Returns the Cross-Entropy loss function for classification.
    """
    return nn.CrossEntropyLoss()

def get_triplet_loss_function(margin=1.0):
    """
    Returns the Triplet Margin loss function for embedding learning.
    """
    # p=2 means use L2 norm for distance
    # margin is the minimum difference required between d(a,p) and d(a,n)
    return nn.TripletMarginLoss(margin=margin, p=2)


# if __name__ == '__main__':
#     # Example usage for Cross-Entropy
#     ce_loss_fn = get_loss_function()
#     print(f"Cross-Entropy Loss function: {ce_loss_fn}")
#     dummy_logits = torch.randn(10, 4) # Batch size 10, 4 classes
#     dummy_targets = torch.randint(0, 4, (10,))
#     ce_loss = ce_loss_fn(dummy_logits, dummy_targets)
#     print(f"Dummy CE loss: {ce_loss.item()}")

#     # Example usage for Triplet Loss
#     triplet_loss_fn = get_triplet_loss_function(margin=1.0)
#     print(f"Triplet Loss function: {triplet_loss_fn}")
#     # Dummy embeddings (e.g., embedding_dim=64, batch_size=10)
#     anchor_embeddings = torch.randn(10, 64)
#     positive_embeddings = torch.randn(10, 64) + 0.1 # Make positives a bit closer
#     negative_embeddings = torch.randn(10, 64) - 0.1 # Make negatives a bit further
#     triplet_loss = triplet_loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
#     print(f"Dummy Triplet loss: {triplet_loss.item()}")