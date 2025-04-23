import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np

# Import tqdm
from tqdm import tqdm

# Import modules from our project
# Import get_cifar10_dataloaders, NUM_CLASSES, CIFAR10_CLASSES
from data_preparation import NUM_CLASSES, CIFAR10_CLASSES
from model import InvariantMappingModel, build_encoder
from loss import get_loss_function


# --- Hyperparameters ---
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 20  # You can adjust this
EMBEDDING_DIM = 64  # The dimensionality of the reduced space
# NUM_CLASSES is imported from data_preparation

# *** Choose your encoder architecture here ***
# ENCODER_ARCHITECTURE = 'cnn' # or 'resnet18'


def train_model(
    train_loader,
    test_loader,
    encoder_model,
    embedding_dim,
    num_classes,
    learning_rate,
    num_epochs,
):
    """
    Handles the training process, evaluation, plotting, and saving.
    Adds tqdm progress bars.
    """
    # Device configuration
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # --- Output directory ---
    OUTPUT_DIR = "output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Include encoder name and num_classes in model save path
    MODEL_SAVE_PATH = os.path.join(
        OUTPUT_DIR, f"{encoder_model}_{NUM_CLASSES}cls_{embedding_dim}dim_invariant_mapping_model.pth"
    )

    print(
        f"LEARNING_RATE: {learning_rate}, \nNUM_EPOCHS: {num_epochs} \nEMBEDDING_DIM: {embedding_dim} \nNUM_CLASSES: {num_classes}"
    )
    print(f"Encoder Architecture: {encoder_model}")

    # 2. Model, Loss, and Optimizer
    encoder = build_encoder(encoder_model, embedding_dim)
    model = InvariantMappingModel(encoder, num_classes).to(device)
    criterion = get_loss_function()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store training history for plotting
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    print("Starting training...")

    # 3. Training Loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_train_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Wrap train_loader with tqdm
        train_loop = tqdm(
            train_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
        )

        for i, (images, labels) in enumerate(train_loop):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Calculate accuracy for the batch
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

            # Update tqdm description with current average loss per sample
            # Use train_loop.n (total samples processed) as denominator
            if train_loop.n > 0:  # Safety check for division by zero
                avg_batch_loss = running_train_loss / train_loop.n
                train_loop.set_postfix(loss=avg_batch_loss)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        epoch_train_accuracy = 100 * correct_predictions / total_predictions
        train_accuracies.append(epoch_train_accuracy)

        # 4. Evaluation Loop
        model.eval()  # Set model to evaluation mode
        running_test_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Wrap test_loader with tqdm
        # tqdm(dataloader) iterates over batches, .n tracks samples
        test_loop = tqdm(
            test_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Test ]"
        )

        with torch.no_grad():  # Disable gradient calculation during evaluation
            for images, labels in test_loop:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_test_loss += loss.item() * images.size(0)

                # Calculate accuracy for the batch
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                # Update tqdm description with current average test loss per sample
                # Use test_loop.n (total samples processed) as denominator
                if test_loop.n > 0:  # Safety check for division by zero
                    avg_test_batch_loss = running_test_loss / test_loop.n
                    test_loop.set_postfix(
                        loss=avg_test_batch_loss
                    )  # Accuracy will be calculated per epoch

        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        epoch_test_accuracy = 100 * correct_predictions / total_predictions
        test_accuracies.append(epoch_test_accuracy)

        # Print final epoch summary below the progress bars
        print(
            f"Epoch [{epoch+1}/{num_epochs}] Summary: "  # Correct variable names here
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Test Loss: {epoch_test_loss:.4f}, "
            f"Test Accuracy: {epoch_test_accuracy:.2f}%"
        )

    print("Training finished.")

    # 5. Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(test_losses, label="Test Loss", color="green")
    plt.plot(train_losses, label="Train Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{encoder_model.upper()} Training and Test Loss ({NUM_CLASSES} Classes)")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{encoder_model}_{NUM_CLASSES}cls_loss_plot.png")
    )
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(test_accuracies, label="Test Accuracy", color="green")
    plt.plot(train_accuracies, label="Train Accuracy", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{encoder_model.upper()} Training and Test Accuracy ({NUM_CLASSES} Classes)")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{encoder_model}_{NUM_CLASSES}cls_accuracy_plot.png")
    )
    plt.close()

    print(f"Plots saved to '{OUTPUT_DIR}' directory.")

    # 6. Saving the Model Parameters
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model parameters saved to '{MODEL_SAVE_PATH}'")
