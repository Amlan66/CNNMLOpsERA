import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
from datetime import datetime
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def show_augmented_images(dataset, save_path='augmented_samples'):
    """Display and save original vs augmented images"""
    os.makedirs(save_path, exist_ok=True)
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # Get images without augmentation
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    basic_dataset = datasets.MNIST('./data', train=True, download=True, transform=basic_transform)
    
    for i in range(5):
        # Get same image from both datasets using fixed index for reproducibility
        img, _ = basic_dataset[i]
        aug_img, _ = dataset[i]
        
        # Display original
        axes[0, i].imshow(img.squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Original {i+1}')
        
        # Display augmented
        axes[1, i].imshow(aug_img.squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Augmented {i+1}')
    
    plt.suptitle('Original vs Augmented MNIST Images', y=1.02, fontsize=12)
    plt.tight_layout()
    
    # Save the comparison plot
    comparison_path = os.path.join(save_path, 'augmentation_comparison.png')
    plt.savefig(comparison_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Augmented images comparison saved to: {comparison_path}")
    
    # Save individual augmented samples
    for i in range(10):  # Save 10 individual samples
        aug_img, label = dataset[i]
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(aug_img.squeeze(), cmap='gray')
        ax.axis('off')
        ax.set_title(f'Sample {i+1} (Label: {label})')
        sample_path = os.path.join(save_path, f'sample_{i+1}.png')
        plt.savefig(sample_path, bbox_inches='tight', dpi=300)
        plt.close()

def train():
    # Set device and seed
    device = torch.device("cpu")
    torch.manual_seed(42)
    
    # Create output directories
    os.makedirs('augmented_samples', exist_ok=True)
    os.makedirs('training_plots', exist_ok=True)
    
    # Data augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
    ])
    
    # Create dataset and show augmented images
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    print("\nGenerating augmented images...")
    show_augmented_images(train_dataset)
    
    # Rest of your training code remains the same...
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model initialization
    model = MNISTModel().to(device)
    
    # Calculate total steps for OneCycleLR
    total_steps = len(train_loader)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5, betas=(0.9, 0.99))
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        steps_per_epoch=total_steps,
        epochs=1,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000,
        anneal_strategy='cos'
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training metrics storage
    lrs = []
    losses = []
    accuracies = []
    
    # Training loop
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    best_accuracy = 0.0
    
    print("\nStarting training...")
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        running_loss += loss.item()
        
        # Store metrics
        lrs.append(scheduler.get_last_lr()[0])
        losses.append(loss.item())
        accuracies.append(100 * correct / total)
        
        if batch_idx % 100 == 0:
            accuracy = 100 * correct / total
            avg_loss = running_loss / (batch_idx + 1)
            print(f'Batch [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {avg_loss:.4f} '
                  f'Accuracy: {accuracy:.2f}% '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), 'model_best.pth')
    
    # Plot and save training metrics
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(lrs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Batch')
    plt.ylabel('Learning Rate')
    
    plt.subplot(1, 3, 2)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_plots/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Load best model and save final results
    model.load_state_dict(torch.load('model_best.pth'))
    
    final_accuracy = 100 * correct / total
    final_loss = running_loss / len(train_loader)
    print(f'\nFinal Results - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.2f}%')
    print(f'Best Accuracy: {best_accuracy:.2f}%')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'model_{timestamp}.pth'
    torch.save(model.state_dict(), save_path)
    
    print(f'\nSaved files:')
    print(f'- Model: {save_path}')
    print(f'- Training metrics: training_plots/training_metrics.png')
    print(f'- Augmented samples: augmented_samples/')
    
    if os.path.exists('model_best.pth'):
        os.remove('model_best.pth')
    
    return save_path

if __name__ == "__main__":
    train() 