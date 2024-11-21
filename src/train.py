import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

def show_augmented_images(dataset, num_images=5):
    """Display original and augmented images side by side"""
    fig, axes = plt.subplots(2, num_images, figsize=(15, 5))
    
    # Get images without augmentation
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    basic_dataset = datasets.MNIST('./data', train=True, download=True, transform=basic_transform)
    
    for i in range(num_images):
        # Get same image from both datasets
        img, _ = basic_dataset[i]
        aug_img, _ = dataset[i]
        
        # Display original
        axes[0, i].imshow(img.squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')
        
        # Display augmented
        axes[1, i].imshow(aug_img.squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Augmented')
    
    plt.tight_layout()
    plt.show()

def train():
    # Set device and seed
    device = torch.device("cpu")
    torch.manual_seed(42)
    
    # Enhanced data augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomAffine(
            degrees=10,  # rotation
            translate=(0.1, 0.1),  # translation
            scale=(0.9, 1.1),  # scale
            shear=5  # shear
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
    ])
    
    # Create dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Show augmented images
    print("Displaying augmented images...")
    show_augmented_images(train_dataset)
    
    # Create dataloader
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model initialization
    model = MNISTModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader)//3, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    best_accuracy = 0.0
    
    # Create directory for augmented samples if it doesn't exist
    os.makedirs('augmented_samples', exist_ok=True)
    
    # Save some augmented batches for visualization
    def save_batch_samples(data, batch_idx):
        if batch_idx in [0, len(train_loader)//2, len(train_loader)-1]:  # Save at start, middle, and end
            fig, axes = plt.subplots(4, 8, figsize=(20, 10))
            for idx, img in enumerate(data[:32]):  # Save first 32 images from batch
                ax = axes[idx//8, idx%8]
                ax.imshow(img.squeeze(), cmap='gray')
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(f'augmented_samples/batch_{batch_idx}.png')
            plt.close()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Save batch samples
        save_batch_samples(data, batch_idx)
        
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
    
    # Load best model
    model.load_state_dict(torch.load('model_best.pth'))
    
    # Final metrics
    final_accuracy = 100 * correct / total
    final_loss = running_loss / len(train_loader)
    print(f'\nFinal Results - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.2f}%')
    
    # Save final model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'model_{timestamp}.pth'
    torch.save(model.state_dict(), save_path)
    print(f'Model saved as {save_path}')
    
    if os.path.exists('model_best.pth'):
        os.remove('model_best.pth')
    
    return save_path

if __name__ == "__main__":
    train() 