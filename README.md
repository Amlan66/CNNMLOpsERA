# MNIST CNN with MLOps Pipeline

A Convolutional Neural Network (CNN) implementation for MNIST digit classification with MLOps best practices, including automated testing, continuous integration, and model artifact management.

## 🎯 Objectives

- Implement a lightweight CNN model (<25,000 parameters)
- Achieve >95% accuracy on MNIST dataset
- Establish automated testing and CI pipeline
- Implement data augmentation for better generalization
- Use modern optimization techniques
- Maintain code quality and testing standards

## 🏗️ Model Architecture

This project implements a compact Convolutional Neural Network optimized for MNIST digit classification. The architecture is carefully designed to balance model size (<25K parameters) with performance (>95% accuracy).

### Network Structure

The network follows a classic CNN architecture with modern optimizations:

- **Input Layer**
  - Accepts 28x28 grayscale images
  - Normalized input with mean=0.1307, std=0.3081

- **First Convolutional Block**
  - 6 filters of size 3x3 with padding=1
  - Batch Normalization for training stability
  - ReLU activation for non-linearity
  - 2x2 MaxPooling reducing spatial dimensions to 14x14

- **Second Convolutional Block**
  - 12 filters of size 3x3 with padding=1
  - Batch Normalization
  - ReLU activation
  - 2x2 MaxPooling reducing dimensions to 7x7

- **Classifier Block**
  - Flatten layer (12 * 7 * 7 = 588 features)
  - Dense layer with 40 neurons
  - Batch Normalization
  - ReLU activation
  - Final Dense layer with 10 outputs (one per digit)

Parameter Distribution:
- First Conv Block: 60 params
- Second Conv Block: 660 params
- First Dense Layer: 23,560 params
- Output Layer: 410 params
- Total: 24,690 parameters

### Architecture Features
- Input: 28x28 grayscale images
- Two convolutional layers with batch normalization
- MaxPooling for spatial dimension reduction
- ReLU activation functions
- Two fully connected layers
- Total parameters: ~24,690

## 🔧 Training Features

### Data Augmentation
- Random rotation (±10 degrees)
- Random translation (±10%)
- Random scaling (90-110%)
- Random perspective transformation
- Random erasing

### Optimization Strategy
- Optimizer: Adam
- Learning Rate: OneCycleLR scheduler
  - Max LR: 0.01
  - Initial LR: max_lr/25
  - Final LR: initial_lr/1000
  - Cosine annealing
- Batch Size: 32
- Weight Decay: 1e-5
- Gradient Clipping: 1.0

## 🧪 Tests

### Core Tests
1. **Parameter Count Test**
   - Ensures model has <25,000 parameters
   - Validates architecture constraints

2. **Input Shape Test**
   - Verifies input/output tensor dimensions
   - Checks model compatibility with MNIST format

3. **Accuracy Test**
   - Validates model achieves >95% accuracy
   - Tests on MNIST test set

### Advanced Tests
4. **Noise Robustness Test**
   - Adds controlled random noise to inputs
   - Ensures prediction stability >50%
   - Tests model resilience

5. **Activation Range Test**
   - Monitors activation statistics
   - Prevents vanishing/exploding gradients
   - Checks for reasonable value ranges

6. **Layer Properties Test**
   - Validates layer configurations
   - Checks model depth
   - Ensures correct layer dimensions

## 🚀 CI/CD Pipeline

GitHub Actions workflow automatically:
1. Sets up Python environment
2. Installs dependencies
3. Trains model
4. Runs all tests
5. Archives model artifacts and plots

## 📊 Visualization

The training process generates:
- Augmented sample visualizations
- Training metrics plots
  - Learning rate schedule
  - Loss curve
  - Accuracy progression

## 🛠️ Setup and Usage

1. Clone the repository:
```bash
git clone https://github.com/Amlan66/CNNMLOpsERA.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python src/train.py
```

4. Run tests:
```bash
pytest tests/ -v -s
```

## 📦 Dependencies

- PyTorch 2.2.1
- torchvision 0.17.1
- numpy 1.24.3
- matplotlib 3.8.3
- pytest 7.3.1

## 📈 Results

- Model Size: ~24.7K parameters
- Training Accuracy: >95%
- Test Accuracy: >95%
- Noise Stability: >50%

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

</rewritten_file>

