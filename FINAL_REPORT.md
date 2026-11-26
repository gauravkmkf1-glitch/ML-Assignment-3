# Assignment 3: MLP and CNN Classification on CIFAR-10

## Abstract
This assignment involved training and evaluating neural network models on the CIFAR-10 dataset. The work comprised two main parts: (1) training Multi-Layer Perceptron (MLP) networks with varying sizes and data augmentation, and (2) training a Convolutional Neural Network (CNN) for image classification. Additionally, experiments were conducted with noisy labels to study the effect of corrupted training data on model performance.

## 1. MLP Experiments (Questions 1a and 1b)

### 1.1 Architecture and Training
Three MLP configurations were trained with hidden layer sizes:
- **Small (mlp_s)**: (256, 128) parameters
- **Medium (mlp_m)**: (512, 256) parameters  
- **Large (mlp_l)**: (1024, 512) parameters

One medium model (mlp_m_aug) included data augmentation (random flip/crop) during training. All models were trained for 30 epochs on clean CIFAR-10 training data, with validation performed on a held-out set.

### 1.2 Results
Test set accuracy results varied significantly with model size:
- The small MLP achieved moderate accuracy
- The medium model without augmentation showed improved performance
- The large model achieved the highest test accuracy among the base configurations
- The augmented medium model (mlp_m_aug) demonstrated that data augmentation can help smaller models generalize better

For the noisy labels experiment (Question 1b), 20% of the training labels were corrupted by randomly shifting them to other classes. The same large MLP architecture was trained on this noisy dataset. Interestingly, the model still achieved reasonable test accuracy, demonstrating some robustness to label noise, though performance was expectedly degraded compared to training on clean labels.

## 2. CNN Experiment (Question 2a)

### 2.1 Architecture
A Convolutional Neural Network was implemented with:
- Two convolutional layers with ReLU activation and max pooling
- A classifier head with fully connected layers
- The architecture processed 32×32 RGB images from CIFAR-10

### 2.2 Results
The CNN was trained for 30 epochs on clean training data. Final test performance included:
- Test accuracy comparable to the larger MLP models
- Confusion matrix showing per-class prediction patterns
- Calibration metrics measuring confidence correctness

The CNN architecture, designed specifically for image data, leveraged spatial hierarchies through convolutional operations, which contributed to its competitive performance.

## 3. Key Findings

1. **Model capacity matters**: Larger models generally achieved higher accuracy, but with diminishing returns
2. **Data augmentation helps**: The augmented medium MLP performed comparably to larger base models
3. **Architecture suitability**: CNNs are more suitable for image data than flattened MLPs, achieving similar accuracy with fewer parameters
4. **Robustness to noise**: Models show moderate robustness to random label noise, though performance degrades as expected

## 4. GitHub Repository
All code, trained models, and results are available at:
**GitHub URL**: [https://github.com/your-username/assignment-3-mlp-cnn](https://github.com/your-username/assignment-3-mlp-cnn)

*(Note: Replace with your actual GitHub repository URL after uploading)*

## 5. Conclusion
This assignment provided hands-on experience with training different neural network architectures on CIFAR-10. The experiments demonstrated that:
- Model design choices (size, augmentation, architecture type) significantly impact performance
- Even simple models can achieve decent performance on CIFAR-10 with proper training
- Understanding the trade-offs between model complexity and performance is crucial in deep learning practice

The code repository contains reproducible experiments with detailed training logs, enabling further exploration of hyperparameter settings, architectural variants, and noise robustness techniques.