# COMP5313 Artificial Intelligence
## Department of Computer Science
### MLP for Image Classification: Connectionist AI

## Introduction

This repository contains implementations of different Multilayer Perceptron (MLP) models for classifying handwritten digits from the MNIST dataset. The models are implemented using various frameworks and libraries, including TensorFlow/Keras, NumPy, and PyTorch.

### 1. Simple Three-Layer MLP

#### a. With Keras
The code implements a simple MLP using TensorFlow and Keras, achieving an accuracy of 98.4% on the MNIST dataset.

#### b. Without Keras
The second code implements a three-layer MLP using NumPy, achieving an accuracy of 87.29% on the MNIST dataset.

### 2. ResMLP
The code implements a Residual MLP model using PyTorch, achieving an accuracy of 97.43% on the MNIST dataset.

## Comparison

The performance of each model is compared based on their accuracy and implementation details:

- **Simple Three-layer MLP with Keras:** Achieved the highest accuracy of 98.4%. Utilizes Keras for simplicity but may lack flexibility.
- **Simple Three-layer MLP without Keras (NumPy-based):** Achieved an accuracy of 87.29%. Provides more control but requires more manual coding.
- **Residual Multilayer Perceptron (ResMLP) with PyTorch:** Achieved an accuracy of 97.43%. Offers a balance between abstraction and control.

## Conclusion

- For simplicity and high-level abstraction, the Simple Three-Layer MLP with Keras is recommended, especially with its impressive accuracy.
- For more control over the model and lower-level implementation, the NumPy-based approach is an option.
- The ResMLP with PyTorch offers a balance between abstraction and control, achieving good accuracy with the benefits of PyTorch's features.

## References

1. [MLP - Multilayer Perceptron: Simple Overview](https://www.analyticsvidhya.com/blog/2020/12/mlp-multilayer-perceptron-simple-overview/)
2. [Keras Documentation: MLP Image Classification Example](https://keras.io/examples/vision/mlp_image_classification/)
3. [Review: ResMLP — Feedforward Networks for Image Classification with Data-Efficient Training](https://sh-tsang.medium.com/review-resmlp-feedforward-networks-for-image-classification-with-data-efficient-training-4eeb1eb5efa6)
4. Lecture slides of Dr. Sabah Mohammad

---

**Author:** Chandana Sree Krishna (Student ID: 1226847)  
**Date:** 07 February 2024
