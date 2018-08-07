# Deep_Learning_Practice
The purpose of this repository is to practice deep-learning and reinforcement learning. 

## 1. [Concept Notes.ipynb](https://github.com/shiwang0211/deep_learning_practice/blob/master/Concepts_Notes.ipynb)
- Key concepts in NN
- CNN (different architectures for CNN)
- RNN (from RNN to LSTM/GRU)


## 2. Neural_Network_Scratch - Part I.ipynb
- Build neural network from scratch using Numpy without using Tensorflow or Keras. 
- The notebook includes the forward and backward propagation of different layers written in Numpy
  - [X] Basic Activation Functions
  - [X] Fully-connected Layer 
  - [X] Different Update/Optimization Methods
  - [X] Batch Normalization Layer
  - [X] Drop-out Layer
  - [X] Convolutional layer
  - [ ] Recurrent Layer

## 3. Neural_Network_Scratch - Part II.ipynb
- This notebook includes different applications built from components in Part I (e.g., CNN)
- Dataset
  - Synthetic
  - MNIST (2d, 3d)
  - IRIS
- Gradient check and result evaulation

## 4. Tensorflow Practice.ipynb
Practice with Tensorflow:
- Eager Execution
- Estimator API
    - Built-in classifier
    - Customized classifier
- Dataset API
- Keras with TF backend
    - Sequential model
    - Model class with functional API (LSTM example)

## 5. CNN RNN Pratice.ipynb
This notebook includes some practice examples to apply tensorflow to build DL models.
- Example 1:
  - Methodology: CNN 
  - Tool: Tensorflow Estimator API
  - Dataset: MNIST

- Example 2: 
  - Methodology: RNN (LSTM)
  - Tool: Tensorflow Estimator API
  - Dataset: Synthetic
  
- Example 3: 
  - Methodology: RNN (LSTM)
  - Tool: Tensorflow Lower-level API
  - Dataset: Synthetic

## 6. GCP_CPU.ipynb
A comparison of computational performance between CPU and GPU based on a test run on Google Cloud Platform (GCP)

## 7. Generative Adversarial Networks (GAN).ipynb
- Concept Notes
- An example based on MNIST dataset to generate fake digits

