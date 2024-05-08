# -*- coding: utf-8 -*-
"""ChandanaSreeKrishna_RL.ipynb
1. a) Simple three layer MLP - As per tutorial
"""

import numpy as np
from tensorflow.keras.datasets import mnist

# The mnist.load_data() method is convenient, as there is no need to load all 70,000
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# The labels are in the form of digits, from 0 to 9.
num_labels = len(np.unique(y_train))
print("total labels:{}".format(num_labels))
print("labels:{0}".format(np.unique(y_train)))

# The most suitable format is one-hot, a 10-dimensional vector-like all 0 values, except the class index.
#converter em one-hot
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
input_size = image_size * image_size

# Our model is an MLP, so inputs must be a 1D tensor. as such, x_train and x_test must be transformed into [60,000, 2828] and [10,000, 2828],
print("x_train:t{}".format(x_train.shape))
print("x_test:tt{}n".format(x_test.shape))

x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype('float32') / 255

x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255

print("x_train:t{}".format(x_train.shape))
print("x_test:tt{}".format(x_test.shape))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

# Parameters
batch_size = 128 # It is the sample size of inputs to be processed at each training stage.
hidden_units = 256
dropout = 0.45

# Nossa  MLP com ReLU e Dropout
model = Sequential()

model.add(Dense(hidden_units, input_dim=input_size))
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(num_labels))

model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=batch_size)

_, acc = model.evaluate(x_test,
                        y_test,
                        batch_size=batch_size,
                        verbose=0)
print("nAccuracy: %.1f%%n" % (100.0 * acc))

"""1.b) An alternate implementation of the MLP without using Keras, but instead using NumPy for building and training the model. It follows a similar architecture as the Keras model as per the tutorial mentioned in the assignment."""

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
num_labels = 10
input_size = 28 * 28

x_train = x_train.reshape(-1, input_size).astype('float32') / 255.0
x_test = x_test.reshape(-1, input_size).astype('float32') / 255.0

y_train = to_categorical(y_train, num_labels)
y_test = to_categorical(y_test, num_labels)

# Parameters
batch_size = 128
hidden_units = 256
dropout = 0.45
learning_rate = 0.001
epochs = 20

# Initialize weights and biases
'''
weights1 = np.random.randn(input_size, hidden_units)
bias1 = np.zeros((1, hidden_units))

weights2 = np.random.randn(hidden_units, hidden_units)
bias2 = np.zeros((1, hidden_units))

weights3 = np.random.randn(hidden_units, num_labels)
bias3 = np.zeros((1, num_labels))
'''
# Activation function (ReLU)
def relu(x):
    return np.maximum(0, x)

# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Xavier/Glorot initialization
def initialize_weights(input_size, output_size):
    limit = np.sqrt(6 / (input_size + output_size))
    return np.random.uniform(-limit, limit, size=(input_size, output_size))

# Initialize weights and biases using Xavier/Glorot initialization
weights1 = initialize_weights(input_size, hidden_units)
bias1 = np.zeros((1, hidden_units))

weights2 = initialize_weights(hidden_units, hidden_units)
bias2 = np.zeros((1, hidden_units))

weights3 = initialize_weights(hidden_units, num_labels)
bias3 = np.zeros((1, num_labels))

# Training loop
for epoch in range(epochs):
    # Shuffle the training data
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    # Mini-batch training
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        # Forward pass
        hidden_layer_input = np.dot(x_batch, weights1) + bias1
        hidden_layer_output = relu(hidden_layer_input)
        hidden_layer_output = hidden_layer_output * (1 - dropout)  # Apply dropout

        hidden_layer_input2 = np.dot(hidden_layer_output, weights2) + bias2
        hidden_layer_output2 = relu(hidden_layer_input2)
        hidden_layer_output2 = hidden_layer_output2 * (1 - dropout)  # Apply dropout

        output_layer_input = np.dot(hidden_layer_output2, weights3) + bias3
        predicted_output = softmax(output_layer_input)

        # Loss calculation (cross-entropy)
        loss = -np.sum(y_batch * np.log(predicted_output)) / len(x_batch)

        # Backpropagation
        output_error = predicted_output - y_batch
        hidden_error2 = output_error.dot(weights3.T) * (hidden_layer_output2 > 0)
        hidden_error = hidden_error2.dot(weights2.T) * (hidden_layer_output > 0)

        # Update weights and biases
        weights3 -= learning_rate * hidden_layer_output2.T.dot(output_error) / len(x_batch)
        bias3 -= learning_rate * np.sum(output_error, axis=0, keepdims=True) / len(x_batch)

        weights2 -= learning_rate * hidden_layer_output.T.dot(hidden_error2) / len(x_batch)
        bias2 -= learning_rate * np.sum(hidden_error2, axis=0, keepdims=True) / len(x_batch)

        weights1 -= learning_rate * x_batch.T.dot(hidden_error) / len(x_batch)
        bias1 -= learning_rate * np.sum(hidden_error, axis=0, keepdims=True) / len(x_batch)

    if epoch % 1 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')

# Testing the trained model
hidden_layer_input = np.dot(x_test, weights1) + bias1
hidden_layer_output = relu(hidden_layer_input)

hidden_layer_input2 = np.dot(hidden_layer_output, weights2) + bias2
hidden_layer_output2 = relu(hidden_layer_input2)

output_layer_input = np.dot(hidden_layer_output2, weights3) + bias3
predicted_output = softmax(output_layer_input)

# Accuracy calculation
accuracy = np.mean(np.argmax(predicted_output, axis=1) == np.argmax(y_test, axis=1))
print(f'Test Accuracy: {accuracy * 100:.2f}%')

pip install torch torchvision

"""5. ResMLP Model for MNIST Dataset

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define the ResMLP model
class ResMLPBlock(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=None):
        super(ResMLPBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_features or in_features),
            nn.GELU(),
            nn.Linear(hidden_features or in_features, out_features),
        )

    def forward(self, x):
        return x + self.block(x)

class ResMLP(nn.Module):
    def __init__(self, input_size, num_classes, num_blocks=6, hidden_size=256):
        super(ResMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            *[ResMLPBlock(hidden_size, hidden_size) for _ in range(num_blocks)],
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
input_size = 28 * 28  # MNIST image size
num_classes = 10  # Number of classes (digits 0-9)
resmlp_model = ResMLP(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resmlp_model.parameters(), lr=0.001)

# Training loop
num_epochs = 20

for epoch in range(num_epochs):
    for batch_inputs, batch_labels in train_loader:
        batch_inputs = batch_inputs.view(-1, input_size)
        optimizer.zero_grad()
        outputs = resmlp_model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# Test the model
resmlp_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_inputs, batch_labels in test_loader:
        batch_inputs = batch_inputs.view(-1, input_size)
        outputs = resmlp_model(batch_inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

accuracy = correct / total
print(f'Accuracy on the test set: {100 * accuracy:.2f}%')

