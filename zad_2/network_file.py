import numpy as np
import pickle
import os

# Funkcje aktywacji i pochodne
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

# Warstwa Dense z możliwością zapisu i wczytywania
class DenseLayer:
    def __init__(self, input_dim, output_dim, activation=linear, activation_derivative=None):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.biases = np.zeros(output_dim)
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.weights) + self.biases
        self.output = self.activation(self.z)
        return self.output

    def backward(self, grad_output, learning_rate):
        grad_activation = grad_output * self.activation_derivative(self.z)
        grad_weights = np.outer(self.input, grad_activation)
        grad_biases = grad_activation
        grad_input = np.dot(self.weights, grad_activation)

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input

    def get_params(self):
        return {'weights': self.weights, 'biases': self.biases}

    def set_params(self, params):
        self.weights = params['weights']
        self.biases = params['biases']

# Sieć neuronowa z obsługą zapisu/odczytu
class SimpleNeuralNetwork:
    def __init__(self, input_dim):
        self.layer1 = DenseLayer(input_dim, 128, activation=relu, activation_derivative=relu_derivative)
        self.layer2 = DenseLayer(128, 64, activation=relu, activation_derivative=relu_derivative)
        self.layer3 = DenseLayer(64, 16, activation=relu, activation_derivative=relu_derivative)
        self.output_layer = DenseLayer(16, 2, activation=linear, activation_derivative=linear_derivative)

    def forward(self, x):
        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        x = self.layer3.forward(x)
        return self.output_layer.forward(x)

    def backward(self, loss_grad, learning_rate):
        grad = self.output_layer.backward(loss_grad, learning_rate)
        grad = self.layer3.backward(grad, learning_rate)
        grad = self.layer2.backward(grad, learning_rate)
        self.layer1.backward(grad, learning_rate)

    def get_all_params(self):
        return {
            'layer1': self.layer1.get_params(),
            'layer2': self.layer2.get_params(),
            'layer3': self.layer3.get_params(),
            'output_layer': self.output_layer.get_params()
        }

    def set_all_params(self, params):
        self.layer1.set_params(params['layer1'])
        self.layer2.set_params(params['layer2'])
        self.layer3.set_params(params['layer3'])
        self.output_layer.set_params(params['output_layer'])

    def save(self, filename="model_weights.pkl"):
        params = self.get_all_params()
        with open(filename, 'wb') as f:
            pickle.dump(params, f)
        print(f"[SAVE] Model saved to '{filename}'")
        for layer_name, layer_params in params.items():
            w = layer_params['weights']
            b = layer_params['biases']
            print(f"[SAVE] {layer_name} weights (preview):\n{w[:3, :5]}")
            print(f"[SAVE] {layer_name} biases (preview):\n{b[:5]}")

    def load(self, filename="model_weights.pkl"):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                params = pickle.load(f)
                self.set_all_params(params)
            print(f"[LOAD] Model loaded from '{filename}'")
            for layer_name, layer_params in params.items():
                w = layer_params['weights']
                b = layer_params['biases']
                print(f"[LOAD] {layer_name} weights (preview):\n{w[:3, :5]}")
                print(f"[LOAD] {layer_name} biases (preview):\n{b[:5]}")
        else:
            print(f"[LOAD] No saved model found at '{filename}'. Starting from scratch.")

    def train(self, x_train, y_train, epochs=100, learning_rate=0.01, checkpoint_interval=10):
        for epoch in range(epochs):
            total_loss = 0
            for x, y_true in zip(x_train, y_train):
                y_pred = self.forward(x)
                loss = np.mean((y_pred - y_true) ** 2)
                total_loss += loss
                loss_grad = 2 * (y_pred - y_true) / y_true.size
                self.backward(loss_grad, learning_rate)
            avg_loss = total_loss / len(x_train)
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.6f}")

            if (epoch + 1) % checkpoint_interval == 0:
                self.save(f"model_weights_epoch_{epoch + 1}.pkl")
                print(f"Checkpoint saved at epoch {epoch + 1}.")