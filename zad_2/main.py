import numpy as np
import file_reader

# Funkcje aktywacji
def relu(x):
    return np.maximum(0, x)

def linear(x):
    return x

class DenseLayer:
    def __init__(self, input_dim, output_dim, activation=linear, activation_derivative=None):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.biases = np.zeros(output_dim)
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.output = None
        self.input = None

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

        # Aktualizacja wag i biasów
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input

# Pochodne funkcji aktywacji
def relu_derivative(x):
    return (x > 0).astype(float)

def linear_derivative(x):
    return np.ones_like(x)

# Prosta sieć neuronowa z uczeniem
class SimpleNeuralNetwork:
    def __init__(self, input_dim):
        self.layer1 = DenseLayer(input_dim, 64, activation=relu, activation_derivative=relu_derivative)
        self.layer2 = DenseLayer(64, 32, activation=relu, activation_derivative=relu_derivative)
        self.output_layer = DenseLayer(32, 2, activation=linear, activation_derivative=linear_derivative)

    def forward(self, x):
        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        return self.output_layer.forward(x)

    def backward(self, loss_grad, learning_rate):
        grad = self.output_layer.backward(loss_grad, learning_rate)
        grad = self.layer2.backward(grad, learning_rate)
        self.layer1.backward(grad, learning_rate)

    def train(self, x_train, y_train, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            total_loss = 0
            for x, y_true in zip(x_train, y_train):
                predicted = self.forward(x)
                loss = np.mean((predicted - y_true) ** 2)
                total_loss += loss
                loss_grad = 2 * (predicted - y_true) / y_true.size
                self.backward(loss_grad, learning_rate)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss / len(x_train):.6f}")

# x_train = np.random.rand(100, 33)           # 100 przykładów, 33 cechy
# y_train = np.random.rand(100, 2)            # 100 docelowych wyjść (x, y)

measured, real = file_reader.read_from_file("./f8_1p.xlsx")

measured_mean = np.mean(measured, axis=0)
measured_std = np.std(measured, axis=0)
measured_std[measured_std == 0] = 1  # unikanie dzielenia przez 0

real_mean = np.mean(real, axis=0)
real_std = np.std(real, axis=0)
real_std[real_std == 0] = 1

measured_norm = (measured - measured_mean) / measured_std
real_norm = (real - real_mean) / real_std


network = SimpleNeuralNetwork(input_dim=10)
network.train(measured_norm, real_norm, epochs=50, learning_rate=0.001)
