import numpy as np
import matplotlib.pyplot as plt
class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.array([
            np.random.randn(),
            np.random.randn()
        ])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (
            1 - self._sigmoid(x)
        )
    def predict(self, input_vector):
        layer_1 = (
            np.dot(input_vector, self.weights)
            + self.bias
        )
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction
    def _compute_gradients(self, input_vector, target):
        layer_1 = (
            np.dot(input_vector, self.weights)
            + self.bias
        )
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        error_prediction = 2 * (
            prediction - target
        )
        prediction_layer1 = (
            self._sigmoid_deriv(layer_1)
        )
        dlayer1_dbias = 1
        dlayer1_dweights = input_vector
        error_dbias = (
            error_prediction
            * prediction_layer1
            * dlayer1_dbias
        )
        error_dweights = (
            error_prediction
            * prediction_layer1
            * dlayer1_dweights
        )
        return error_dbias, error_dweights
    def _update_parameters(
        self,
        error_dbias,
        error_dweights
    ):
        self.bias = self.bias - (
            error_dbias
            * self.learning_rate
        )
        self.weights = self.weights - (
            error_dweights
            * self.learning_rate
        )
    def train(
        self,
        input_vectors,
        targets,
        iterations
    ):
        cumulative_errors = []
        for current_iteration in range(iterations):
            random_data_index = np.random.randint(
                len(input_vectors)
            )
            input_vector = input_vectors[
                random_data_index
            ]
            target = targets[
                random_data_index
            ]
            error_dbias, error_dweights = (
                self._compute_gradients(
                    input_vector,
                    target
                )
            )
            self._update_parameters(
                error_dbias,
                error_dweights
            )
            if current_iteration % 100 == 0:
                cumulative_error = 0
                for data_instance_index in range(
                    len(input_vectors)
                ):
                    data_point = input_vectors[
                        data_instance_index
                    ]
                    target = targets[
                        data_instance_index
                    ]
                    prediction = self.predict(
                        data_point
                    )
                    error = np.square(
                        prediction - target
                    )
                    cumulative_error += error
                # Store cumulative error
                cumulative_errors.append(
                    cumulative_error
                )
        return cumulative_errors
input_vectors = np.array([
    [3, 1.5],
    [2, 1],
    [4, 1.5],
    [3, 4],
    [3.5, 0.5],
    [2, 0.5],
    [5.5, 1],
    [1, 1]
])
targets = np.array([0,1,0,1,0,1,1,0])
learning_rate = 0.1
neural_network = NeuralNetwork(learning_rate)
training_error = neural_network.train(
    input_vectors,
    targets,
    10000
)
plt.plot(range(0, 10000,100),training_error)
plt.xlabel("Iterations")
plt.ylabel("Cumulative Error")
plt.title("Training Error Curve")
plt.show()