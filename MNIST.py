from tensorflow.keras.datasets import mnist # type: ignore
import numpy as np
 
# initialisation - reads the mnist dataset from the tensorflow library. Done this way to maintain ease-of-use, but would work in other ways.
def read():
	(train_X, train_y), (test_X, test_y) = mnist.load_data()
	return train_X, train_y, test_X, test_y

# Normalisation - evenly distributes all numbers to be between 0 and 1 for easier training. 
def normalise(num, min, max): # 0, 255
	return (num - min) / (max-min)

# For converting train_y and test_y to arrays of ints that mimic the intended output - much, much easier for training. 
def y_fix(num):
	array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	array[num] = 1
	return array

# The sigmoid function.
#     1
# ----------
# 1 + e^(-x)
# Also known as the activation function.
# Very short, but very important - this smooths any input to be much, much closer to zero. 
# X Axis - Input Number
# Y Axis - Output Number
#  1 ┌─────────────────────┐
#    │                xxxxx│
#    │             xxx     │
#    │           xx        │
# .5 │          x          │
#    │        xx           │
#    │     xxx             │
#    │xxxxx                │
#  0 └─────────────────────┘
#     -6         0        6 
# Notice how, the further the input is from zero, the closer the output is to either 0 or -1: it represents the same extremes, but with an exponential scale between 0 and 1. 
def sigmoid(x):
	x = np.clip(x, -500, 500)
	return 1/(1+np.exp(-x))

def sigmoidderivative(x):
	return x * (1-x)
    
def relu(x):
	return np.where(x > 0, x, x * 0.01)

def relu_derivative(x):
	return np.where(x > 0, 1, 0.01)

# The Neuron - can be used for pretty much anything. Set up correctly, you can use these to simulate NAND gates, and therefore anything a computer can do. 
# Takes an input of one set of data - as large or as small a set of numbers as you'd like - a corresponding set of weights, to show how significant each number should be in the final equation, and a single bias number, to make the output easier or harder to approach 1.
def neuron(data, weights, bias, activation='sigmoid'):
	z = np.dot(data, weights) + bias
	if activation == 'relu':
		a = relu(z)
	else:
		a = sigmoid(z)
	return z, a
# A layer of neurons. Takes the outputs from another layer, the weights and bias matrices for this layer, and how many neurons are in the layer. Gives the correct weights and biases to each neuron along with all of the data, then returns the output of every neuron.
def layer(previous_layer_outputs, this_weights, this_biases, neuron_count, activation='sigmoid'):
	zs, layer_outputs = [], []
	for i in range(neuron_count):
		z, output = neuron(previous_layer_outputs, this_weights[i], this_biases[i], activation)
		layer_outputs.append(output)
		zs.append(z)
	return zs, layer_outputs

# Very abstract. Essentially, this gets all of the training data, sends it to the first layer to be correctly summed up and returned, then sends the results of the first layer to the next.
# Note: Only one hidden layer for this. I could make more, of course, but this is a demonstration.
# Best way to implement further layers wouldn't be more matrices, it would be a 3D matrix - x and y stay the same, but stepping in z would represent further layers.
# Returns the output of the last layer.
def trainnn(training_data, weight_matrix, bias_matrix, weights2, biases2):
	trained_data = []
	output_data = []
	hidden_zs = []
	# Hidden Layer
	for i in range(len(training_data)):
		zs, output = layer(training_data[i].flatten(), weight_matrix, bias_matrix, 16, activation='relu')
		trained_data.append(output)
		hidden_zs.append(zs)

	# Output Layer
	for i in range(len(trained_data)):
		z, output = layer(trained_data[i], weights2, biases2, 10, activation='sigmoid')
		output_data.append(output)

	return trained_data, output_data, hidden_zs

# Generates the cost function - a single number that represents how close the outputs of the entire neural network were to being correct. The goal is to make this number 0, meaning the neural networks outputs perfectly matched what is in train_y and test_y.
# Making the cost equal 0 is not possible, so the goal is to get close.
def generateC(trained_data, check_list):
	sum = 0
	for i in range(len(trained_data)):
		for x in range(len(trained_data[i])):
			sum += (check_list[i][x] - trained_data[i][x]) ** 2
	return (1/(2 * len(trained_data))) * sum
	
def generateError(trained_data, check_list):
	errors = []
	for p, a in zip(trained_data, check_list):
		p = np.array(p)
		a = np.array(a)
		error = (p - a) * sigmoidderivative(p)
		errors.append(error)
	return errors

def generateHiddenError(error, hidden_outputs):
	hidden_errors = []
	for output_error, hidden_output in zip(error, hidden_outputs):
		hidden_error = np.dot(weight_matrix2.T, output_error) * relu_derivative(np.array(hidden_output))
		hidden_errors.append(hidden_error)
	return hidden_errors

	

if __name__ == "__main__":

	# Higher for faster but less accurate. Lower for slower but more accurate.
	learning_rate = .2
	# Higher for better learning. Takes longer the higher. Diminishing returns - will not approach 100% accuracy, so 1,000,000 epochs will take a VERY long time and be - in all likelihood - exactly as accurate as 1,000, which won't be 100% accurate.
	epochs = 20

	# the following few lines are entirely dedicated to preparing the data for training

	# This process would be more efficient if I used the actual images directly instead of a preprocessed dataset. However, this way is more efficient for me to code - I wouldn't learn anything by wasting time reading the images.

	train_X, train_y, test_X, test_y = read()
    
	train_X = list(train_X)
	train_y = list(train_y)
	test_X = list(test_X)
	test_y = list(test_y)
    
	print("Converting training and testing data to proper data types. Please wait.")
	for i in range(len(train_y)):
		train_y[i] = y_fix(train_y[i])
	for i in range(len(test_y)):
		test_y[i] = y_fix(test_y[i])
	print("Converted train_y and test_y elements to arrays.")
	print("Normalising training data.")
	for i in range(len(train_X)):
		train_X[i] = normalise(train_X[i], 0, 255)
	print("Training data normalised. Model to begin training. Press enter.")
	input()

	# Done. Onto the actual model.

	weight_matrix = np.random.rand(16, 784) * 0.01 # Initialise first layer matrices with random values between 0 and 1.
	# Weight Matrix
	# Rows are neurons. Each neuron reads a specific row. 
	# Columns are pixels - the weights top left pixel in each image goes into column 0.
	bias_matrix = np.random.rand(16,) * 0.01
	# Formatted as one column to go along with Weight Matrix - each row corresponds to a neuron as its inbuilt bias.

	# The formula for calculating the output of each neuron is sigmoid(((pixel_weight * actual_pixel_value) + (pixel2_weight * actual_pixel2_value)... + (pixel784weight * actual_pixel784_value)) + bias). 
	# Check sigmoid() for the explanation of the sigmoid function - a little verbose to do it again here. Otherwise, it's just adding the values of each input * the weights of each input together, then the bias on the end.

	# Similar to above, but only with 10 neurons - because there's only 10 numbers to output - and 16 inputs, for the 16 neurons in the other matrices. Replace "pixel" with something like "last layer output" to understand this layer.
	weight_matrix2 = np.random.rand(10, 16) * 0.01
	bias_matrix2 = np.random.rand(10,) * 0.01
	
	for epoch in range(epochs):
	# Forward Pass
		print(weight_matrix[0][:5])
		hidden_outputs, trained_data, hidden_zs = trainnn(train_X, weight_matrix, bias_matrix, weight_matrix2, bias_matrix2)
		print("Forward pass complete.")

		cost = generateC(trained_data, train_y)
		error = generateError(trained_data, train_y)
		hidden_error = generateHiddenError(error, hidden_zs)
		print(hidden_error[0][:5])
        
	# Backpropagation
		for i in range(len(train_X)):
			for j in range(10):
				for k in range(16):
					weight_matrix2[j][k] -= learning_rate * error[i][j] * hidden_outputs[i][k]
				bias_matrix2[j] -= learning_rate * error[i][j]

		for i in range(len(train_X)):
			input_flat = train_X[i].flatten()
			for j in range(16):
				for k in range(784):
					weight_matrix[j][k] -= learning_rate * hidden_error[i][j] * input_flat[k]
				bias_matrix[j] -= learning_rate * hidden_error[i][j]
		# NOT MY CODE #
		# Outputs how many of the training samples it got right for each epoch
		correct = 0
		for i in range(len(train_y)):
			if np.argmax(trained_data[i]) == np.argmax(train_y[i]):
				correct += 1
		accuracy = correct / len(train_y)
		print(f"\nTraining Accuracy: {accuracy * 100:.2f}%")
	
	# Once epochs are done, this outputs a proper accuracy.
	print("Training complete. Testing.")
	hidden_test_outputs, test_predictions = trainnn(test_X, weight_matrix, bias_matrix, weight_matrix2, bias_matrix2)
	correct = 0
	for i in range(len(test_y)):
		if np.argmax(test_predictions[i]) == np.argmax(test_y[i]):
			correct += 1

	test_accuracy = correct / len(test_y)
	print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")