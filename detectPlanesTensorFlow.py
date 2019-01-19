## DATA PREPROCESSING ##
import glob
import numpy as np
import os.path as path
from scipy import misc
from sklearn.model_selection import train_test_split

# IMAGE_PATH should be the path to the downloaded planesnet folder
IMAGE_PATH = 'data\planesnet'
file_paths = glob.glob(path.join(IMAGE_PATH, '*.png'))

# Load the images and normalize
images = [misc.imread(path) for path in file_paths]
images = np.asarray(images) / 255

# Get image size
image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])

# Read the labels from the filenames
n_images = images.shape[0]
labels = np.zeros((n_images, 1))
for i in range(n_images):
    filename = path.basename(file_paths[i])[0]
    labels[i, :] = int(filename[0])

# Split into test and training sets
TRAIN_TEST_SPLIT = 0.1

# Split the train and test data with random state=0 for repeatability
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=TRAIN_TEST_SPLIT, random_state=0)

## MODEL CREATION ##
import tensorflow as tf
from sklearn.metrics import accuracy_score
from datetime import datetime

# Model Hyperparamaters
N_LAYERS = 4

def dense_layer(X_prev, i_layer, neurons, activation):
    # INPUTS
    # X_prev       - tensor being passed from previous layer. Dimensions = (Batch, channels_in)
    # i_layer      - layer number
    # neurons      - number of neurons to be used in the dense operation
    # activation   - non-linear activation to be applied to the output of the dense operation
    #
    # OUTPUTS
    # X            - tensor output after dense operation, with activation. Dimensions = (Batch, neurons)
    # dense_output - tensor output after dense operation, without activation. Dimensions = (Batch, neurons)

    # Identify the dimension the incoming layer
    dimension_in = X_prev.get_shape()[1]

    # Declare all variables and operations within this scope to Dense_0 for Tensorboard
    with tf.variable_scope("Dense_" + str(i_layer)):

        # Create the weight variable for dense layer operations
        W = tf.get_variable(name="W_dense_0", shape=(dimension_in, neurons),
                            initializer=tf.initializers.glorot_uniform, dtype=tf.float32)

        # Create the bias variable for dense layer operations
        b = tf.get_variable(name="b_dense_0", shape=(neurons), initializer=tf.initializers.zeros,
                            dtype=tf.float32)

        # Perform matrix multiplication
        matmul = tf.matmul(X_prev, W)

        # Add bias to matrix multiplication output
        dense_output = matmul + b

        # Apply the specified non-linear activation function
        if activation == 'relu':
            X = tf.nn.relu(dense_output)
        elif activation == 'sigmoid':
            X = tf.nn.sigmoid(dense_output)

        return X, dense_output

def conv_layer(X_prev, i_layer, kernel, nuerons):
    # INPUTS
    # X_prev      - tensor being passed from previous layer. Dimensions = (Batch, H, W, channels_in)
    # i_layer     - layer number
    # kernel      - height and width of the convolutional kernel as a tuple, (H, W)
    # neurons     - number of neurons to be utilized for the convolution operation
    #
    # OUTPUTS
    # X           - tensor output after convolution operation. Dimensions = (Batch, H, W, nuerons)

    # Identify the number of channels present in the incoming layer
    channels_in = X_prev.get_shape()[3]

    # Declare all variables and operations within this scope to the appropriate convolution layer for TensorBoard
    with tf.variable_scope("Conv_" + str(i_layer)):

        # Create the weight variable for convolution operations
        W = tf.get_variable(name="W_conv_" + str(i_layer), shape=(kernel[0], kernel[1], channels_in, nuerons),
                                 initializer=tf.initializers.glorot_uniform, dtype=tf.float32)

        # Create the bias variable to be added to the tensor post-convolution
        b = tf.get_variable(name="b_conv_" + str(i_layer), shape=(nuerons), initializer=tf.initializers.zeros,
                                 dtype=tf.float32)

        # Calculate convolution operation on previous layer with the newly constructed convolutional filter
        convolution_output = tf.nn.conv2d(X_prev, filter=W, strides=(1, 1, 1, 1), padding='VALID')

        # Add the bias variable to the convolution output
        convolution_output = convolution_output + b

        # App a non-linear ReLu activation to the output
        X = tf.nn.relu(convolution_output)

        return X

def cnn(size, n_layers):
    # INPUTS
    # size     - size of the input images
    # n_layers - number of layers
    # OUTPUTS
    # model    - compiled CNN
    #
    # OUTPUTS
    # X_input                - Placeholder for passing input image data in. Dim = (batch, 20, 20, 3)
    # y                      - Placeholder for passing ground truth labels in. Dim = (batch, 1)
    # _dense_1_activated     - Output Tensor of final dense layer with sigmoid activation applied. Dim = (batch, 1)
    # matmul_1_not_activated - Output Tensor of final dense layer with no sigmoid activation applied. Dim = (batch, 1)

    # Define model hyperparamters
    MIN_NEURONS = 20
    MAX_NEURONS = 120
    KERNEL = (3, 3)

    # Determine the # of neurons in each convolutional layer
    steps = np.floor(MAX_NEURONS / (n_layers + 1))
    neurons = np.arange(MIN_NEURONS, MAX_NEURONS, steps).astype(np.int32)

    # Define the input placeholder. Dim = (batch, 20, 20, 3)
    X_input = tf.placeholder(name="X_Placeholder", dtype=tf.float32, shape=(None, size[0], size[1], size[2]))

    # Define the label placeholder
    y = tf.placeholder(name="Y_Placeholder", dtype=tf.float32, shape=(None, 1))

    # Apply 4 convolution layers
    X_conv_0 = conv_layer(X_input, 0, KERNEL, neurons[0]) # Dim = (batch, 18, 18, 20)
    X_conv_1 = conv_layer(X_conv_0, 1, KERNEL, neurons[1])      # Dim = (batch, 16, 16, 44)
    X_conv_2 = conv_layer(X_conv_1, 2, KERNEL, neurons[2])      # Dim = (batch, 14, 14, 68)
    X_conv_3 = conv_layer(X_conv_2, 3, KERNEL, neurons[3])      # Dim = (batch, 12, 12, 92)

    # Declare all variables and operations within this scope to Max_Pool for Tensorboard
    with tf.variable_scope("Max_Pool"):
        # Apply max pooling to minimize the number of parameters in the model
        X_max_pool = tf.nn.max_pool(X_conv_3, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID') # Dim = (batch, 6, 6, 92)

    # Declare all variables and operations within this scope to Flatten for Tensorboard
    with tf.variable_scope("Flatten"):
        # Get the shape of the incoming tensor
        single_example_dims = X_max_pool.get_shape()[1:]
        dim_product = np.prod(single_example_dims)

        # Reshape Tensor to a 2-D matrix so that we can apply dense layers
        X_reshaped = tf.reshape(X_max_pool, [-1, dim_product]) # Dim = (batch, 3312)

    # Build intermediate dense layer
    X_dense_0, _ = dense_layer(X_reshaped, 0, MAX_NEURONS, 'relu') # Dim = (batch, 120)

    # Build final dense layer that condenses output to a single neuron to allow for binary classification
    X_dense_1_activated, matmul_1_not_activated = dense_layer(X_dense_0, 1, 1, 'sigmoid') # Dim = (batch, 1)

    return X_dense_1_activated, X_input, y, matmul_1_not_activated

# Instantiate the model
logits, X_input, y, matmul_1 = cnn(size=image_size, n_layers=N_LAYERS)

## MODEL TRAINING ##
# Training Hyperparamters
EPOCHS = 150
BATCH_SIZE = 200

# Declare all variables and operations within the "cost" scope for TensorBoard
with tf.variable_scope("Cost"):
    # Use binary crossentropy on the logits. The input, "matmul_1" has not yet had an activation applied. The sigmoid
    # activation will be applied as part of the "sigmoid_cross_entropy_with_logits" operation. Finally, reduce the cost
    # function such that it is a sum of the cost from all training examples.
    cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=matmul_1))

# Declare all variables and operations with the "Optimizer" scope for TensorBoard
with tf.variable_scope("Optimizer"):
    # Use Adam optimization to determine how gradients and parameter updates should be handled
    optimizer = tf.train.AdamOptimizer()

    # Have the optimizer seek the minimization of the previously defined cost function
    operation = optimizer.minimize(cost)

# Create a logging directory so we can observe the graph we have created in TensorBoard
LOG_DIRECTORY_ROOT = 'logdir'
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
log_dir = "{}/run-{}/".format(LOG_DIRECTORY_ROOT, now)

# Declare all variables and operations with the "Variable_Initializer" scope for TensorBoard
with tf.variable_scope("Variable_Initializer"):
    # Create an operation that will initialize all of the variables (Weight, biases, etc.) when we run it in the session
    init = tf.global_variables_initializer()

# Create a TensorFlow Session for running data through the computation graph we have designed
with tf.Session() as sess:

    # Run the initialization to actually create our variables
    sess.run(init)

    # Create a summary writer to log our graph in TensorBoard
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    # Calculate the number of iterations in each epoch
    n_iterations = int(X_train.shape[0] / BATCH_SIZE)

    # Iterate through each epoch
    for i_epoch in range(EPOCHS):

        # Declare a variable to hold the average batch_cost
        batch_cost = np.zeros((n_iterations))

        # Iterate through each iteration within the epoch.
        for i_iteration in range(n_iterations):

            # Select a batch of data (both images and labels)
            batch_indices = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
            X_batch = X_train[batch_indices, :, :, :]
            y_batch = y_train[batch_indices, :]

            # Run the batch through the data. The feed dictionary is like a roller coaster cart, and the computation graph
            # we defined previously is the roller coaster. The feed dict keeps taking new passengers (data) on and repeatedly
            # runs through the graph. The sess.run function also takes in [operation, cost]. Operation is associated with
            # the optimizer. When "operation" is passed to the graph, it will modify our variables using the backpropagation algorithm.
            # There is no interesting output for this operation, which is why the 1st output on the LHS of the equation is empty.
            # "cost" simply calculates the cost for the given batch and records the scalar in the "batch_cost" numpy array we have create
            _, batch_cost[i_iteration] = sess.run([operation, cost], feed_dict={X_input:X_batch, y:y_batch})

        # This will calculate the output probabilities which are stored in the logit tensor. This was the version of the graph
        # output that had the sigmoid activation applied to it. The output will be values between 0-1 that represent how likely
        # it is that the image contains an airplane. We only need to pass it input data
        test_preds = sess.run(logits, feed_dict={X_input:X_test})

        # Caluclate the test set accuracy
        test_accuracy = accuracy_score(y_test, np.round(test_preds))

        # Report the average training loss per batch. This can also be done via TensorBoard
        print('Epoch ' + str(i_epoch) + ': Average Train Loss: ' + str(np.mean(batch_cost) / BATCH_SIZE) + ', Average Test Accuracy: ' + str(100 * test_accuracy) + '%')

