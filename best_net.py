from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from cs231n.classifiers.neural_net import TwoLayerNet

from cs231n.data_utils import load_CIFAR10

from cs231n.gradient_check import eval_numerical_gradient


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

def rel_error(x, y):
	""" returns relative error """
	return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)



best_net = None # store the best model into this 



#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################
pass

try_times = 10
input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
hidden_size_history = []
learning_rate_history = []
reg_history = []
val_acc_history = []
for i in range(try_times):

	hidden_size = np.random.randint(80, 120)
	learning_rate_val = 10 ** np.random.uniform(-4, -3)
	reg_val = 10 ** np.random.uniform(-1, 1)

	hidden_size_history.append(hidden_size)
	learning_rate_history.append(learning_rate_val)
	reg_history.append(reg_val)

	net = TwoLayerNet(input_size, hidden_size, num_classes)

	# Train the network
	stats = net.train(X_train, y_train, X_val, y_val,
	            num_iters = 1000, batch_size = 200,
	            learning_rate = learning_rate_val, learning_rate_decay = 0.95,
	            reg = reg_val, verbose=False)

	# Predict on the validation set
	val_acc = (net.predict(X_val) == y_val).mean()
	val_acc_history.append(val_acc)
	print('Validation accuracy: ', val_acc)
	print(hidden_size, learning_rate_val, reg_val)

max_idx = np.argmax(val_acc_history)
print('best val_acc', val_acc_history[max_idx])
print('best params ', hidden_size_history[max_idx], learning_rate_history[max_idx], reg_history[max_idx])

#################################################################################
#                               END OF YOUR CODE                                #
#################################################################################