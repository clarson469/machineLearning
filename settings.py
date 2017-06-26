import sys, os

# sys info
BASE_DIR = os.path.realpath(os.path.dirname(sys.argv[0]))
WEIGHT_DIR = os.path.join(BASE_DIR, 'data\\weights')
MNIST_DIR = os.path.join(BASE_DIR, 'data\\MNIST')
STATS_DIR = os.path.join(BASE_DIR, 'data\\stats')
LOG_DIR = os.path.join(BASE_DIR, 'log')

# model hyperparameters
iterations = 200
hidden_dim = 40
r_lambda = 4
