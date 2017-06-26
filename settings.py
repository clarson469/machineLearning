import sys, os

# sys info
BASE_DIR = os.path.realpath(os.path.dirname(sys.argv[0]))
WEIGHT_DIR = os.path.join(BASE_DIR, 'data\\weights')
if not os.path.exists(WEIGHT_DIR):
    os.makedirs(WEIGHT_DIR)
    
MNIST_DIR = os.path.join(BASE_DIR, 'data\\MNIST')
STATS_DIR = os.path.join(BASE_DIR, 'data\\stats')
LOG_DIR = os.path.join(BASE_DIR, 'log')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# model hyperparameters
iterations = 200
hidden_dim = 40
r_lambda = 4
