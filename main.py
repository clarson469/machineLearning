import sys, os, io, pickle, datetime
import numpy as np
import settings
from lib.mnist import MNIST
from models import NN_Classifier
import matplotlib.pyplot as plt

def train_nn():
    stamp = datetime.datetime.now()
    f_stamp = stamp.strftime('%Y%m%d%H%M%S')
    v_stamp = stamp.isoformat(sep=' ')

    nn = NN_Classifier(MNIST)
    nn.train()

    plt.plot(range(len(nn.train_costs)), nn.train_costs, 'b-')
    plt.plot(range(len(nn.validation_costs)), nn.validation_costs, 'y-')
    plt.xlabel('No. of Iterations')
    plt.ylabel('Cost J')
    plt.legend(['Training Set Cost', 'Validation Set Cost'])
    plt.savefig(os.path.join(settings.LOG_DIR, 'Cost History {}.png'.format(f_stamp)))
    plt.show()

    log_text = [
        '##### 3-Layer Neural Net Classifier {} #####\n\n'.format(v_stamp),
        'Dataset used: MNIST\n',
        'Final Cost: {}\n'.format(nn.train_costs[-1]),
        '## Settings ##\n',
        'Iterations: {0}\nLambda: {1}\nHidden Dimension: {2}\n'.format(settings.iterations, settings.r_lambda, settings.hidden_dim),
        'Cost Graph saved to file: Cost History {}.png'.format(f_stamp)
    ]

    data.save_pk(os.path.join(settings.WEIGHT_DIR, 'weights{}'.format(f_stamp)), nn.params)

    data.save_log(os.path.join(settings.LOG_DIR, 'Log {}.txt'.format(f_stamp)), log_text)

def display_nn():
    f_name = os.path.join(settings.WEIGHT_DIR, sys.argv[2])
    params = data.load_pk(f_name)
    Xte = data.load_pk(os.path.join(settings.MNIST_DIR, 'test-images'))
    Yte = data.load_pk(os.path.join(settings.MNIST_DIR, 'test-labels'))

    sys.stdout = tmp = io.StringIO()
    nn = NN_Classifier(data.MNIST)
    sys.stdout = sys.__stdout__

    y_n = input('Make Prediction? (Y/n)').lower()

    while y_n == 'y':
        print('\n')
        i = np.random.randint(10000)
        plt.imshow(Xte[i])
        plt.show()

        h = nn.predict(params, Xte[i].reshape(1, 28 ** 2))

        if input('Show full prediction? (Y/n)').lower() == 'y':
            print(h)

        print('\nPrediction: {}'.format(np.argmax(h)))
        print('Expected: {}'.format(Yte[i]))

        y_n = input('Predict again? (Y/n)').lower()


if __name__ == '__main__':
    if '-d' in sys.argv:
        display_nn()
    else:
        train_nn()
