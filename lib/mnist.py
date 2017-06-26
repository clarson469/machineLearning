import numpy as np
from .data import load_pk

class MNIST(object):
    def __init__(self):
        self.f_names = [os.path.join(settings.MNIST_DIR, f) for f in os.listdir(settings.MNIST_DIR) if 'training-images' in f]
        self.load_train()
        self.collapse_images()
        self.expand_labels()
        self.separate_validation_set()
        self.join()
        self.input_dim = 28 ** 2
        self.output_dim = 10

    def load_train(self):
        self.training_batches = [load_pk(f) for f in self.f_names]
        train_labels = load_pk(os.path.join(settings.MNIST_DIR, 'training-labels'))
        self.training_labels = [train_labels[i * 10000 : 10000 + i * 10000] for i in range(6)]

    def separate_validation_set(self):
        self.validation_set = self.training_batches.pop()
        self.validation_labels = self.training_labels.pop()
        self.bin_validation_labels = self.bin_training_labels.pop()

    def collapse_images(self):
        for i, batch in enumerate(self.training_batches):
            self.training_batches[i] = batch.reshape(10000, 28 ** 2)

    def expand_labels(self):
        self.bin_training_labels = [np.zeros((10000, 10)) for i in range(6)]
        for i, batch in enumerate(self.bin_training_labels):
            batch[range(10000), self.training_labels[i]] = 1

    def join(self):
        self.Xtr = np.vstack(self.training_batches)
        self.Ytr = np.vstack(self.training_labels)
        self.Ytr_V = np.vstack(self.bin_training_labels)
