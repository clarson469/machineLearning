import settings
import numpy as np
from scipy.optimize import fmin_cg
from lib import util

class Base(object):
    def __init__(self, data_obj):
        print('Loading Data...\n')
        self.data = data_obj()
        print('Done!\n')
        self.regularise = True
        self.m = self.data.Xtr.shape[0]

class Logistic_Regression(Base):
    def _init__(self, data_obj):
        super(Logistic_Regression, self).__init__(data_obj)

    def cost(self, params, *args):
        X, y = args
        h = self.predict(params, X)
        h[y==0] = 1 - h[y==0]
        J = -np.mean(np.sum(np.log(h), axis=1))

        if self.regularise:
            reg_theta = params.reshape(self.data.output_dim, self.data.input_dim + 1)[:, 1:]
            reg = np.dot(reg_theta, reg_theta.T) * (settings.r_lambda / (2 * self.m))
            J += reg

        return J

    def gradient(self, params, *args):
        X, y = args
        h = self.predict(params, X)
        grad = np.mean((h - y) * util.add_intercept(X), axis=0)

        if self.regularise:
            reg_theta = reg_theta = params.reshape(self.data.output_dim, self.data.input_dim + 1)
            reg_theta[:, 0] = 0
            grad += reg_theta * (settings.r_lambda / self.m)

        return grad

    def train(self):
        print('Training logistic regression with conjugate gradient...\n')
        init_params = np.random.randn(self.data.output_dim, self.data.input_dim + 1).flatten()

        self.ic = util.IterCount('No. of Iterations:', auto=True)
        self.train_costs = []
        self.validation_costs=  []

        fmin = fmin_cg(self.cost, init_params, **{
            'fprime':self.gradient,
            'args':(self.data.Xtr, self.data.Ytr),
            'maxiter':settings.iterations,
            'gtol':1e-7,
            'callback':self.callback
        })

        self.params = fmin

    def predict(self, params, X):
        theta = params.reshape(self.data.output_dim, self.data.input_dim + 1)
        a_1 = util.add_intercept(X)
        return util.sigmoid(np.dot(a_1, theta.T))

    def callback(self, params):
        # turn off regularisation checking costs
        self.regularise = False

        self.train_costs.append(self.cost(params, self.data.Xtr, self.data.Ytr))
        self.validation_costs.append(self.cost(
            params,
            self.data.Xcv,
            self.data.Ycv
        ))

        self.ic.auto_update()

        # restore for optimisation
        self.regularise = True


class NN_Classifier(Base):
    def __init__(self, data_obj):
        super(NN_Classifier, self).__init__(data_obj)
        self.sizes = [
            (settings.hidden_dim, self.data.input_dim + 1),
            (settings.hidden_dim, settings.hidden_dim + 1),
            (self.data.output_dim, settings.hidden_dim + 1)
        ]
        self.cache_f_prop = True

    def forward_propagate(self, X, *thetas):
        t_1, t_2, t_3 = thetas

        # separate z and a values for caching purposes
        # the z values are required for backprop separately to the a values
        a_1 = util.add_intercept(X)
        z_2 = np.dot(a_1, t_1.T)
        a_2 = util.add_intercept(util.sigmoid(z_2))
        z_3 = np.dot(a_2, t_2.T)
        a_3 = util.add_intercept(util.sigmoid(z_3))

        # this is also "h", the hypothesis
        # it does not require an intercept
        # its z value is not needed
        a_4 = util.sigmoid(np.dot(a_3, t_3.T))

        if self.cache_f_prop:
            self.a_vals = [a_1, a_2, a_3, a_4]
            self.z_vals = [z_2, z_3]

        return a_4

    def back_propagate(self, y, *thetas):
        t_1, t_2, t_3 = thetas

        d_4 = self.a_vals[-1] - y
        # remove intercept values and get sig_prime on the z values
        # because we don't include intercept / bias terms in back propagation
        d_3 = np.dot(d_4, t_3)[:, 1:] * util.sig_prime(self.z_vals[-1])
        d_2 = np.dot(d_3, t_2)[:, 1:] * util.sig_prime(self.z_vals[-2])

        return d_2, d_3, d_4

    def cost(self, params, *args):
        X, y = args
        thetas = util.unwrap(params, self.sizes)

        h = self.forward_propagate(X, *thetas)

        # vectorization of the logistic cost function:
        # J = -Ave(y * log(h_i) - (1 - y) * log(1 - h_i))
        h[y==0] = 1 - h[y==0]
        J = -np.mean(np.sum(np.log(h), axis=1))

        if self.regularise:
            # remove intercept / bias terms again
            reg_theta = np.hstack([t[:, 1:].flatten() for t in thetas])
            # vectorize elementwise square
            # make reg_theta 2-D so the dot function works properly
            # (or, rather, the transpose)
            reg_theta = reg_theta.reshape(reg_theta.shape[0], 1)
            J += np.dot(reg_theta.T, reg_theta) * (settings.r_lambda / (2 * self.m))

        return J

    def gradient(self, params, *args):
        X, y = args
        thetas = util.unwrap(params, self.sizes)
        # forward prop first, to get the most recent a and z values for backprop
        # (cost is not called first in fmin_cg)
        self.forward_propagate(X, *thetas)
        d_2, d_3, d_4 = self.back_propagate(y, *thetas)

        # weird transposition scheme will probably change
        # but this was how Octave wanted it to be
        t_1_grad = np.dot(self.a_vals[0].T, d_2).T / self.m
        t_2_grad = np.dot(self.a_vals[1].T, d_3).T / self.m
        t_3_grad = np.dot(self.a_vals[2].T, d_4).T / self.m

        if self.regularise:
            rt_1, rt_2, rt_3 = thetas
            # removing intercept / bias terms, but keeping
            # full shapes so that the regularisation terms
            # broadcast properly against the gradients
            rt_1[:, 0] = 0
            rt_2[:, 0] = 0
            rt_3[:, 0] = 0
            t_1_grad += rt_1 * (settings.r_lambda / self.m)
            t_2_grad += rt_2 * (settings.r_lambda / self.m)
            t_3_grad += rt_3 * (settings.r_lambda / self.m)

        return np.hstack((t_1_grad.flatten(), t_2_grad.flatten(), t_3_grad.flatten()))

    def train(self):
        print('Training neural network with conjugate gradient...\n')

        self.cache_f_prop = True
        self.ic = util.IterCount('No. of Iterations:', auto=True)
        self.train_costs = []
        self.validation_costs = []

        init_params = util.init_params(self.sizes)

        fmin = fmin_cg(self.cost, init_params, **{
            'fprime':self.gradient,
            'args':(self.data.Xtr, self.data.Ytr_V),
            'maxiter':settings.iterations,
            'gtol':1e-7,
            'callback':self.callback
        })

        self.params = fmin

    def predict(self, params, X):
        self.cache_f_prop = False
        return self.forward_propagate(X, *util.unwrap(params, self.sizes))


    def callback(self, params):
        # turn off regularisation and caching for taking checking costs
        self.regularise = False
        self.cache_f_prop = False

        self.train_costs.append(self.cost(params, self.data.Xtr, self.data.Ytr_V))
        self.validation_costs.append(self.cost(
            params,
            self.data.validation_set,
            self.data.bin_validation_labels
        ))

        self.ic.auto_update()

        # restore these for optimisation
        self.regularise = True
        self.cache_f_prop = True
