# machineLearning
This is basically a collection of machine learning attempts. It will also (eventually) contain a `docs` folder with blog posts on the topic.  
Everything here is pretty basic - it's my attempt at learning the subject, and is essentially a modified version of the coding assignments on [Andrew Ng's Coursera](https://www.coursera.org/learn/machine-learning/home/welcome), re-written for Python.  
Everything is implemented using NumPy and Scipy.  
  
### Contents
A model for Logistic Regression (that may or may not be broken at the minute), and a model for a three-layer Neural Network (i.e. two hidden layers and one output layer) -- both of these are in the `models.py` file, though they may be separated at some point.  
The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset as a collection of pickle files, and the [UCI Machine Learning](http://archive.ics.uci.edu/ml/index.php) [adult](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/) dataset, renamed as "census-wage" because it's more clear.  Each of these datasets have a class that turns them into a single, usable 2-D numpy array  
  
#### A Note on Requirements.txt
This repo uses the NumPy and Scipy libraries, but specifically it uses numpy+mkl, which contains numpy and required DLLs for the Intel Math Kernel Library - basically, numpy and (probably) scipy should be installed from a pre-downloaded wheel rather than by just directing pip (or equivalent) to the requirements file. Those can be found here -- [numpy+mkl](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy) & [scipy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy).  
