# DeepTMB
## A probabilistic regression framework

DeepTMB uses TensorFlow Probability to model the distribution of the label as a function of input predictors.  The approach is similar to https://arxiv.org/abs/2109.07250, however we did not find the sinh-arcsinh normal distribution to be stable for our data, and that distribution is defined for all values whereas a metric such as TMB is strictly positive.  The log-normal distribution is a common distribution only defined for positive values, and we found a mixture of these distributions to be a flexible approach to modeling strictly positive data.  One issue with mixture models and TensorFlow Probability is that certain built-in functions are not defined, which required us to implement them with our own code starting from the predicted probability density.  Although designed for TMB, this framework could be used for any data.  For data that can be negative or positive the log-normal distributions can be replaced with normal distributions.

## Publication
This repository is associated with this publication: https://www.biorxiv.org/content/10.1101/2022.04.22.489230v2.  To reproduce the first version: https://www.biorxiv.org/content/10.1101/2022.04.22.489230v1, refer to the first release: https://github.com/OmnesRes/DeepTMB/releases/tag/v1.0.0

## Dependencies
The code was run with Python 3.8, TensorFlow 2.7.0, and TensorFlow Probability 0.15.0.

## Use
All the code required to run the model is present in the "model" folder.  An IPython Notebook that demonstrates how to train a model and make predictions with example data is present at https://github.com/OmnesRes/DeepTMB/blob/master/example/example.ipynb.  Alternatively the code can be run by calling
```
python example/example.py
```
while in the DeepTMB repository.
