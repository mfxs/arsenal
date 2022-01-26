# All the packages needed
import time
import math
import torch
import warnings
import argparse
import cvxpy as cp
import numpy as np
import seaborn as sns
from torch import nn, optim
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_digits
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error
