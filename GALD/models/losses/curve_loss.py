import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.optimize import curve_fit

def compute_curvature(x, params):

    # first derivative f'(x)
    first_derivative = 3 * params[0] * x**2 + 2 * params[1] * x + params[2]
    # second derivative f''(x)
    second_derivative = 6 * params[0] * x + 2 * params[1]

    curvature = torch.abs(second_derivative) / (1 + first_derivative**2)**1.5
    return curvature


def fit_polynomial(x, y):

    xdata = x.detach().cpu().numpy()
    ydata = y.detach().cpu().numpy()

    popt, pcov = curve_fit(cubic_func, xdata, ydata)
    return popt

def cubic_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


def curve_loss(ys, reg_preds, reg_targets, img_w):
    # The key code will be released later.
    return 0
