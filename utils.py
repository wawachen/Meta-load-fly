import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import torch
from torch import nn as nn


def swish(x):
    return x * torch.sigmoid(x)


def truncated_normal(size, std):
    # We use TF to implement initialization function for neural network weight because:
    # 1. Pytorch doesn't support truncated normal
    # 2. This specific type of initialization is important for rapid progress early in training in cartpole

    # Do not allow tf to use gpu memory unnecessarily
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True

    sess = tf.Session(config=cfg)
    val = sess.run(tf.random.truncated_normal(shape=size, stddev=std))

    # Close the session and free resources
    sess.close()

    return torch.tensor(val, dtype=torch.float32)


def get_affine_params(ensemble_size, in_features, out_features):

    w = truncated_normal(size=(ensemble_size, in_features, out_features),
                         std=1.0 / (2.0 * np.sqrt(in_features)))
    w = nn.Parameter(w)

    b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32))

    return w, b

def gradient_f(y, x, grad_outputs=None):
    """reference: https://github.com/vsitzmann/siren"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad
    