from scipy.stats import entropy
import torch
import numpy as np

def jsd(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m), entropy(q, m))

def get_logits(hidden, model):
    normed = torch.nn.functional.layer_norm(hidden, (hidden.shape[-1],))
    logits = torch.matmul(normed, model.lm_head.weight.t())
    return torch.nn.functional.softmax(logits, dim=-1)
