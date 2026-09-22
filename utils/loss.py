import torch

def elbo_loss(y_pred, y_true, log_var, kl_divergence, kl_weight=0.001):
    """
    Evidence Lower Bound Loss
    """
    # 1. Negative Log Likelihood (NLL)
    # y ~ Gaussian(y_pred, exp(log_var))
    precision = torch.exp(-log_var)
    mse = (y_pred - y_true) ** 2
    nll = 0.5 * torch.mean(precision * mse + log_var)
    
    # 2. Total Loss
    loss = nll + kl_weight * kl_divergence
    return loss, nll

