def fit_sigmoid_scaling(nll_list, acc_list, predict_nll):
    """
    Fit acc = 1 / (1 + exp(a*nll + b)) via least squares in logit space and
    predict accuracy at predict_nll.

    Returns:
        [a, b, predicted_acc] as a list of floats.
    """
    import numpy as np
    nll = np.array(nll_list)
    acc = np.array(acc_list)

    acc_clipped = np.clip(acc, 1e-9, 1 - 1e-9)

    z = np.log((1-acc_clipped)/acc_clipped)

    # Design matrix: [nll, 1]
    X = np.column_stack([nll, np.ones(len(nll))])
    
    # Solve normal equations: [a, b] = (X^T X)^{-1} X^T z
    a, b = np.linalg.lstsq(X, z, rcond=None)[0]
    
    # Predict accuracy: acc = 1 / (1 + exp(a * nll + b))
    predicted_acc = 1.0 / (1.0 + np.exp(a * predict_nll + b))
    
    return [float(a), float(b), float(predicted_acc)]