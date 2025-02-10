import numpy as np

def RSE(pred, true):
    true = true[:, -pred.shape[1]:, :]  
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    true = true[:, -pred.shape[1]:, :]
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    true = true[:, -pred.shape[1]:, :]  
    return np.mean(np.abs(pred - true)) * 0.1


def MSE(pred, true):
    true = true[:, -pred.shape[1]:, :] 
    return np.mean((pred - true) ** 2) * 0.01


def RMSE(pred, true):
    true = true[:, -pred.shape[1]:, :] 
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    true = true[:, -pred.shape[1]:, :]  
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    true = true[:, -pred.shape[1]:, :]  
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    true = true[:, -pred.shape[1]:, :]  

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def simple_metric(pred, true):
    true = true[:, -pred.shape[1]:, :]  

    mae = MAE(pred, true)
    mse = MSE(pred, true)

    return mae, mse

