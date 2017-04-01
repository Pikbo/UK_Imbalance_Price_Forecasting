# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:46:09 2017

@author: adamd
"""

import pandas as pd

# function to calculate in sample mean absolute error, naive MASE, predicted MASE, naive prediction and predicted absolute scaled error
# pass 1d arrays and seasonality as a integer
def MASE(y_act,y_pred,seasonality):
    
    y_pred = pd.Series(y_pred)
    y_act = pd.Series(y_act)    
    
    y_pred_cut = y_pred[seasonality:]
    y_act_cut = y_act[seasonality:]
    y_naive = y_act.shift(seasonality) 
    y_naive_cut = y_naive[seasonality:]
    # naive MASE
    naive_error = (y_act_cut - y_naive_cut)
    naive_abs_error = naive_error.abs()
    in_sample_MAE = naive_abs_error.mean()
    naive_abs_scaled_error = naive_abs_error / in_sample_MAE
    naive_MASE = naive_abs_scaled_error.mean()
    # prediction MASE
    pred_error = y_act_cut - y_pred_cut
    pred_error_abs = pred_error.abs()
    pred_abs_scaled_error = pred_error_abs / in_sample_MAE
    pred_MASE = pred_abs_scaled_error.mean()
    # return(in_sample_MAE,naive_MASE,pred_MASE,y_naive,pred_abs_scaled_error)
    return (pred_MASE)    
