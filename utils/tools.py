# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:57:13 2023

@author: ab978
"""

import torch, pdb

def back_to_matcov(y,y_x):
    # Initialize the covariance matrix with zeros
    cov = torch.diag(y_x.reshape(-1,))
    # Create a mask for the upper triangle of the covariance matrix
    mask = torch.triu(torch.ones_like(cov), diagonal=1).bool()
    # Fill the upper triangle of the covariance matrix with the covariances from edge_weights_pred
    cov[mask] = y.flatten()
    # Mirror the upper triangle to the lower triangle to make the matrix symmetric
    cov = cov + cov.T - torch.diag(cov.diag())
    
    return cov

def get_mean_std(dataset):

    # Initialize variables to accumulate sum and count
    sum_combined = 0
    total_count = 0
    
    # Iterate over the dataset
    for data in dataset:
        x = data.x
        edge_attr = data.edge_attr
        
        # Flatten the tensors and concatenate them
        combined = torch.cat((x.flatten(), edge_attr.flatten()))
        
        # Update the sum and count
        sum_combined += torch.sum(combined)
        total_count += combined.numel()
    
    # Calculate the average
    avg_combined = sum_combined / total_count
    
    # Calculate the standard deviation
    sum_squared_diff = 0
    for data in dataset:
        x = data.x
        edge_attr = data.edge_attr
        
        # Flatten the tensors and concatenate them
        combined = torch.cat((x.flatten(), edge_attr.flatten()))
        
        # Calculate the squared difference from the average
        squared_diff = torch.pow(combined - avg_combined, 2)
        
        # Update the sum of squared differences
        sum_squared_diff += torch.sum(squared_diff)
    
    # Calculate the standard deviation
    std_combined = torch.sqrt(sum_squared_diff / total_count)
    
    return avg_combined.item(),std_combined.item()
