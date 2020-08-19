import os
import sys
import numpy as np
import networkx as nx
import time
import copy
import random
import multiprocessing
import itertools
from scipy.spatial.distance import cdist
import scipy

from .. import utils
from .common import extract_subgraph, compute_quantities, compute_scores

def naive_node_matching(nodes_gt, nodes_pos_gt, nodes_pred, nodes_pos_pred, dist_match=25):
    if len(nodes_gt)==0 or len(nodes_pred)==0:
        return []
    
    dist_matrix = cdist(nodes_pos_gt, nodes_pos_pred)
    idx_mins = dist_matrix.argmin(axis=1)
    dists = dist_matrix[np.arange(len(nodes_gt)), idx_mins]  
    nodes_pred_sel = [nodes_pred[i] for i in idx_mins]
    matches_pred = [n if d<dist_match else None for n,d in zip(nodes_pred_sel,dists)] 
    
    return matches_pred

def holes_marbles(G_gt, G_pred, spacing=10, dist_limit=300, dist_matching=25, 
                  N=1000, verbose=True, seed=999):
    '''
    Holes and Marbles metric
    
    James Biagioni and Jakob Eriksson
    Inferring Road Maps from Global Positioning System Traces
    
    Parameters
    ----------
    G_gt : networkx object
        ground-truth graph
    G_pred : networkx object
        reconstructed graph        
    spacing : float
        regularly spaced nodes are added in the graph. spacing will be
        the average distance between adjacent ones
    dist_limit : float
        radius of the subgraphs
    dist_matching : float
        distance defining if two nodes match
    N : int
        this function returns when number of subgraph that have been extracted is >N
    seed : int
        random seed for reproducibility
        
    Return
    ------
    the results for this sample and the quantities for the aggregation
    '''    
    
    if utils.is_empty(G_gt):
        raise ValueError("Ground-truth graph is empty!")
    
    if utils.is_empty(G_pred):
        print("!! Predicted graph is empty !!")

    np.random.seed(seed)
    
    _G_gt = copy.deepcopy(G_gt)
    _G_pred = copy.deepcopy(G_pred)    

    # this metric is more stable if the edges are as long as possible
    _G_gt = utils.simplify_graph_ramer_douglas_peucker(_G_gt, epsilon=5, verbose=verbose, inplace=True)
    _G_pred = utils.simplify_graph_ramer_douglas_peucker(_G_pred, epsilon=5, verbose=verbose, inplace=True)

    # add regularly spaced nodes in each edge
    _G_gt = utils.oversampling_graph(_G_gt, spacing=spacing)
    _G_pred = utils.oversampling_graph(_G_pred, spacing=spacing)       
        
    n_spurious_marbless, n_empty_holess, n_preds, n_gts = [],[],[],[]
    for _ in range(N):
        
        # pick a node
        random_node = utils.uniform_node_sampling(_G_gt, dist_matching=dist_matching)
                
        # crop circular subgraphs
        nodes_gt, nodes_pred = extract_subgraph(_G_gt, random_node, _G_pred, dist_limit, dist_matching)

        nodes_pos_gt = np.array([_G_gt.nodes[n]['pos'] for n in nodes_gt])
        nodes_pos_pred = np.array([_G_pred.nodes[n]['pos'] for n in nodes_pred])        

        # match the nodes from gt to prediction
        matches_pred = naive_node_matching(nodes_gt, nodes_pos_gt, nodes_pred, nodes_pos_pred, dist_matching)
        
        # compute various quantities. These should be summed across the whole 
        # dataset to compute the aggregated results
        n_pred, n_gt, \
        n_matched_holes, n_matched_marbles, \
        n_spurious_marbles, n_empty_hole = compute_quantities(nodes_gt, nodes_pred, matches_pred)  

        n_spurious_marbless.append(n_spurious_marbles)
        n_empty_holess.append(n_empty_hole)
        n_preds.append(n_pred)
        n_gts.append(n_gt)
        
    # quantities for aggregation
    n_spurious_marbless_sum = sum(n_spurious_marbless)
    n_empty_holess_sum = sum(n_empty_holess)
    n_preds_sum = sum(n_preds)
    n_gts_sum = sum(n_gts) 
        
    # results for this sample
    f1_aggr, spurious_aggr, missing_aggr = compute_scores(n_preds_sum, n_gts_sum, 
                                                          n_spurious_marbless_sum, n_empty_holess_sum)
            
    return f1_aggr, spurious_aggr, missing_aggr, \
            n_preds_sum, n_gts_sum, n_spurious_marbless_sum, n_empty_holess_sum
