import os
import sys
import numpy as np
import networkx as nx
import time
import copy
import random
import itertools
import scipy
from scipy.spatial.distance import cdist

from .. import utils
from .common import extract_subgraph, compute_quantities, compute_scores
            
def node_matching_hungarian(nodes_gt, nodes_pos_gt, nodes_pred, nodes_pos_pred, dist_match=25):
    if len(nodes_gt)==0 or len(nodes_pred)==0:
        return [],[],[]

    # cap the distance at dist_match
    dist= cdist(nodes_pos_gt, nodes_pos_pred)
    dist[dist>dist_match]=dist_match
    
    # make the distance matrix square - not used in the score
    num_ad=dist.shape[0]-dist.shape[1]
    if num_ad>0 :
        weight=np.concatenate((dist,np.ones((dist.shape[0],num_ad))*dist_match),axis=1)
    elif num_ad<0 :
        weight=np.concatenate((dist,np.ones((-num_ad,dist.shape[1]))*dist_match),axis=0)
    else:
        weight=dist
    weight_orig=np.copy(weight)
    col_matches,row_matches=hungarian.lap(weight)
    
    
    # pairs of matching node indeces
    matches_gt_pred = [(nodes_gt[row_matches[n]],nodes_pred[n]) for n in range(len(nodes_pred)) if weight_orig[row_matches[n]][n] < dist_match ]
    # matched ground truth nodes
    matched_gt_nodes = [nodes_gt[n] for n in range(len(nodes_gt)) if weight_orig[n][col_matches[n]] < dist_match ]
    # matched predicted nodes
    matched_pred_nodes = [nodes_pred[n] for n in range(len(nodes_pred)) if weight_orig[row_matches[n]][n] < dist_match ]

    return  matched_pred_nodes, matched_gt_nodes, matches_gt_pred

def node_matching_greedy(nodes_gt, nodes_pos_gt, nodes_pred, nodes_pos_pred, dist_match=25):
    if len(nodes_gt)==0 or len(nodes_pred)==0:
        return [],[],[]

    dist = cdist(nodes_pos_gt, nodes_pos_pred)

    sortedinds = np.argsort(dist, axis=None)
    xsortedinds, ysortedinds = np.unravel_index(sortedinds, dist.shape)
    
    matched_x=set()
    matched_y=set()
    matches=[]
    matched_gt_nodes=[]
    matched_pred_nodes=[]
    for xi,yi in zip(xsortedinds, ysortedinds):
        if dist[xi][yi]>=dist_match:
            break
            
        if not xi in matched_x and not yi in matched_y:
            matches.append((nodes_gt[xi], nodes_pred[yi]))
            matched_gt_nodes.append(nodes_gt[xi])
            matched_pred_nodes.append(nodes_pred[yi])
            matched_x.add(xi)
            matched_y.add(yi)
            
        if len(matched_x)==min(dist.shape):
            break            

    return matched_pred_nodes, matched_gt_nodes, matches       

def opt_g(G_gt, G_pred, spacing=10, dist_limit=300, dist_matching=25,
          N=1000, matching='greedy', verbose=True, seed=999):
    '''
    OPT-G metric
    
    Leonardo Citraro, Mateusz Kozinski, Pascal Fua
    Towards Reliable Evaluation of Road Network Reconstructions
    ECCV 2020
    
    Parameters
    ----------
    G_gt : networkx object
        ground-truth graph
    G_pred : networkx object
        reconstructed graph        
    spacing : float
        regularly spaced nodes are inserted into the graph. 
        This parameter defines the distance on average among adjacent nodes.
    dist_limit : float
        size (radius) of the subgraphs
    dist_matching : float
        a node in one subgraph is matched to the other if it lies within this distance 
    N : int
        number of subgraphs to evaluate. (stop criteria)
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
    
    if matching == 'greedy':
        node_matching = node_matching_greedy
    elif matching == "hungarian":
        node_matching = node_matching_hungarian
    else:
        raise ValueError("Unrecognized option for 'matching'. Choose 'greedy' or 'hungarian'")
    
    _G_gt = copy.deepcopy(G_gt)
    _G_pred = copy.deepcopy(G_pred)
    
    # add regularly spaced holes and marbles in the graph
    if spacing is not None and spacing>0:
        
        # this metric is more stable if the edges are as long as possible
        _G_gt = utils.simplify_graph_ramer_douglas_peucker(_G_gt, epsilon=5, verbose=verbose, inplace=True)
        _G_pred = utils.simplify_graph_ramer_douglas_peucker(_G_pred, epsilon=5, verbose=verbose, inplace=True)       
        
        # add regularly spaced nodes in each edge
        _G_gt = utils.oversampling_graph(_G_gt, spacing=spacing)
        _G_pred = utils.oversampling_graph(_G_pred, spacing=spacing)
        
    n_spurious_marbless, n_empty_holess, n_preds, n_gts = [],[],[],[]
    for _ in range(N):
        
        if random.random()<0.5:
            
            # pick a node from gt graph
            random_node = utils.uniform_node_sampling(_G_gt, dist_matching=dist_matching)
            
            # crop circular subgraphs
            nodes_gt, nodes_pred = extract_subgraph(_G_gt, random_node, _G_pred, dist_limit, dist_matching)
        else:
            
            # pick a node from reconstructed graph
            random_node = utils.uniform_node_sampling(_G_pred, dist_matching=dist_matching) 
                    
            # crop circular subgraphs
            nodes_pred, nodes_gt = extract_subgraph(_G_pred, random_node, _G_gt, dist_limit, dist_matching)            

        nodes_pos_gt = np.array([_G_gt.nodes[n]['pos'] for n in nodes_gt])
        nodes_pos_pred = np.array([_G_pred.nodes[n]['pos'] for n in nodes_pred])        

        # match the nodes from gt to prediction
        matches_pred,_,__ = node_matching(nodes_gt, nodes_pos_gt, nodes_pred, nodes_pos_pred, dist_matching)

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