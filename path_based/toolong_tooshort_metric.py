import numpy as np
import os
import sys
import networkx as nx
import random

from .. import utils

def toolong_tooshort(G_gt, G_pred, max_node_dist=25, perc_length=0.05, n_paths=1000, seed=None):
    '''
    Too-long too-short metric
    
    J.D. Wegner, J.A. Montoya-Zegarra, and K. Schindler. A
    Higher-Order CRF Model for Road Network Extraction. CVPR
    
    Parameters
    ----------
    G_gt : networkx object
        ground-truth graph
    G_pred : networkx object
        reconstructed graph        
    max_node_dist : float
        a point is conisdered "touching" the graph if it lies within this distance.
        Points that do not touch the graph are not considered endpoints of a path.
    perc_length : float
        maximum length error in percentage a path can have to be considered correct
    n_paths : int
        number of randomly sampled paths 
    seed : int
        random seed for reproducibility
        
    Return
    ------
    percentage of correct, too long, too short and infeasible paths
    '''
    
    if utils.is_empty(G_gt):
        raise ValueError("Ground-truth graph is empty!")
    
    if utils.is_empty(G_pred):
        print("!! Predicted graph is empty !!")
        correct, too_long, too_short, infeasible = 0,0,0,1
        return correct, too_long, too_short, infeasible    
    
    np.random.seed(seed)
    
    # oversampling the graph and working with nodes only is not the optimal
    # solution but it is fast. In the worst case scenario, oversampling with 
    # max_node_dist/3 means that points that are in the region 
    # [0.986*max_node_dist, max_node_dist] may be considered infeasible
    # or correct/matched depending on their position w.r.t the nodes on the graph.
    G_gt = utils.oversampling_graph(G_gt, max_node_dist//3)
    G_pred = utils.oversampling_graph(G_pred, max_node_dist//3)
    
    nodes_gt = list(G_gt.nodes())
    nodes_gt_pos = np.vstack([G_gt.nodes[n]['pos'] for n in G_gt.nodes()])
    nodes_pred = list(G_pred.nodes())
    nodes_pred_pos = np.vstack([G_pred.nodes[n]['pos'] for n in G_pred.nodes()])
    
    correct = 0
    too_long = 0
    too_short = 0
    infeasible = 0

    i = 0
    atempts = 0
    while True:
        
        if i>n_paths:
            return correct/i, too_long/i, too_short/i, infeasible/i

        # pick two nodes from the ground-truth graph
        s = utils.uniform_node_sampling(G_gt, dist_matching=max_node_dist)
        t = utils.uniform_node_sampling(G_gt, dist_matching=max_node_dist)
        if s==t:
            continue

        # find shortest path if it exists otherwise pick two new nodes
        try:
            path_gt = nx.shortest_path(G_gt, s, t) 
            length_gt = utils.length_path(G_gt, path_gt)
            atempts += 1
            if atempts>100*n_paths:
                return correct/i, too_long/i, too_short/i, infeasible/i
        except:
            continue

        i+=1  

        # find corresponding starting node in prediction
        d, idx = utils.find_closest(nodes_gt_pos[nodes_gt.index(s)], nodes_pred_pos)
        if d<max_node_dist:
            s_pred = nodes_pred[idx]
            
            # find corresponding ending node in prediction
            d, idx = utils.find_closest(nodes_gt_pos[nodes_gt.index(t)], nodes_pred_pos)
            if d<max_node_dist:
                t_pred = nodes_pred[idx]

                # find shortest path in prediction
                try:
                    path_pred = nx.shortest_path(G_pred, s_pred, t_pred)
                    length_pred = utils.length_path(G_pred, path_pred)
                except:
                    infeasible += 1
                    continue           

                # compute relative length
                p = (length_gt-length_pred)/length_gt
                if p>perc_length:
                    too_short += 1
                elif p<-perc_length:
                    too_long += 1
                else:
                    correct += 1

            else:
                infeasible += 1                
                continue        
        else:
            infeasible += 1
            continue