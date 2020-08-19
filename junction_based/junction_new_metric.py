import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import KDTree

import time

from .. import utils

def matching_with_snapping(junctions_g, junctions_pos_g, G, H, th_existing=1, th_snap=25, alpha=100):
    
    # snap the node of graph G into graph H if they are sufficiently close
    H_snap, corresps = utils.snap_points_to_graph(H, junctions_pos_g, th_existing=th_existing, th_snap=th_snap)
    
    # get control nodes
    junctions_h = [n for n in H_snap.nodes() 
                    if utils.is_control_nodes(H_snap, n) or n in corresps]
    junctions_pos_h = np.array([H_snap.nodes[n]['pos'] for n in junctions_h])
    
    # find some candidates control points to match
    dists_all, idxs_all = KDTree(junctions_pos_h).query(junctions_pos_g, k=5)
    candidates_all = []
    for dists, idxs in zip(dists_all, idxs_all):
        candidates_all.append([(d,junctions_h[i]) for d,i in zip(dists,idxs)])    
        
    # find a node in H that correspond at best to each node in G
    matches = {}
    matches_pos = []
    seen = []
    for node_g, candidates in zip(junctions_g, candidates_all):
        
        order_g = len(G.edges(node_g))
        
        best_cost_h = np.inf
        best_node_h = None 
        # try to find the best control points
        for dist, node_h in candidates:
            
            # consider close candidates only
            if dist<=th_snap:
            
                # during the matching, we add some priviledges
                # to the nodes with similar order.
                order_h = len(H_snap.edges(node_h))

                cost = dist + alpha*np.abs(order_g-order_h)
                if cost<best_cost_h and node_h not in seen:
                    best_cost_h = cost
                    best_node_h = node_h

        matches[node_g] = best_node_h       
        seen.append(best_node_h)
        
    return matches, H_snap

def twoway_matching(G, H, th_existing=0.2, th_snap=0.5, alpha=100):

    junctions_g = [n for n in G.nodes() if utils.is_control_nodes(G, n)]
    junctions_g_pos = np.array([G.nodes[n]['pos'] for n in junctions_g])
    matches_g, H_snap = matching_with_snapping(junctions_g, junctions_g_pos, G, H,
                                               th_existing, th_snap, alpha)

    junctions_h = [n for n in H.nodes() if utils.is_control_nodes(H, n)]
    junctions_hg = [n for n in junctions_h if n not in matches_g.values()]
    if len(junctions_hg)>0:
        junctions_hg_pos = np.array([H.nodes[n]['pos'] for n in junctions_hg])

        matches_hg, G_snap = matching_with_snapping(junctions_hg, junctions_hg_pos, H, G,
                                                    th_existing, th_snap, alpha)
    else:
        matches_hg = {}
        G_snap = G
    
    return matches_g, matches_hg, G_snap, H_snap # matches from graph G to graph H with snapped nodes as well

def f1_score(precision, recall):
    return 2*(precision*recall)/(precision+recall+1e-12)

def compute_scores(tp, ap, pp):
    
    precision = tp/(pp+1e-12)
    recall = tp/(ap+1e-12)
    f1 = f1_score(precision, recall)
    
    return f1, precision, recall

def opt_j(G_gt, G_pred, th_existing=10, th_snap=25, alpha=100):
    '''
    OPT-J metric
    
    Leonardo Citraro, Mateusz Kozinski, Pascal Fua
    Towards Reliable Evaluation of Road Network Reconstructions
    ECCV 2020
    
    Parameters
    ----------
    G_gt : networkx object
        ground-truth graph
    G_pred : networkx object
        reconstructed graph
    th_snap : float
        a point is snapped into the graph if its distance from the 
        closest edge is less than th_snap 
    th_existing : float
        during the snapping prcedure, an additional node is inserted into an edge only if 
        none of endpoints of the edge are within th_existing
    alpha : float
        parameter that encourage matching two nodes that have similar order
        
    Return
    ------
    matches_g : dict
        matched nodes from G_gt to G_pred_snap
    matches_hg : dict
        remaining matches from G_pred to G_gt_snap
    g_gt_snap : networkx object
        G_gt with the added nodes
    g_pred_snap : networkx object
        G_pred with the added nodes
    '''
    
    if utils.is_empty(G_gt):
        raise ValueError("Ground-truth graph is empty!")
    
    if utils.is_empty(G_pred):
        print("!! Predicted graph is empty !!")
        f1, precision, recall = 0,0,0
        tp, pp, ap = 0,0,0
        matches_g, matches_hg = {},{}
        g_gt_snap, g_pred_snap = G_gt, G_pred
        return f1, precision, recall, tp, pp, ap, matches_g, matches_hg, g_gt_snap, g_pred_snap
    
    matches_g, matches_hg, g_gt_snap, g_pred_snap = twoway_matching(G_gt, G_pred,
                                                                    th_existing, 
                                                                    th_snap, 
                                                                    alpha=alpha)
    tp = 0
    pp = 0
    ap = 0
    for node_gt, node_pred in matches_g.items():
        
        order_gt = len(g_gt_snap.edges(node_gt))           
        if node_pred is not None:
            order_pred = len(g_pred_snap.edges(node_pred))
        else:
            order_pred = 0

        tp += np.minimum(order_gt, order_pred)
        pp += order_pred
        ap += order_gt
        
    for node_pred, node_gt in matches_hg.items():
        if node_gt is not None:
            order_gt = len(g_gt_snap.edges(node_gt))
        else:
            order_gt = 0
        
        order_pred = len(g_pred_snap.edges(node_pred))

        tp += np.minimum(order_gt, order_pred)
        pp += order_pred
        ap += order_gt        
        
    f1, precision, recall = compute_scores(tp, ap, pp)  
    
    return f1, precision, recall, tp, pp, ap, matches_g, matches_hg, g_gt_snap, g_pred_snap