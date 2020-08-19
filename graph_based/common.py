import numpy as np
import networkx as nx

from .. import utils

def squared_dist(p1, p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2

def dfs_nodes_pos(G, source, dist_limit=100):
    
    squared_dist_limit = dist_limit**2
    p_start = G.nodes[source]['pos']
    
    nodes = [source]
    visited = set()
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack = [(start, utils.is_intersection(G, start), 
                  squared_dist(p_start, p_start), iter(G[start]))]
        yield start
        while stack:
            parent, parent_is_int, parent_dist, children = stack[-1]
            
            try:
                child = next(children)
                if child not in visited:
                    child_dist = squared_dist(p_start, G.nodes[child]['pos'])
                    
                    # on the intersections check if nodes go away from start point
                    # otherwise skip them
                    if parent_is_int and child_dist<parent_dist*0.75:
                        visited.add(child)
                        continue
                    #if child_dist<parent_dist:
                    #    continue
                    
                    yield child
                    visited.add(child)
                    if child_dist <= squared_dist_limit:
                        stack.append((child, utils.is_intersection(G, child), 
                                      child_dist, iter(G[child])))
            except StopIteration:
                stack.pop()

def extract_subgraph(G_gt, node_gt, G_pred, dist_limit, dist_matching):   
    
    nodes_pred = list(G_pred.nodes())
    nodes_pos_pred = np.array([G_pred.nodes[n]['pos'] for n in nodes_pred])    
       
    pos_gt = G_gt.nodes[node_gt]['pos']
    if not isinstance(pos_gt, np.ndarray):
        pos_gt = np.array(pos_gt)
    pos_gt = pos_gt.reshape(1,2)

    nodes_gt_sub = list(dfs_nodes_pos(G_gt, node_gt, dist_limit))

    dists = np.linalg.norm(nodes_pos_pred-pos_gt, axis=1)
    idx_min = np.argmin(dists)
    if dists[idx_min]>dist_matching:
        return nodes_gt_sub, []

    n_pred = nodes_pred[idx_min]
    nodes_pred_sub = list(dfs_nodes_pos(G_pred, n_pred, dist_limit))
    
    return nodes_gt_sub, nodes_pred_sub               

def compute_scores(n_pred, n_gt, n_spurious_marbles, n_empty_holes, eps=1e-12):
    
    spurious = (n_spurious_marbles)/(n_pred+eps) # precision
    missing = (n_empty_holes)/(n_gt+eps) # recall
    
    f1 = 2*((1-spurious)*(1-missing))/((1-spurious)+(1-missing)+eps)
    
    return f1, spurious, missing

def compute_quantities(nodes_gt, nodes_pred, matches_pred, eps=1e-12):
    
    n_pred = len(nodes_pred)
    n_gt = len(nodes_gt)
    n_matched_holes = len(list(filter(None, matches_pred)))
    n_matched_marbles = len(set(list(filter(None, matches_pred)))) # use set because a marbel can be matched twice
    n_spurious_marbles = n_pred-n_matched_marbles
    n_empty_holes = n_gt-n_matched_holes    
    
    return n_pred, n_gt, n_matched_holes, n_matched_marbles, n_spurious_marbles, n_empty_holes