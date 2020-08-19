import os
import sys
import imageio
import numpy as np

import metrics_delin as md

r4096 = (-4096, -4096)
SHIFTS = {
    "toronto": r4096,
    "la": r4096,
    "new york": r4096,
    "boston": (4096, -4096),
    "chicago": (-4096, -8192),
    "amsterdam": r4096,
    "denver": r4096,
    "kansas city": r4096,
    "montreal": r4096,
    "paris": r4096,
    "pittsburgh": r4096,
    "saltlakecity": r4096,
    "san diego": r4096,
    "tokyo": r4096,
    "vancouver": r4096,
    "columbus": (-4096, -8192),
    "minneapolis": (-4096, -8192), 
    "nashville": (-4096, -8192)}

print("loading the graphs")
G_gt   = md.load_graph_txt("vancouver.graph")
G_pred = md.load_graph_txt("vancouver.0.25_0.25.newreconnect.graph")

city = "vancouver"
#h,w = 8192, 8192
h,w = 1000, 1000 # smaller crop to speed up this script
xmin, ymin = SHIFTS[city]
xmax, ymax = SHIFTS[city][0]+h, SHIFTS[city][1]+w 

print("cropping and shifting the graphs")
G_gt = md.crop_graph(G_gt, xmin, ymin, xmax, ymax) 
G_pred = md.crop_graph(G_pred, xmin, ymin, xmax, ymax)

G_gt = md.shift_graph(G_gt, -xmin, -ymin)
G_pred = md.shift_graph(G_pred, -xmin, -ymin) 

# --------------------------------------------------
f1, precision, recall, \
tp, pp, ap, \
matches_g, matches_hg, \
g_gt_snap, g_pred_snap = md.opt_j(G_gt, 
                                  G_pred, 
                                  th_existing=1, 
                                  th_snap=25, 
                                  alpha=100)
print("OPT-J:          precision={:0.3f} recall={:0.3f} f1={:0.3f}\n".format(precision, recall, f1))

# --------------------------------------------------
scale = 1/4
segments = np.array([[G_gt.nodes[s]['pos'], G_gt.nodes[t]['pos']] for s,t in G_gt.edges()])
gt_s = md.render_graph(segments*scale, 
                       filename=None, 
                       height=h*scale, 
                       width=w*scale, 
                       thickness=1)

segments = np.array([[G_pred.nodes[s]['pos'], G_pred.nodes[t]['pos']] for s,t in G_pred.edges()])
pred_s = md.render_graph(segments*scale, 
                         filename=None, 
                         height=h*scale, 
                         width=w*scale, 
                         thickness=1)        

corr, comp, qual, TP_g, TP_p, FN, FP = md.corr_comp_qual(gt_s, 
                                                         pred_s, 
                                                         slack=2)
print("Corr-Comp-Qual: corr={:0.3f} comp={:0.3f} qual={:0.3f}\n".format(corr, comp, qual))

# --------------------------------------------------
correct, too_long, too_short, infeasible = md.toolong_tooshort(G_gt, G_pred, 
                                                               n_paths=50, 
                                                               max_node_dist=25)
print("2Long-2Short:   correct={:0.3f} 2l+2s={:0.3f} inf={:0.3f}\n".format(correct, too_long+too_short, infeasible))

# --------------------------------------------------
n_conn_precis, n_conn_recall, \
n_inter_precis, n_inter_recall, \
con_prob_precis, con_prob_recall, con_prob_f1 = md.opt_p(G_gt, G_pred)
print("OPT-P:          con_prob_precis={:0.3f} con_prob_recall={:0.3f} con_prob_f1={:0.3f}\n".format(con_prob_precis, con_prob_recall, con_prob_f1))

# --------------------------------------------------
f1, spurious, missings, \
n_preds_sum, n_gts_sum, \
n_spurious_marbless_sum, \
n_empty_holess_sum = md.holes_marbles(G_gt, G_pred, 
                                      spacing=20, 
                                      dist_limit=300,
                                      dist_matching=25,
                                      N=50,
                                      verbose=False) 
print("Hole-Marbles:   spurious={:0.3f} missings={:0.3f} f1={:0.3f}\n".format(spurious, missings, f1))

# --------------------------------------------------
f1, spurious, missings, \
n_preds_sum, n_gts_sum, \
n_spurious_marbless_sum, \
n_empty_holess_sum = md.opt_g(G_gt, G_pred, 
                              spacing=20, 
                              dist_limit=300,
                              dist_matching=25,
                              N=50,
                              verbose=False) 
print("OPT-G:          spurious={:0.3f} missings={:0.3f} f1={:0.3f}\n".format(spurious, missings, f1))

# --------------------------------------------------