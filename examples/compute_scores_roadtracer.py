import os
import sys
import numpy as np
import argparse
import multiprocessing

import metrics_delin as md

base = "."
path_gt = (base+"RoadTracer/data/graphs_output/gt", "*.graph")
path_rt = (base+"RoadTracer/data/graphs_output/roadtracer", "*0.25_0.25.newreconnect.graph")
path_seg = (base+"RoadTracer/data/graphs_output/segmentation/all", "*thr20.*.graph")
path_drm = (base+"RoadTracer/data/graphs_output/deeproadmapper", "*fix.connected.graph")
path_segp = (base+"seg_path/graphs/", "*.graph")
path_drcnn = (base+"deep_rcnn_unet/graphs_filtered/", "*th0.1.graph")  

methods = [["roadtracer", path_rt], 
           ["segmentation", path_seg], 
           ["deeproadmapper", path_drm], 
           ["seg-path", path_segp],
           ["deeprcnn", path_drcnn]]

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
    "vancouver": r4096}

class OPTJ(object):
    
    def __init__(self, th_existing=1, th_snap=25, alpha=100):
        self.th_existing = th_existing
        self.th_snap = th_snap
        self.alpha = alpha
        
    def compute(self, G_gt, G_pred, f_pred):

        f1, precision, recall, \
        tp, pp, ap, \
        matches_g, matches_hg, \
        g_gt_snap, g_pred_snap = md.opt_j(G_gt, G_pred, self.th_existing, self.th_snap, self.alpha)    

        print(os.path.basename(f_pred), 'f1', f1, 'precision', precision, 'recall', recall)
        return os.path.basename(f_pred), {"f1":f1, "precision":precision, "recall":recall, "tp":tp, "pp":pp, "ap":ap}

    def aggregate(self, out):
        
        # compute aggregated score
        tp_aggr = sum([data["tp"] for name,data in out.items()])
        pp_aggr = sum([data["pp"] for name,data in out.items()])
        ap_aggr = sum([data["ap"] for name,data in out.items()])

        f1_aggr, precision_aggr, recall_aggr = md.junction_based.junction_new_metric.compute_scores(tp_aggr, ap_aggr, pp_aggr) 

        return {"f1_aggr":f1_aggr,
                "precision_aggr":precision_aggr, 
                "recall_aggr":recall_aggr,
                "out":out}
    
class CorrCompQual(object):
    
    def __init__(self, slack=8, scale=1/4):
        self.scale = scale
        self.slack = slack*scale
        
    def compute(self, G_gt, G_pred, f_pred):
        
        # render the graphs
        segments = np.array([[G_gt.nodes[s]['pos'], G_gt.nodes[t]['pos']] for s,t in G_gt.edges()])
        gt_s = md.render_segments(segments*self.scale, filename=None, 
                               height=8192*self.scale, width=8192*self.scale, thickness=1)

        segments = np.array([[G_pred.nodes[s]['pos'], G_pred.nodes[t]['pos']] for s,t in G_pred.edges()])
        pred_s = md.render_segments(segments*self.scale, filename=None, 
                                 height=8192*self.scale, width=8192*self.scale, thickness=1)
        
        if np.ndim(gt_s)>2:
            gt_s = gt_s[:,:,0]
            pred_s = pred_s[:,:,0]            

        corr, comp, qual, TP_g, TP_p, FN, FP = md.corr_comp_qual(gt_s, pred_s, slack=self.slack)

        print(os.path.basename(f_pred), "corr",corr, "comp",comp, "qual",qual)
        return os.path.basename(f_pred), {"corr":corr, "comp":comp, "qual":qual, 
                                          "TP_g":TP_g, "TP_p":TP_p, "FN":FN, "FP":FP} 

    def aggregate(self, out):
        
        # compute aggregated score
        TP_g = sum([data["TP_g"] for name,data in out.items()])
        TP_p = sum([data["TP_p"] for name,data in out.items()])
        FN = sum([data["FN"] for name,data in out.items()])
        FP = sum([data["FP"] for name,data in out.items()])

        correctness_aggr, \
        completeness_aggr, \
        quality_aggr = md.pixel_based.corr_comp_qual_metric.compute_scores(TP_g, TP_p, FN, FP)

        return {"correctness_aggr":correctness_aggr, 
                "completeness_aggr":completeness_aggr,
                "quality_aggr": quality_aggr,
                "out":out}   
    
class TooLongTooShort(object):
    
    def __init__(self, d=25, n_paths=1000):
        self.d = d
        self.n_paths = n_paths
        
    def compute(self, G_gt, G_pred, f_pred):

        correct, too_long, too_short, infeasible = md.toolong_tooshort(G_gt, G_pred, 
                                                                       n_paths=self.n_paths, 
                                                                       max_node_dist=self.d)   

        print(os.path.basename(f_pred), 'correct', correct)
        return os.path.basename(f_pred), {"correct":correct, "toolong":too_long, 
                                          "tooshort":too_short, "infeasible":infeasible}

    def aggregate(self, out):
        
        # compute aggregated score
        correct_aggr = np.mean([data["correct"] for name,data in out.items()])
        toolong_aggr = np.mean([data["toolong"] for name,data in out.items()])
        tooshort_aggr = np.mean([data["toolong"] for name,data in out.items()])
        infeasible_aggr = np.mean([data["infeasible"] for name,data in out.items()])

        return {"correct_aggr":correct_aggr, 
                "toolong_aggr":toolong_aggr,
                "tooshort_aggr":tooshort_aggr,
                "infeasible_aggr":infeasible_aggr,
                "out":out} 
    
class OPTP(object):
    
    def __init__(self, d=25):
        self.d = d
        
    def compute(self, G_gt, G_pred, f_pred):

        n_conn_precis, n_conn_recall, \
        n_inter_precis, n_inter_recall, \
        con_prob_precis, con_prob_recall, con_prob_f1 = md.opt_p(G_gt, G_pred, th_snap=self.d)  

        print(os.path.basename(f_pred), "precision", con_prob_precis, "recall", con_prob_recall, "f1", con_prob_f1)
        return os.path.basename(f_pred), {"con_prob_precis":con_prob_precis, 
                                          "con_prob_recall":con_prob_recall,
                                          "con_prob_f1":con_prob_f1}

    def aggregate(self, out):
        
        # compute aggregated score
        connection_probability_precision_aggr = np.mean([data["con_prob_precis"] for name,data in out.items()])
        connection_probability_recall_aggr = np.mean([data["con_prob_recall"] for name,data in out.items()])
        connection_probability_f1_aggr = md.f1_score(connection_probability_precision_aggr,
                                                     connection_probability_recall_aggr)

        return {"connection_probability_precision_aggr":connection_probability_precision_aggr, 
                "connection_probability_recall_aggr":connection_probability_recall_aggr,
                "connection_probability_f1_aggr":connection_probability_f1_aggr,
                "out":out}  
    
class HolesMarbles(object):
    
    def __init__(self, r=300, d=25, spacing=10, N=1000):
        self.r = r
        self.d = d
        self.spacing = spacing
        self.N = N
        
    def compute(self, G_gt, G_pred, f_pred):

        f1, spurious, missings, \
        n_preds_sum, n_gts_sum, \
        n_spurious_marbless_sum, \
        n_empty_holess_sum = md.holes_marbles(G_gt, G_pred, 
                                              spacing=self.spacing, 
                                              dist_limit=self.r,
                                              dist_matching=self.d,
                                              N=self.N,
                                              verbose=False)    

        print(os.path.basename(f_pred), "f1", f1)
        return os.path.basename(f_pred), {"f1":f1, 
                                          "spurious":spurious, 
                                          "missings":missings, 
                                          "n_preds_sum":n_preds_sum,
                                          "n_gts_sum":n_gts_sum, 
                                          "n_spurious_marbless_sum":n_spurious_marbless_sum, 
                                          "n_empty_holes_sum":n_empty_holess_sum}

    def aggregate(self, out):
        
        # compute aggregated score
        n_preds_sum = sum([data["n_preds_sum"] for name,data in out.items()])
        n_gts_sum = sum([data["n_gts_sum"] for name,data in out.items()])
        n_spurious_marbless_sum = sum([data["n_spurious_marbless_sum"] for name,data in out.items()])
        n_empty_holes_sum = sum([data["n_empty_holes_sum"] for name,data in out.items()])

        f1_aggr, spurious_aggr, missing_aggr = md.graph_based.common.compute_scores(n_preds_sum, n_gts_sum,
                                                                                    n_spurious_marbless_sum,
                                                                                    n_empty_holes_sum)
        return {"f1_aggr":f1_aggr, 
                "spurious_aggr":spurious_aggr,
                "missing_aggr":missing_aggr,
                "out":out} 
    
class OPTG(object):
    
    def __init__(self, r=300, d=25, spacing=10, N=1000, matching='greedy'):
        self.r = r
        self.d = d
        self.spacing = spacing
        self.N = N
        self.matching = matching
        
    def compute(self, G_gt, G_pred, f_pred):

        f1, spurious, missings, \
        n_preds_sum, n_gts_sum, \
        n_spurious_marbless_sum, \
        n_empty_holess_sum = md.opt_g(G_gt, G_pred, 
                                      spacing=self.spacing, 
                                      dist_limit=self.r,
                                      dist_matching=self.d,
                                      N=self.N,
                                      matching=self.matching,
                                      verbose=False)    

        print(os.path.basename(f_pred), "f1", f1)
        return os.path.basename(f_pred), {"f1":f1, 
                                          "spurious":spurious, 
                                          "missings":missings, 
                                          "n_preds_sum":n_preds_sum,
                                          "n_gts_sum":n_gts_sum, 
                                          "n_spurious_marbless_sum":n_spurious_marbless_sum, 
                                          "n_empty_holes_sum":n_empty_holess_sum}

    def aggregate(self, out):
        
        # compute aggregated score
        n_preds_sum = sum([data["n_preds_sum"] for name,data in out.items()])
        n_gts_sum = sum([data["n_gts_sum"] for name,data in out.items()])
        n_spurious_marbless_sum = sum([data["n_spurious_marbless_sum"] for name,data in out.items()])
        n_empty_holes_sum = sum([data["n_empty_holes_sum"] for name,data in out.items()])

        f1_aggr, spurious_aggr, missing_aggr = md.graph_based.common.compute_scores(n_preds_sum, n_gts_sum,
                                                                                    n_spurious_marbless_sum,
                                                                                    n_empty_holes_sum)
        return {"f1_aggr":f1_aggr, 
                "spurious_aggr":spurious_aggr,
                "missing_aggr":missing_aggr,
                "out":out}

def run_in_parallel(metric, f_gt, f_pred):
    
    G_gt = md.load_graph_txt(f_gt)
    G_pred = md.load_graph_txt(f_pred)
    
    city = os.path.basename(f_gt).split('_')[0].split('.')[0]
    xmin, ymin = SHIFTS[city]
    xmax, ymax = SHIFTS[city][0]+8192, SHIFTS[city][1]+8192

    G_gt = md.crop_graph(G_gt, xmin, ymin, xmax, ymax)
    G_pred = md.crop_graph(G_pred, xmin, ymin, xmax, ymax)
    
    G_gt = md.shift_graph(G_gt, -xmin, -ymin)
    G_pred = md.shift_graph(G_pred, -xmin, -ymin)
    
    '''
    The borders of the road networks for the method 'segmentation' (roadtracer results) 
    have not been predicted (blank). For this reason, to perform a fair comparison,  
    we remove the borders from all predictions.
    '''
    border = 260
    G_gt = md.crop_graph(G_gt, border, border, 8192-border, 8192-border)
    G_pred = md.crop_graph(G_pred, border, border, 8192-border, 8192-border)    

    return metric.compute(G_gt, G_pred, f_pred)

def main(metric='opt_j', output_file='results.pickle', threads=40):
    
    if metric.lower() in ['opt-j', 'optj', 'opt_j']:
        metric_f = OPTJ()
    elif metric.lower() in ['toolongtooshort', 'toolong-tooshort', 'toolong/tooshort']:
        metric_f = TooLongTooShort()    
    elif metric.lower() in ['opt-p', 'optp', 'opt_p']:
        metric_f = OPTP()  
    elif metric.lower() in ['corrcompqual', 'corr-comp-qual']:
        metric_f = CorrCompQual() 
    elif metric.lower() in ['holesmarbles', 'holes&marbles']:
        metric_f = HolesMarbles() 
    elif metric.lower() in ['opt-g', 'optg' 'opt_g']:
        metric_f = OPTG() 
    else:
        raise ValueError("Unrecognized metric '{}'".format(metric))
    
    filename_graphs_gt = md.find_files(*path_gt)
    cities = [os.path.basename(f).split('.')[0] for f in filename_graphs_gt]
    
    # load filenames and check
    methods_filenames = []
    for method, path in methods:
        filename_graphs = md.find_files(*path)
        assert len(filename_graphs)==15, method
        methods_filenames.append([method, filename_graphs])

    results = {}
    for method, filenames_pred in methods_filenames:
    
        def do_stuff(filename_graphs_pred):
            inputs = []
            for f_pred in filename_graphs_pred:
                city = os.path.basename(f_pred).split('_')[0].split('.')[0]
                f_gt = filename_graphs_gt[cities.index(city)]
                inputs.append((metric_f, f_gt, f_pred))

            pool = multiprocessing.Pool(threads)
            res = pool.starmap(run_in_parallel, inputs)
            pool.close()
            pool.join()

            out = {}
            for name, data in res:
                out[name] = data
                
            aggregated_results = metric_f.aggregate(out)
            
            print("----- {} ------".format(method))
            keys = list(aggregated_results.keys())
            keys.remove("out")
            for key in keys:
                print(key, aggregated_results[key])
            print("---------------")
                
            return aggregated_results

        results[method] = do_stuff(filenames_pred)

        md.pickle_write(output_file, results) 
        
if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')    

    parser = argparse.ArgumentParser()   
    parser.add_argument("--metric", "-m", type=str, required=True, default="OPT-J",
                        help='[\'OPT-J\', \'OPT-P\', \'OPT-G\', \'toolongtooshort\', \'corrcompqual\', \'holesmarbles\']')
    parser.add_argument("--output_file", "-o", type=str, required=True)
    parser.add_argument("--threads", "-t", type=int, required=False)

    args = parser.parse_args()

    main(**vars(args))

# python compute_scores_roadtracer.py -m OPT-J -o "results_opt_j.pickle" --threads 40       