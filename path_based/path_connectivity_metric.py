import numpy as np
import networkx as nx
import random
import time 
import enum

from .. import utils

def deduplicate_nodes(g):
    # remove spatially coincident nodes from a graph
    pos2node={}
    nodes2del=[]
    for n in g:
        pos=g.nodes[n]['pos']
        if pos in pos2node:
            duplicated_node=pos2node[pos]
            for m in g[n]:
                if m!=duplicated_node: # avoid self-loops
                    g.add_edge(duplicated_node,m)
            nodes2del.append(n)
        else:
            pos2node[pos]=n
    for n in nodes2del:
        g.remove_node(n)
    
    return len(nodes2del)

def closest_points_on_edges(edges,points):
    s=edges[:,np.newaxis,0:2]
    t=edges[:,np.newaxis,2:4]
    pts=points[np.newaxis,:,:]

    # the formulas follow from an analytical solution to a quadrativ equation
    # the optimal alpha=[(-px+sx)(sx-tx)+(-py+sy)(sy-ty)]/[(sx-tx)^2+(sy-ty)^2]
    # where (s,t) forms the edge and p is the point

    stdif=s-t
    alpha_den=np.sum(stdif**2,axis=2)
    alpha_num=np.sum((-pts+s)*stdif,axis=2)
    alpha=alpha_num/alpha_den
    alpha[alpha>1]=1
    alpha[alpha<0]=0

    closestPointOnEdge=s-stdif*alpha[:,:,np.newaxis] # s*(1-alpha)+t*alpha
    dist2=np.sum((pts-closestPointOnEdge)**2,axis=2)

    return alpha, closestPointOnEdge, dist2

def get_candidate_path_node_positions_5_nogpu(g,path,thresholdExisting,thresholdSnap):
    # g is a networkx.Graph
    # p is a nX2 array
    t0=-time.time()
    thSnap2=thresholdSnap**2
    thExst2=thresholdExisting**2
    
    ind2edge=list(g.edges) #snap_points_to_graph.map_inds_2_edges(g)
    
    edges=np.array([g.nodes[s]['pos']+g.nodes[t]['pos'] for s,t in g.edges])
    alpha,closestPtsOnEdges,dist2=closest_points_on_edges(edges,path)
    t0+=time.time()
    
    t1=-time.time()
    alpha[dist2>=thSnap2]=10 # this is a magick value instead of an indicater dist2>=thSnap2
    inds=np.argsort(alpha,axis=1)
    t1+=time.time()
    
    t2=-time.time()
    largest_node=max(g.nodes)
    cand_positions=[[(0,)] for n in path]
    t2+=time.time()
    for ei in range(len(edges)):
        t2_s=time.time()
        removeEdge=False
        e=ind2edge[ei]
        de=dist2[ei]
        ce=closestPtsOnEdges[ei]
        ie=inds[ei]
        ae=alpha[ei]
        prev_node=e[0]
        prev_pos=g.nodes[prev_node]['pos']
        last_pos=g.nodes[e[1]]['pos']
        i=0
        while i<len(ie) and ae[ie[i]] <= 1 and \
              (ce[ie[i]][0]-last_pos[0])**2+(ce[ie[i]][1]-last_pos[1])**2 > thExst2:
            # while far enough from edge end point
            ii=ie[i]
            pos=ce[ii]
            d2=(pos[0]-prev_pos[0])**2+(pos[1]-prev_pos[1])**2
            if d2>thExst2:
                largest_node+=1
                g.add_node(largest_node,pos=(pos[0],pos[1]))
                g.add_edge(prev_node,largest_node)
                prev_node=largest_node
                prev_pos=pos
                removeEdge=True
            cand_positions[ii].append((de[ii],prev_node,prev_pos))
            i+=1
            
        while i<len(ie) and ae[ie[i]]<=1: # map the ones that are close to edge end to existing graph node
            ii=ie[i]
            cand_positions[ii].append((de[ii],e[1],last_pos,))
            i+=1

        if removeEdge:
            g.add_edge(prev_node,e[1])
            g.remove_edge(e[0],e[1])
            
    #t1+=time.time()
    return cand_positions,t0,t1,t2

def get_pairwise_costs_3(g,path,cand_positions,threshold,disconnectedPenaltyFactor,shortestPathCutoff):
    ind2node=list(g.nodes)
    nodes=np.array([ g.nodes[n]['pos'] for n in g.nodes ])
    edges=np.array([ list(path[k-1])+list(path[k]) for k in range(1,len(path))])
    #a,c,dist2=snap_points_to_graph.closest_points_on_edges(edges,nodes)
    a,c,dist2=closest_points_on_edges(edges,nodes)
    #print(dist2.shape)
    #print(cand_positions)
    pairwiseCosts=[]
    allowedCombinations=[]
    sps=[]
    sptime=0
    sptim2=0
    th2=threshold**2
    for i in range(1,len(cand_positions)):
        #sptime_start=time.time()
        #h=g.copy()
        #for j in np.nonzero(segm_node_dist2>=th2)[0]:
        #    h.remove_node(ind2node[j])
        #sptime+=time.time()-sptime_start
        sptime_start=time.time()
        segm_node_dist2=dist2[i-1]
        h=g.__class__()
        #selected_nodes=[ind2node[k] for k in np.nonzero(segm_node_dist2<th2)[0]]
        #for j in selected_nodes:
        #    h.add_node(j)
        #    for k in selected_nodes:
        #        if (j,k) in g.edges:
        #            h.add_edge(j,k)
        selected_node_inds=np.nonzero(segm_node_dist2<th2)[0]
        for k in range(len(selected_node_inds)):
            j=ind2node[selected_node_inds[k]]
            h.add_node(j)
            for l in range(k+1,len(selected_node_inds)):
                m=ind2node[selected_node_inds[l]]
                if (j,m) in g.edges:
                    h.add_edge(j,m)
        sptime+=time.time()-sptime_start
        
        
        sptim2_start=time.time()
        shortestPaths=dict(nx.all_pairs_shortest_path(h,cutoff=shortestPathCutoff))
        sps.append({})
        
        cp=cand_positions[i]
        cprev=cand_positions[i-1]
        discPenalty=np.sum((path[i]-path[i-1])**2)*disconnectedPenaltyFactor
        allowedCombinations.append(np.zeros((len(cprev),len(cp))))
        pairwiseCosts.append(np.ones((len(cprev),len(cp)))*discPenalty)
        ci=0
        for curr in cp[1:]:
            ci+=1
            pi=0
            for prev in cprev[1:]:
                pi+=1
                if prev[1] in shortestPaths and curr[1] in shortestPaths[prev[1]] :
                    pairwiseCosts[-1][pi,ci]=0
                    allowedCombinations[-1][pi,ci]=True
                    sps[-1][(pi,ci)]=shortestPaths[prev[1]][curr[1]]
        sptim2+=time.time()-sptim2_start
        
    return pairwiseCosts,allowedCombinations,sps, sptime, sptim2

def viterbi_2(candPositions, pairwiseCosts):
    # index 0 means the node is not matched
    cost=[]
    cost.append( np.array( [ a[0] for a in candPositions[0] ]) )
    prev=[]
    for i in range(1,len(candPositions)):
        prevCosts=cost[-1]
        pc=pairwiseCosts[i-1]
        c=prevCosts.reshape(-1,1)+pc
        minPrevCost=np.amin(c,axis=0)
        indMinPrevCost=np.argmin(c,axis=0)
        curr_cost=np.array([j[0] for j in candPositions[i]])
        cost.append(curr_cost+minPrevCost)
        prev.append(indMinPrevCost)
    last_prev=np.argmin(cost[-1])
    y=[last_prev]
    for p in reversed(prev):
        y.append(p[y[-1]])
    
    y_r=list(reversed(y))
    
    return y_r

class NodeState(enum.IntEnum):
    unseen=0
    onPath=1
    offPath=2
    
class EdgeState(enum.IntEnum):
    unseen=0
    onPath=1
    offPath=2
    
def sample_chain(g):
    for n in g.nodes:
        g.nodes[n]['state']=NodeState.unseen
    for e in g.edges:
        g.edges[e]['state']=EdgeState.unseen
        
    endPoints=[n for n in g.nodes if len(g[n])==1]
    if len(endPoints)>0:
        startingNode=endPoints[random.randrange(len(endPoints))]
    else:
        nds=[n for n in g.nodes]
        startingNode=nds[random.randrange(len(nds))]
    
    g.nodes[startingNode]['state']=NodeState.onPath
    chain=[startingNode]# represents the current chai
    loop=[startingNode] # returned if no open-ended chain is found
    
    while len(chain)>0:
        n=chain[-1]
        # if n has one edge onPath and no other edges
        if len(g[n])==1 and g.edges[(n,list(g.neighbors(n))[0])]['state']==EdgeState.onPath :
            return chain, True # found a chain
        else:
            neighborsOnUnseenEdges=[m for m in g.neighbors(n) if g.edges[(m,n)]['state']==EdgeState.unseen and n!=m]
            if len(neighborsOnUnseenEdges)>0:
                m=neighborsOnUnseenEdges[random.randrange(len(neighborsOnUnseenEdges))]
                if g.nodes[m]['state']==NodeState.onPath: # loop found, do not descend to m
                    loop=chain.copy()
                    g.edges[(n,m)]['state']=EdgeState.offPath
                elif g.nodes[m]['state']==NodeState.offPath: # do not descend to m
                    g.edges[(n,m)]['state']=EdgeState.offPath
                elif g.nodes[m]['state']==NodeState.unseen: # descend to m
                    g.edges[(n,m)]['state']=EdgeState.onPath
                    g.nodes[n]['state']=NodeState.onPath
                    chain.append(m)
            else: # backtrack
                if len(chain)>1:
                    g.edges[(chain[-2],n)]['state']=EdgeState.offPath
                g.nodes[n]['state']=NodeState.offPath
                chain.pop(-1)
                
    return loop, False    

def decompose_graph_to_paths(g):
    paths=[]
    while len(g)>0:
        p,_=sample_chain(g)
        ip=iter(p)
        prev=next(ip)
        for n in ip:
            g.remove_edge(prev,n)
            if len(g[prev])==0:
                g.remove_node(prev)
            prev=n
        if len(g[prev])==0:
            g.remove_node(prev)
        paths.append(p)
    
    return paths

def break_down_segments(path,lthr):
    thr2=lthr**2
    newpath=list(path)
    curr_pos=0
    while curr_pos<len(newpath)-1:
        currnode=newpath[curr_pos]
        nextnode=newpath[curr_pos+1]
        l=np.sum((currnode-nextnode)**2)
        if l>thr2:
            newnode=0.5*(currnode+nextnode)
            newpath.insert(curr_pos+1,newnode)
        else:
            curr_pos+=1
            
    return np.array(newpath)

def calcPathLength(p):
    l=0
    ni=iter(p)
    prev=next(ni)
    for curr in ni:
        l+=((curr[0]-prev[0])**2+(curr[1]-prev[1])**2)**0.5
        prev=curr
    return l
    
def new2l2s_v0(g, h, th_existing=10, th_snap=25):

    deduplicate_nodes(g)
    deduplicate_nodes(h)

    pathSegmThreshold=25
    connectionEdgeDistThreshold=25
    disconnectionPenaltyFactor=10000
    maxInspectedNodes=5
    thE=th_existing
    thS=th_snap
    
    # decompose graph h into paths
    paths=decompose_graph_to_paths(h.copy())
    
    n_paths=len(paths)
    n_interruptions=0
    pos_interruptions=[]
    connection_probability=[]
    n_connected=0
    tot_lenght = 0
    path_lenghts = []

    if len(h.nodes)==0:
        n_connected=0
        n_interruptions=0
        connection_probability=1
        pos_interruptions=[]
        npaths=0
        return n_connected, n_interruptions, connection_probability
    
    # for each path
    max_node_before_snapping=1
    if len(g.nodes)>0:
        max_node_before_snapping=max(g.nodes)
    for p in paths:
        tt0_start=time.time()
        if len(list(g.edges()))==0:
            print("the graph has no more edges")
            print(g.nodes)
            break
        if len(p)<2:
            print("pth of size 1 encountered")
            print(p)
            n_paths-=1
            continue
        
        # get node positions
        path=np.array([h.nodes[n]['pos'] for n in p])
        
        # break the path into short segments
        newpath=break_down_segments(path,pathSegmThreshold)
        
        # get candidate snapped node positions
        cand_positions,post0,post1,post2=get_candidate_path_node_positions_5_nogpu(g,newpath,thE,thS)
        
        # get pairwise costs
        pairwiseCosts,allowedConnections,shortestPaths,sptime,sptim2=get_pairwise_costs_3(g,newpath,cand_positions,
                                                            connectionEdgeDistThreshold,
                                                            disconnectionPenaltyFactor,
                                                            maxInspectedNodes)
        ac=allowedConnections
        sp=shortestPaths
        
        # align path to graph
        alignedPathNodes=viterbi_2(cand_positions,pairwiseCosts)
        al=alignedPathNodes
        
        # compute contribution to the score
        # first identify the parts of both graphs that were matched
        #pathEdgeAligned=[ac[i-1][al[i-1],al[i]] for i in range(1,len(cand_positions))]           
        all_matched_paths=[] # parts of the graph that were aligned
        alignedPathPieces=[] # parts of the path that were aligned
        hasMissingEdge=False
        onPath=False
        if not ac[0][al[0],al[1]]:
            n_interruptions+=1
            pos_interruptions.append(newpath[0])
           
        for i in range(1,len(cand_positions)):
            if ac[i-1][al[i-1],al[i]]:
                if not onPath:
                    all_matched_paths.append([])
                    alignedPathPieces.append([])
                    if i>1:
                        n_interruptions+=1
                        pos_interruptions.append(newpath[i-1])
                        
                onPath=True
                pos_snapped=tuple(cand_positions[i-1][al[i-1]][2])
                snapped_node=cand_positions[i-1][al[i-1]][1]
                all_matched_paths[-1].append((pos_snapped,snapped_node))
                all_matched_paths[-1]+=[(g.nodes[n]['pos'],n) for n in sp[i-1][(al[i-1],al[i])] ]
                alignedPathPieces[-1].append((newpath[i-1], i-1))
            else:
                if onPath:
                    pos_snapped=tuple(cand_positions[i-1][al[i-1]][2])
                    snapped_node=cand_positions[i-1][al[i-1]][1]
                    all_matched_paths[-1].append((pos_snapped,snapped_node))
                    alignedPathPieces[-1].append((newpath[i-1],i-1))
                    
                    n_interruptions+=1
                    pos_interruptions.append(newpath[i-1])
                    
                onPath=False
                hasMissingEdge=True
        if onPath:
            pos_snapped=tuple(cand_positions[-1][al[-1]][2])
            snapped_node=cand_positions[-1][al[-1]][1]
            all_matched_paths[-1].append((pos_snapped,snapped_node))
            alignedPathPieces[-1].append((newpath[-1],len(newpath)-1))
        else:
            n_interruptions+=1
            pos_interruptions.append(newpath[-1])
        
        if hasMissingEdge:
            pass
        else:
            n_connected+=1
        
        sum_sq_lengths=0
        for app in alignedPathPieces:
            alpath=np.array([n[0] for n in app])
            #print("segment:", calcPathLength(alpath))
            sum_sq_lengths+=calcPathLength(alpath)**2
        path_length=calcPathLength(path)
        path_length2=path_length**2
        connection_probability.append(sum_sq_lengths/path_length2) #* path_length
        tot_lenght += path_length
        path_lenghts.append(path_length)
        
        # remove the matched part of the graph
        i=0
        while i<len(all_matched_paths):
            p=all_matched_paths[i]
            prev_node=None
            for n in p:
                if prev_node and n[1]!=prev_node:
                    if(prev_node,n[1]) in g.edges:
                        g.remove_edge(prev_node,n[1])
                    if prev_node in g:
                        prevn=list(nx.neighbors(g,prev_node))
                        if len(prevn)==0 or prevn==[prev_node]:
                            g.remove_node(prev_node)
                    if n[1] in g:
                        nn=list(nx.neighbors(g,n[1]))
                        if len(nn)==0 or nn==[n[1]]:
                            g.remove_node(n[1])
                           
                prev_node=n[1]
            i+=1
        
        # remove the nodes that were inserted to snap the path (probably redundant)
        nds=list(g.nodes)
        for n in nds:
            if n>max_node_before_snapping:
                nbs=list(nx.neighbors(g,n))
                if len(nbs)==2:
                    g.add_edge(nbs[0],nbs[1])
                    g.remove_node(n)
                elif len(nbs)>2:
                    raise ValueError("a snapped node has more than two neighbors ?")
                      
    connection_probability = np.array(connection_probability)
    connection_probability = np.mean(connection_probability)
    
    return n_connected, n_interruptions, connection_probability
        
def remove_self_loops(g):
    edges2remove=[]
    for e in g.edges:
        if e[0]==e[1]:
            edges2remove.append(e)
    for e in edges2remove:
        g.remove_edge(e[0],e[1])
        
def f1_score(precision, recall):
    return 2*(precision*recall)/(precision+recall)        

def opt_p(G_gt, G_pred, th_existing=10, th_snap=25):
    '''
    OPT-P metric
    
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
        
    Return
    ------
    n_conn_precis : int
        number of connected paths (G_pred matched to G_gt)
    n_conn_recall : int
        number of connected paths (G_gt matched to G_pred)
    n_inter_precis : int
        number of paths that have interruption(s) (G_pred matched to G_gt)
    n_inter_recall : int
        number of paths that have interruption(s) (G_gt matched to G_pred)
    con_prob_precis : float
        connection probability precision (G_pred matched to G_gt)
    con_prob_recall : float
        connection probability recall (G_gt matched to G_pred)
    con_prob_f1 : float
        f1 score for con_prob_precis and con_prob_recall
    '''
    
    if utils.is_empty(G_gt):
        raise ValueError("Ground-truth graph is empty!")
    
    if utils.is_empty(G_pred):
        print("!! Predicted graph is empty !!")
        n_conn_precis, n_conn_recall, n_inter_precis, n_inter_recall = 0,0,0,0
        con_prob_precis, con_prob_recall, f1 = 0,0,0
        return n_conn_precis, n_conn_recall, n_inter_precis, n_inter_recall, \
               con_prob_precis, con_prob_recall, f1   
    
    remove_self_loops(G_pred)
    remove_self_loops(G_gt)

    deduplicate_nodes(G_pred)
    deduplicate_nodes(G_gt)

    isolated_nodes=[n for n in G_pred.nodes if len(G_pred[n])==0]
    G_pred.remove_nodes_from(isolated_nodes)
    isolated_nodes=[n for n in G_gt.nodes if len(G_gt[n])==0]
    G_gt.remove_nodes_from(isolated_nodes) 
     
    n_conn_recall, n_inter_recall, \
    con_prob_recall = new2l2s_v0(G_pred.copy(), G_gt.copy())

    n_conn_precis, n_inter_precis, \
    con_prob_precis = new2l2s_v0(G_gt.copy(), G_pred.copy())
    
    con_prob_f1 = f1_score(con_prob_precis, con_prob_recall)

    return n_conn_precis, n_conn_recall, n_inter_precis, n_inter_recall, \
           con_prob_precis, con_prob_recall, con_prob_f1

