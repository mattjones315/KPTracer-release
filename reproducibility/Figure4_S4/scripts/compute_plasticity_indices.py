import os

import copy
from collections import defaultdict
import colorcet as cc
import ete3
import matplotlib.pyplot as plt
import networkx as nx
import numba
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import scipy as sp
from tqdm.auto import tqdm

from cassiopeia.Analysis import small_parsimony
from cassiopeia.TreeSolver.Node import Node
import utilities


def infer_ancestral_states(graph, character_matrix):
    
    root = [n for n in graph if graph.in_degree(n) == 0][0]
    
    for n in nx.dfs_postorder_nodes(graph, source = root):
        if graph.out_degree(n) == 0:
            graph.nodes[n]['character_states'] = character_matrix.loc[n].tolist()
            continue
        
        children = [c for c in graph.successors(n)]
        character_states = [graph.nodes[c]['character_states'] for c in children]
        reconstructed = utilities.get_lca_characters(
            character_states
        )
        graph.nodes[n]['character_states'] = reconstructed
        
    return graph

def preprocess_exhausted_regions(g, sigma):
    
    is_leaf = lambda x: g.out_degree(x) == 0
    
    root = [n for n in g if g.in_degree(n) == 0][0]
    
    _exhausted = []
    
    # find all nodes corresponding to an exhausted area
    for n in nx.dfs_preorder_nodes(g, root):
        
        if is_leaf(n):
            continue
        
        children = list(g.successors(n))
    
        if np.all([is_leaf(c) for c in children]):
            
            n_uniq = np.unique([g.nodes[c]['S1'][0] for c in children])
        
            if len(n_uniq) == 1 or len(children) < 3:
                continue
            
            _exhausted.append(n)
        
    # split exhausted nodes by label
    
    for n in _exhausted:
        
        children = [(c, g.nodes[c]['S1'][0]) for c in list(g.successors(n))]
            
        for s in sigma:
            
            sub_c = [c[0] for c in children if c[1] == s]
            
            if len(sub_c) > 0:
            
                g.add_edge(n, str(n) + "-" + s, length=1)

                for sc in sub_c:
                    g.add_edge(str(n) + "-" + s, sc, length = 0)
                    g.remove_edge(n, sc)
            
    return g

def ete3_to_nx(tree, cm):
    
    g = nx.DiGraph()
    
    node_iter = 0
    for n in tree.traverse():
        if n.name == "" or "|" in n.name:
            n.name = f'node{node_iter}'
            node_iter += 1
        if n.is_root():
            continue
        
        g.add_edge(n.up.name, n.name)
    
    g = infer_ancestral_states(g, cm)
    
    for (u, v) in g.edges():
        
        uarr = np.array(g.nodes[u]['character_states'])
        varr = np.array(g.nodes[v]['character_states'])
        if get_modified_edit_distance(uarr, varr) == 0:
            g[u][v]['length'] = 0
        else:
            g[u][v]['length'] = 1 
    
    return g

@numba.jit(nopython=True)
def get_modified_edit_distance(s1: np.array, s2:np.array, missing_state = -1):
    
    d = 0
    num_present = 0
    for i in range(len(s1)):
        
        if s1[i] == missing_state or s2[i] == missing_state:
            continue

        num_present += 1
        
        if s1[i] != s2[i]:
            if s1[i] == 0 or s2[i] == 0:
                d += 1
            else:
                d += 2

    return d

def compute_pairwise_dist_nx(g):
    
    tree_dist = []
    edit_dist = []
    
    root = [n for n in g if g.in_degree(n) == 0][0]
    DIST_TO_ROOT = nx.single_source_dijkstra_path_length(g, root, weight='length')
    n_targets = len(g.nodes[root]['character_states'])
    
    _leaves = [n for n in g if g.out_degree(n) == 0]
    
    leaf_to_int = dict(zip(_leaves, range(len(_leaves))))
    int_to_leaf = dict([(leaf_to_int[k], k) for k in leaf_to_int])    
    
    all_pairs = []
    pair_names = []
    for i1 in tqdm(range(len(_leaves)), desc = "Creating pairs to compare"):
        l1 = _leaves[i1]
        for i2 in range(i1+1, len(_leaves)):
            l2 = _leaves[i2]
            
            all_pairs.append((l1, l2))

    print("Finding LCAs for all pairs...")
    # lcas = nx.algorithms.lowest_common_ancestors.all_pairs_lowest_common_ancestor(g, pairs=all_pairs)
    lcas = nx.algorithms.tree_all_pairs_lowest_common_ancestor(g, pairs=all_pairs)
    
    print("Computing pairwise distances...")

    ug = g.to_undirected()
    
    phylogenetic_distance_matrix = np.zeros((len(_leaves), len(_leaves)))
    edit_distance_matrix = np.zeros((len(_leaves), len(_leaves)))
    
    for _lca in lcas:
                
        n_pair, mrca = _lca[0], _lca[1]        
        l1, l2 = n_pair[0], n_pair[1]
        
        i, j = leaf_to_int[l1], leaf_to_int[l2]
    
        dA = DIST_TO_ROOT[l1]
        dB = DIST_TO_ROOT[l2]
        dC = DIST_TO_ROOT[mrca]

        tree_dist = (dA + dB - 2*dC)
        
        edit_dist = get_modified_edit_distance(np.array(g.nodes[l1]['character_states']), np.array(g.nodes[l2]['character_states']))
        
        phylogenetic_distance_matrix[i, j] = phylogenetic_distance_matrix[j, i] = tree_dist
        edit_distance_matrix[i,j] = edit_distance_matrix[j, i] = edit_dist
    
    diam = np.max(phylogenetic_distance_matrix)
    phylogenetic_distance_matrix /= diam
    edit_distance_matrix /= n_targets
    
    phylogenetic_distance_matrix = pd.DataFrame(phylogenetic_distance_matrix, index = _leaves, columns = _leaves)
    edit_distance_matrix = pd.DataFrame(edit_distance_matrix, index = _leaves, columns= _leaves)
    
    return phylogenetic_distance_matrix, all_pairs, edit_distance_matrix

def get_nearest_neighbors(distance_matrix, k = 10):
    
    # tree_dists, pair_names, edit_distance  = compute_pairwise_dist_nx(graph)
    
    _leaves = distance_matrix.index.values
    
    dmat = distance_matrix.copy()
    
    np.fill_diagonal(dmat.values, 1e6)
    
    leaf_to_nn = {}
    leaf_to_nn_dist = {}
    for l in _leaves:
        
        _idx = np.argpartition(dmat.loc[l].values, k)[:k]
        dist = np.max(dmat.loc[l].values[_idx])
        
        idx = np.where(dmat.loc[l].values <= dist)[0]
        
        neighbors = dmat.columns[idx]
        phydists = dmat.loc[l].values[idx]
        leaf_to_nn[l] = neighbors
        leaf_to_nn_dist[l] = phydists
        
    return leaf_to_nn, leaf_to_nn_dist

def calculate_stability(nn, adata, column):
    
    cluster_to_stability = defaultdict(float)
    cluster_freq = defaultdict(int)
    for n in nn:

        neighbors = nn[n]
        cluster = adata.obs.loc[n, column]
        cluster_freq[cluster] += 1

        score = 0.0
        for neigh in neighbors:
            if adata.obs.loc[neigh, column] != cluster:
                score += 1

        cluster_to_stability[cluster] += (score / len(neighbors))

    for cluster in cluster_to_stability:
        cluster_to_stability[cluster] = (1 - cluster_to_stability[cluster] / cluster_freq[cluster])
        
    return cluster_to_stability

def score_plasticity_of_clades(graph, root):
    
    for n in nx.dfs_postorder_nodes(graph, source = root):

        succ = [k for k in nx.descendants(graph, n)] + [n]
        if len(succ) == 1:
            continue
        subg = graph.subgraph(succ)
        subg = small_parsimony.fitch_hartigan(subg)
        parsimony = small_parsimony.score_parsimony(subg)

        graph.nodes[n]['plasticity'] = (parsimony / len(subg.edges()))
    
    return graph

def calculate_l2(cell, neighbors, latent, meta):
    
    cell_vec = latent.loc[cell].values
    
    l2 = []
    for n in neighbors:
        if meta.loc[n] == meta.loc[cell]:
            l2.append(0)
        else:
            l2.append(sp.spatial.distance.minkowski(cell_vec, latent.loc[n].values, p=2))
        
    return np.mean(l2)

def calculate_allele_plasticity(cell, neighbors, meta):

    score = 0
    cluster = meta.loc[cell]
    for neigh in neighbors:
        if meta.loc[neigh] != cluster:
            score += 1

    score /= len(neighbors)
        
    return score

def main():
    FILTER_THRESH = 0.025

    tumor2model = pd.read_csv("/data/yosef2/users/mattjones/projects/kptc/trees/tumor_model.txt", sep='\t', index_col = 0)
    adata_all = sc.read_h5ad("/data/yosef2/users/mattjones/projects/kptc/RNA/ALL/adata_processed.all_filtered_combined_assignments.h5ad")

    tumors = [t for t in tumor2model.index if 'Fam' not in t and 'All' not in t and t.split("_")[2].startswith("T")]

    FILTER_PROP = 0.025
    column = 'Combined_Clusters'

    sc_plasticity_fitch = {}
    sc_plasticity_l2 = {}
    sc_plasticity_allele = {}

    for tumor in tqdm(tumors):
        fp = os.path.join('/data/yosef2/users/mattjones/projects/kptc/trees/', tumor, tumor2model.loc[tumor, 'Newick'])
        tree = ete3.Tree(fp, 1)

        cm = pd.read_csv(os.path.join("/data/yosef2/users/mattjones/projects/kptc/trees/", tumor, f'{tumor}_character_matrix.txt'), sep='\t', index_col = 0)
        cm = cm.replace("-", "-1").astype(int)
        
        keep_cells = np.intersect1d(tree.get_leaf_names(), adata_all.obs_names)
        
        filter_thresh = int(FILTER_PROP*len(keep_cells))
        print("Filtering out states with frequency less than " + str(filter_thresh))
        
        obs_counts = adata_all.obs.loc[keep_cells].groupby(column).agg({column: 'count'})

        keep_vals = obs_counts[obs_counts[column] > filter_thresh].index.values

        tumor_obs = adata_all.obs.loc[keep_cells, column]

        filt = tumor_obs.apply(lambda x: x in keep_vals)
        tumor_obs = tumor_obs.loc[filt]

        keep_cells = np.intersect1d(tumor_obs.index.values, keep_cells)

        tree.prune(keep_cells)
        tree = utilities.collapse_unifurcations(tree)

        graph = ete3_to_nx(tree, cm)

        tree_dists, pair_names, edit_distance  = compute_pairwise_dist_nx(graph)

        # compute l2 score
        phy_nn, phy_dist = get_nearest_neighbors(tree_dists, k=1)
        
        latent_space = pd.DataFrame(adata_all.obsm['X_scVI'], index=adata_all.obs_names)

        for cell in phy_nn.keys():
            sc_plasticity_l2[cell] = calculate_l2(cell, phy_nn[cell], latent_space, adata_all.obs[column])

        # compute allele score
        edit_nn, edit_dist = get_nearest_neighbors(edit_distance, k=1)

        for cell in edit_nn.keys():
            sc_plasticity_allele[cell] = calculate_allele_plasticity(cell, edit_nn[cell], adata_all.obs[column])

        #convert graph names to Nodes
        mapping = {}
        for n in graph:
            mapping[n] = Node(n)

        graph = nx.relabel_nodes(graph, mapping)

        graph = small_parsimony.assign_labels(graph, adata_all.obs[column])
        graph = preprocess_exhausted_regions(graph, adata_all.obs[column].unique())

        root = [n for n in graph if graph.in_degree(n) == 0][0]
        
        # compute fitch plasticity score
        graph = score_plasticity_of_clades(graph, root)
        
        leaves = [n for n in graph if graph.out_degree(n) == 0]
        for l in leaves:
            path = nx.shortest_path(graph, root, l)
            ms = 0
            for p in path:
                if p != l:
                    ms += graph.nodes[p]['plasticity'] 
            ms /= (len(path) - 1)

            sc_plasticity_fitch[l.name] = ms

    plasticity_df = pd.DataFrame.from_dict(sc_plasticity_fitch, orient='index', columns = ['scPlasticity'])
    plasticity_df['scPlasticityL2'] = plasticity_df.index.map(sc_plasticity_l2)
    plasticity_df['scPlasticityAllele'] = plasticity_df.index.map(sc_plasticity_allele)

    plasticity_df.to_csv("./plasticity_scores_v2.tsv", sep='\t')    

if __name__ == "__main__":
    main()




