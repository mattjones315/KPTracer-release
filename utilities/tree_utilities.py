import os

import ete3
import networkx as nx
import numba
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm.auto import tqdm

from cassiopeia.data import utilities as data_utilities
from cassiopeia.solver import solver_utilities

HOME = "/data/yosef2/users/mattjones/projects/kptc"


def infer_ancestral_character_states(graph, character_matrix):

    root = [n for n in graph if graph.in_degree(n) == 0][0]

    for n in nx.dfs_postorder_nodes(graph, source=root):
        if graph.out_degree(n) == 0:
            graph.nodes[n]["character_states"] = character_matrix.loc[
                n
            ].tolist()
            continue

        children = [c for c in graph.successors(n)]
        character_states = [
            graph.nodes[c]["character_states"] for c in children
        ]
        reconstructed = data_utilities.get_lca_characters(character_states, -1)
        graph.nodes[n]["character_states"] = reconstructed

    return graph


def ete3_to_nx(tree, cm, labels):

    g = nx.DiGraph()

    node_iter = 0
    for n in tree.traverse():
        if n.name == "" or n.name not in cm.index.values:
            n.name = f"node{node_iter}"
            node_iter += 1
        if n.is_root():
            continue

        g.add_edge(n.up.name, n.name)

    g = infer_ancestral_character_states(g, cm)

    for (u, v) in g.edges():

        g[u][v]["length"] = get_modified_edit_distance(
            np.array(g.nodes[u]["character_states"]),
            np.array(g.nodes[v]["character_states"]),
        )

    return g


@numba.jit(nopython=True)
def get_modified_edit_distance(s1: np.array, s2: np.array, missing_state=-1):

    d = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            if s1[i] == missing_state or s2[i] == missing_state:
                d += 1
            if s1[i] == 0 or s2[i] == 0:
                d += 1
            else:
                d += 2

    return d


def compute_pairwise_dist_nx(g):

    tree_dist = []
    edit_dist = []

    root = [n for n in g if g.in_degree(n) == 0][0]
    DIST_TO_ROOT = nx.single_source_dijkstra_path_length(
        g, root, weight="length"
    )
    n_targets = len(g.nodes[root]["character_states"])

    _leaves = [n for n in g if g.out_degree(n) == 0]

    leaf_to_int = dict(zip(_leaves, range(len(_leaves))))
    int_to_leaf = dict([(leaf_to_int[k], k) for k in leaf_to_int])

    all_pairs = []
    pair_names = []
    for i1 in tqdm(range(len(_leaves)), desc="Creating pairs to compare"):
        l1 = _leaves[i1]
        for i2 in range(i1 + 1, len(_leaves)):
            l2 = _leaves[i2]

            all_pairs.append((l1, l2))

    print("Finding LCAs for all pairs...")
    lcas = nx.algorithms.tree_all_pairs_lowest_common_ancestor(
        g, pairs=all_pairs
    )

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

        tree_dist = dA + dB - 2 * dC

        edit_dist = get_modified_edit_distance(
            np.array(g.nodes[l1]["character_states"]),
            np.array(g.nodes[l2]["character_states"]),
        )

        phylogenetic_distance_matrix[i, j] = phylogenetic_distance_matrix[
            j, i
        ] = tree_dist
        edit_distance_matrix[i, j] = edit_distance_matrix[j, i] = edit_dist

    diam = np.max(phylogenetic_distance_matrix)
    phylogenetic_distance_matrix /= diam
    edit_distance_matrix /= 2 * n_targets

    phylogenetic_distance_matrix = pd.DataFrame(
        phylogenetic_distance_matrix, index=_leaves, columns=_leaves
    )
    edit_distance_matrix = pd.DataFrame(
        edit_distance_matrix, index=_leaves, columns=_leaves
    )

    return (
        phylogenetic_distance_matrix,
        all_pairs,
        edit_distance_matrix,
        diam,
        n_targets,
    )


def prepare_tumor_tree(
    tumor, adata, tumor2model, column="leiden_sub", FILTER_PROP=0, preprocess=True
):

    fp = os.path.join(HOME, "trees", tumor, tumor2model.loc[tumor, "Newick"])
    tree = ete3.Tree(fp, 1)
    tree = solver_utilities.collapse_unifurcations(tree)

    cm = pd.read_csv(
        os.path.join(HOME, "trees", tumor, f"{tumor}_character_matrix.txt"),
        sep="\t",
        index_col=0,
    )
    cm = cm.replace("-", "-1").astype(int)

    keep_cells = np.intersect1d(tree.get_leaf_names(), adata.obs_names)
    filter_thresh = int(len(keep_cells) * FILTER_PROP)

    obs_counts = (
        adata.obs.loc[keep_cells].groupby(column).agg({column: "count"})
    )
    keep_vals = obs_counts[obs_counts[column] > filter_thresh].index.values

    tumor_obs = adata.obs.loc[keep_cells, column]

    filt = tumor_obs.apply(lambda x: x in keep_vals)
    tumor_obs = tumor_obs.loc[filt]

    _leaves = tree.get_leaf_names()

    keep_cells = np.intersect1d(tumor_obs.index.values, _leaves)

    tree.prune(keep_cells)

    graph = ete3_to_nx(tree, cm, tumor_obs)

    if preprocess:
        graph = preprocess_exhausted_lineages(graph, tumor_obs, tumor_obs.unique())

    return graph


@numba.jit(nopython=True)
def NNI(dmat: np.array, inds1: np.array, inds2: np.array):

    nni = np.inf
    for i in inds1:
        for j in inds2:
            if dmat[i, j] < nni:
                nni = dmat[i, j]

    return nni


# @numba.jit(nopython=True)
def average_nn_dist(dmat: np.array, inds1: np.array, inds2: np.array, k=1):

    _k = min(len(inds2) - 1, k)
    nni = 0
    for i in inds1:
        nn_dist = np.partition(dmat[i, inds2], _k)[_k - 1]
        nni += nn_dist

    _k = min(len(inds1) - 1, k)
    for i in inds2:
        nn_dist = np.partition(dmat[i, inds1], _k)[_k - 1]
        nni += nn_dist
    return nni / (len(inds1) * len(inds2))


@numba.jit(nopython=True)
def NRI(dmat: np.array, inds1: np.array, inds2: np.array):

    nri = 0
    for i in inds1:
        for j in inds2:
            nri += dmat[i, j]

    return nri / (len(inds1) * len(inds2))


@numba.jit(nopython=True)
def _hausdorf(dmat: np.array, inds1: np.array, inds2: np.array):

    cmax = 0.0
    for i in inds1:
        cmin = np.inf
        for j in inds2:
            d = dmat[i, j]
            if d < cmin:
                cmin = d
            if cmin < cmax:
                break
        if cmin > cmax and np.inf > cmin:
            cmax = cmin
    for j in inds2:
        cmin = np.inf
        for i in inds1:
            d = dmat[i, j]
            if d < cmin:
                cmin = d
            if cmin < cmax:
                break
        if cmin > cmax and np.inf > cmin:
            cmax = cmin
    return cmax


def get_inter_cluster_df(leaf_states, distance_matrix, func=NRI, **kwargs):

    uniq_states = leaf_states.unique()
    inter_cluster_df = pd.DataFrame(
        np.zeros((len(uniq_states), len(uniq_states))),
        index=uniq_states.astype(int),
        columns=uniq_states.astype(int),
    )

    for s1 in uniq_states:
        for s2 in uniq_states:
            inter_cluster_df.loc[
                int(s1), int(s2)
            ] = calculate_inter_state_dissimilarity(
                s1, s2, leaf_states, distance_matrix, func, **kwargs
            )

    return inter_cluster_df


def calculate_inter_state_dissimilarity(
    s1, s2, assignments, distance_matrix: pd.DataFrame, func=NRI, **kwargs
):

    # order assignments by distance matrix
    assignments = assignments.loc[distance_matrix.index.values]

    _leaf_inds1 = np.where(np.array(assignments) == s1)[0]
    _leaf_inds2 = np.where(np.array(assignments) == s2)[0]

    return func(distance_matrix.values, _leaf_inds1, _leaf_inds2, **kwargs)

def preprocess_exhausted_lineages(g, labels, sigma):
    
    leaves = [n for n in g if g.out_degree(n) == 0]
    for l in leaves:
        g.nodes[l]['label'] = labels.loc[l]
    
    is_leaf = lambda x: g.out_degree(x) == 0
    
    root = [n for n in g if g.in_degree(n) == 0][0]
    
    _exhausted = []
    
    # find all nodes corresponding to an exhausted area
    for n in nx.dfs_preorder_nodes(g, root):
        
        if is_leaf(n):
            continue
        
        children = list(g.successors(n))
    
        if np.all([is_leaf(c) for c in children]):
            
            n_uniq = np.unique([g.nodes[c]['label'][0] for c in children])
        
            if n_uniq == 1 or len(children) < 3:
                continue
            
            _exhausted.append(n)
        
    # split exhausted nodes by label
    
    for n in _exhausted:
        
        children = [(c, g.nodes[c]['label'][0]) for c in list(g.successors(n))]
            
        for s in sigma:
            
            sub_c = [c[0] for c in children if c[1] == s]
            
            if len(sub_c) > 0:
            
                g.add_edge(n, str(n) + "-" + s, length=1)

                for sc in sub_c:
                    g.add_edge(str(n) + "-" + s, sc, length = 0)
                    g.remove_edge(n, sc)
            
    return g
