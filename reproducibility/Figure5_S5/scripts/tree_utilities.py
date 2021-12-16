import os

import ete3
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import numba
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm.auto import tqdm

from cassiopeia.Analysis import small_parsimony
import cassiopeia.TreeSolver.compute_meta_purity as cmp
from cassiopeia.TreeSolver.Node import Node


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
        reconstructed = get_lca_characters(character_states)
        graph.nodes[n]["character_states"] = reconstructed

    return graph

def get_lca_characters(
    vecs,
):
    """Builds the character vector of the LCA of a list of character vectors,
    obeying Camin-Sokal Parsimony.

    For each index in the reconstructed vector, imputes the non-missing
    character if only one of the constituent vectors has a missing value at that
    index, and imputes missing value if all have a missing value at that index.

    Args:
        vecs: A list of character vectors to generate an LCA for

    Returns:
        A list representing the character vector of the LCA

    """
    k = len(vecs[0])
    for i in vecs:
        assert len(i) == k
    lca_vec = [0] * len(vecs[0])
    for i in range(k):
        chars = set()
        for vec in vecs:
            chars.add(vec[i])
        if len(chars) == 1:
            lca_vec[i] = list(chars)[0]
        else:
            if -1 in chars:
                chars.remove(-1)
                if len(chars) == 1:
                    lca_vec[i] = list(chars)[0]
    return lca_vec

def ete3_to_nx(tree, cm, labels, keep_branch_lengths = False):
    
    g = nx.DiGraph()
    
    node_iter = 0
    for n in tree.traverse():
        if n.name == "" or n.name not in cm.index.values:
            n.name = f'node{node_iter}'
            node_iter += 1
        if n.is_root():
            continue
        
        g.add_edge(n.up.name, n.name)
    
    g = infer_ancestral_character_states(g, cm)
    
    for (u, v) in g.edges():
        
        _dist = get_modified_edit_distance(np.array(g.nodes[u]['character_states']), np.array(g.nodes[v]['character_states']))
        
        if _dist == 0:
            g[u][v]['length'] = 0
        else:
            if keep_branch_lengths:
                g[u][v]['length'] = _dist
            else:
                g[u][v]['length'] = 1 
    
    if not keep_branch_lengths:
        g = preprocess_exhausted_lineages(g, labels, labels.unique())
    
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
    tumor, adata, tree_dir, column="leiden_sub", FILTER_PROP=0, preprocess=True, keep_branch_lengths=False,
):
    
    fp = f'{tree_dir}/{tumor}_tree.nwk'
    tree = ete3.Tree(fp, 1)
    tree = collapse_unifurcations(tree)

    cm = pd.read_csv(
        f"{tree_dir}/{tumor}_character_matrix.txt",
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

    graph = ete3_to_nx(tree, cm, tumor_obs, keep_branch_lengths)

    if preprocess:
        graph = preprocess_exhausted_lineages(graph, tumor_obs, tumor_obs.unique())

    return graph

def collapse_unifurcations(tree: ete3.Tree) -> ete3.Tree:
    """Collapse unifurcations.
    Collapse all unifurcations in the tree, namely any node with only one child
    should be removed and all children should be connected to the parent node.
    Args:
        tree: tree to be collapsed
    Returns:
        A collapsed tree.
    """

    collapse_fn = lambda x: (len(x.children) == 1)

    collapsed_tree = tree.copy()
    to_collapse = [n for n in collapsed_tree.traverse() if collapse_fn(n)]

    for n in to_collapse:
        n.delete()

    return collapsed_tree

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
        
            if len(n_uniq) == 1 or len(children) < 3:
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

def compute_fitch_distance(graph, meta, ground_state='4'):
    
    mapping = {}
    for n in graph:
        mapping[n] = Node(n)

    graph = nx.relabel_nodes(graph, mapping)

    graph = small_parsimony.assign_labels(graph, meta)
    
    _leaves = [n for n in graph if graph.out_degree(n) == 0]
    root = [n for n in graph if graph.in_degree(n) == 0][0]

    # form candidate set of labels for each internal node
    S = np.unique(np.concatenate([graph.nodes[l]["S1"] for l in _leaves]))

    fitch_tree = cmp.set_depth(graph, root)
    fitch_tree = small_parsimony.fitch_hartigan_bottom_up(graph, root, S)
        
    distances = [get_nearest_ancestor(fitch_tree, n, ground_state) for n in _leaves]
    distances = pd.DataFrame(distances, columns = ['distance'], index=[l.name for l in _leaves])
    
    graph.nodes[root]['depth'] = 0
    for u, v in nx.dfs_edges(graph, source=root):
        graph.nodes[v]["depth"] = (
            graph.nodes[u]["depth"] + graph[u][v]["length"]
        )

    max_depth = np.max([graph.nodes[n]['depth'] for n in _leaves])
    distances /= max_depth
    
    return distances

def get_nearest_ancestor(graph, n, ground_state = '4'):
        
    edges = [k for k in nx.dfs_edges(graph.reverse(), n)]
    
    edge_iter = 0
    edge = edges[edge_iter]
    distance = graph[edge[1]][edge[0]]['length']
    parent = edge[1]
    while (ground_state not in graph.nodes[parent]['S1']) and (edge_iter < (len(edges)-1)):
        edge_iter += 1
        edge = edges[edge_iter]
        distance += graph[edge[1]][edge[0]]['length']
        parent = edge[1]
    return distance
        
def find_neighbor_names(index, neighbors, obs_names):
    
    connectivities = neighbors['connectivities'][index,:]
    
    indices = connectivities.indices

    return obs_names[indices]

def perform_smoothing(raw_values, data, K = 10):
    
    data = data[raw_values.index, :].copy()
    sc.pp.neighbors(data, n_neighbors=K, use_rep = 'X_scVI')
    smoothed_values = raw_values.copy()
    smoothed_values['adata_index'] = smoothed_values.apply(lambda x: np.where(data.obs_names == x.name)[0][0], axis=1)
    smoothed_values['distance'] = smoothed_values.apply(lambda x: raw_values.loc[find_neighbor_names(x.adata_index, data.uns['neighbors'], data.obs_names)].mean(), axis=1)

    return pd.DataFrame(smoothed_values['distance'])


def plot_graph_on_umap(adata, df, min_prop = 0.0, weight=1.0, cluster_column='leiden_sub', title=None):
    
    umap_coords = pd.DataFrame(adata.obsm['X_umap'], index = adata.obs_names)
    paga_pos = {}
    for n, g in adata.obs.groupby(cluster_column):

        paga_pos[int(n)] = umap_coords.loc[g.index].mean(0).values

    adjacency_solid = df.fillna(0).copy()
    adjacency_solid.index = [int(n) for n in adjacency_solid.index]
    adjacency_solid.columns = [int(n) for n in adjacency_solid.columns]
    adjacency_solid = adjacency_solid.loc[range(15), range(15)]
    
    adjacency_solid[adjacency_solid < min_prop] = 0.0

    base_edge_width = weight * mpl.rcParams['lines.linewidth']

    nx_g_solid = nx.Graph(adjacency_solid.values)

    widths = [x[-1]['weight'] for x in nx_g_solid.edges(data=True)]
    widths = base_edge_width * np.array(widths)

    h = plt.figure(figsize=(7,7))
    ax = plt.gca()
    nx.draw_networkx_nodes(nx_g_solid, paga_pos, ax=ax)
    nx.draw_networkx_labels(nx_g_solid, paga_pos, ax=ax)
    nx.draw_networkx_edges(
                    nx_g_solid, paga_pos, ax=ax, width=widths, edge_color='black'
                )
    sc.pl.embedding(adata, basis='X_umap', color=cluster_column, ax=ax, show=False)
    if title:
        plt.title(title)
    plt.tight_layout()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()