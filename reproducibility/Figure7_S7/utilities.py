import ete3
import networkx as nx
import numba
import numpy as np
import pandas as pd

@numba.jit(nopython=True)
def compute_average_dissimilarity(mat1, mat2):
    
    sim = 0
    for i1 in range(mat1.shape[0]):
        r1 = mat1[i1,:]
        for i2 in range(mat2.shape[0]):
            _sim = 0
            n_present = 0
            
            r2 = mat2[i2,:]
            
            for j in range(mat1.shape[1]):

                if r1[j] == -1 or r2[j] == -1:
                    _sim += 1
                    
                n_present += 1
            
                if r1[j] != r2[j]:
                    
                    if r1[i] == 0 or r2[i] == 0:
                        _sim += 1
                    else:
                        _sim += 2
                
            sim += (_sim / max(1, (2*n_present)))
            
    return sim /(mat1.shape[0] * mat2.shape[0])

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

def ete3_to_nx(tree, cm):
    
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
        
        _length = get_modified_edit_distance(np.array(g.nodes[u]['character_states']), np.array(g.nodes[v]['character_states']))
        
        g[u][v]['length'] = _length
        
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


def prepare_tumor_tree(tumor, adata, tree_dir):
    
    fp = f'{tree_dir}/{tumor}_tree.nwk'
    tree = ete3.Tree(fp, 1)
    tree = collapse_unifurcations(tree)

    cm = pd.read_csv(f'{tree_dir}/{tumor}_character_matrix.txt', sep='\t', index_col = 0)
    cm = cm.replace("-", "-1").astype(int)

    _leaves = tree.get_leaf_names()

    keep_cells = np.intersect1d(adata.obs.index.values, _leaves)

    tree.prune(keep_cells)

    graph = ete3_to_nx(tree, cm)
    
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

from collections import defaultdict
from cassiopeia.analysis import small_parsimony
from cassiopeia.solver.Node import Node
import cassiopeia.TreeSolver.compute_meta_purity as cmp
import numba

@numba.jit(nopython=True)
def NRI(dmat: np.array, inds1: np.array, inds2: np.array):
    
    nri = 0
    for i in inds1:
        for j in inds2:
            nri += dmat[i,j]
            
    return nri / (len(inds1)*len(inds2))

def calculate_inter_state_dissimilarity(s1, s2, assignments, distance_matrix: pd.DataFrame):
    
    # order assignmnets by distance matrix
    assignments = assignments.loc[distance_matrix.index.values]
    
    _leaf_inds1 = np.where(np.array(assignments) == s1)[0]
    _leaf_inds2 = np.where(np.array(assignments) == s2)[0]
        
    return NRI(distance_matrix.values, _leaf_inds1, _leaf_inds2)

def get_inter_cluster_df(leaf_states, phylogenetic_distance_matrix):

    uniq_states = leaf_states.unique()
    inter_cluster_df = pd.DataFrame(np.zeros((len(uniq_states), len(uniq_states))), index = uniq_states, columns = uniq_states)

    for s1 in (uniq_states):
        for s2 in uniq_states:
            inter_cluster_df.loc[s1, s2] = calculate_inter_state_dissimilarity(s1, s2, leaf_states, phylogenetic_distance_matrix)
            
    return inter_cluster_df

def compute_evolutionary_coupling(_distance_matrix, meta, B = 500):
    
    uniq_states = meta.unique()
    _observed_cluster_df = pd.DataFrame(np.zeros((len(uniq_states), len(uniq_states))), index = uniq_states, columns = uniq_states)
    _null_means = _observed_cluster_df.copy()
    _null_sds = _observed_cluster_df.copy()
    
    _leaves = np.intersect1d(meta.index.values, _distance_matrix.index.values)
    
    leaf_states = meta.loc[_leaves]
    distance_matrix = _distance_matrix.loc[_leaves, _leaves].copy()
    
    observed_inter_cluster_df = get_inter_cluster_df(leaf_states, distance_matrix)

    background = defaultdict(list)
    for i in tqdm(range(B)):

        permuted_assignments = leaf_states.copy()
        permuted_assignments.index = np.random.permutation(leaf_states.index.values)

        bg_df = get_inter_cluster_df(permuted_assignments, distance_matrix)
        for s1 in bg_df.index:
            for s2 in bg_df.index:
                background[(s1,s2)].append(bg_df.loc[s1, s2])
            
    null_means = observed_inter_cluster_df.copy()
    null_sds = observed_inter_cluster_df.copy()

    for s1 in null_means.index:
        for s2 in null_means.index:

            null_means.loc[s1, s2] = np.mean(background[(s1, s2)])
            null_sds.loc[s1, s2] = np.std(background[(s1, s2)])
        
    z_df = observed_inter_cluster_df.copy()
    delta = 0.001
    for ind in z_df.index:
        for col in z_df.columns:
            z_df.loc[ind, col] = (delta + z_df.loc[ind, col] - null_means.loc[ind, col]) / (delta+null_sds.loc[ind, col])
    z_df.fillna(0, inplace=True)    
    
    mu = np.mean(z_df.values.ravel())
    sigma = np.std(z_df.values.ravel())

    zz_df = z_df.apply(lambda x: (x - mu) / sigma, axis=1)
    
    return zz_df


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

def get_fitch_tree(graph, meta):
    
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
    
    return fitch_tree

def compute_fitch_distance(graph, meta, leaf_subset = None, ground_state='4', run_fitch = False):
    
    if run_fitch:
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
    else:
        
        _leaves = [n for n in graph if graph.out_degree(n) == 0]
        root = [n for n in graph if graph.in_degree(n) == 0][0]
        
        fitch_tree = graph
    
    if leaf_subset is None:
        distances = [get_nearest_ancestor(fitch_tree, n, ground_state) for n in _leaves]
        distances = pd.DataFrame(distances, columns = ['distance'], index=[l.name for l in _leaves])
    else:
        _leaf_subset = [n for n in _leaves if n.name in leaf_subset]
        distances = [get_nearest_ancestor(fitch_tree, n, ground_state) for n in _leaf_subset]
        distances = pd.DataFrame(distances, columns = ['distance'], index=[l.name for l in _leaf_subset])
    
    graph.nodes[root]['depth'] = 0
    for u, v in nx.dfs_edges(graph, source=root):
        graph.nodes[v]["depth"] = (
            graph.nodes[u]["depth"] + graph[u][v]["length"]
        )

    max_depth = np.max([graph.nodes[n]['depth'] for n in _leaves])
    distances /= max_depth
    
    return distances