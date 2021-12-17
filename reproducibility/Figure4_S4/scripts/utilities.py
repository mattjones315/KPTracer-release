import ete3
import networkx as nx
import numpy as np


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
        if not n.is_leaf():
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

def infer_ancestral_states(graph, character_matrix):
    
    root = [n for n in graph if graph.in_degree(n) == 0][0]
    
    for n in nx.dfs_postorder_nodes(graph, source = root):
        if graph.out_degree(n) == 0:
            graph.nodes[n]['character_states'] = character_matrix.loc[n].tolist()
            continue
        
        children = [c for c in graph.successors(n)]
        character_states = [graph.nodes[c]['character_states'] for c in children]

        reconstructed = get_lca_characters(
            character_states
        )
        graph.nodes[n]['character_states'] = reconstructed
        
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

def get_modified_edit_distance(x_list, y_list):
    
    count = 0
    for i in range(0, len(x_list)):
        
        if x_list[i] == y_list[i]:
            count += 0

        elif x_list[i] == -1 or y_list[i] == -1:
            count += 0

        elif x_list[i] == 0 or y_list[i] == 0:
            count += 1

        else:
            count += 2

    return count