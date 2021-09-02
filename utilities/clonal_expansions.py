import os

import itertools
import math
import pandas as pd
import numba
import numpy as np
from ete3 import Tree
from statsmodels.stats.proportion import multinomial_proportions_confint

from itolapi import Itol
from itolapi import ItolExport

import cassiopeia as cas
from cassiopeia.solver import solver_utilities

EPSILON = 0.001

# Define math utilities
def nCr(n:int , r:int) -> float:
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def pnk(n: int, b: int, k: int) -> float:
    """Coalescent probability of imbalance

    Args:
        n: Number of leaves in subtree
        b: Number of leaves in one lineage
        k: Number of lineages

    Returns:
        Probability of observing b leaves on one lineage in a tree of n total 
            leaves
    """
    return nCr(n - b - 1, k - 2) / nCr(n-1, k-1)

def calculate_expansion_proportion(tree):
    """ Detects clonal expansion of all children at depth 1. 
    
    :param tree: ete3.TreeNode, the root of the tree
    :return: A mapping from child to confidence interval
    """

    N = len(tree.get_leaf_names())

    props = {}
    for c in tree.children:
        props[c] = len(c)

    ci = multinomial_proportions_confint(list(props.values()), alpha=0.05)
    props = {c: interval for c, interval in zip(props.keys(), ci)}
    return props


def annotate_tree(tree, min_clade_size=10, min_depth=1):
    """Annotate tree with respect to expansion proportions.
    
    :param tree: ete3.TreeNode, the root of the tree.
    :param min_clade_size: minimum size of clade to be evaluated.
    :param min_depth: minimum distance from root for clade to be evaluated.
    """

    node_to_exp = {}
    node2leaves = tree.get_cached_content()

    # instantiate dictionary
    for n in tree.traverse():
        node_to_exp[n.name] = 1.0

    for n in tree.traverse():
        kn = len(node2leaves[n])
        if tree.get_distance(n) >= min_depth:
            # exp_props = calculate_expansion_proportion(n)

            # for c in exp_props:
            #     node_to_exp[c.name] = round(exp_props[c][0], 2)
            num_lineages = len(n.children)
            for c in n.children:
                
                if len(node2leaves[c]) <= min_clade_size:
                    node_to_exp[c.name] = 1.0
                    continue
                
                b = len(node2leaves[c])
                p = np.sum([pnk(kn, b2, num_lineages) for b2 in range(b, kn-num_lineages+2)])
                node_to_exp[c.name] = p

    return tree, node_to_exp


def find_expanding_clones(tree, node_to_exp, pval = 0.05, _last=False):

    def ancestors_are_expanding(tree, n):
        """Detect if there is an ancestor expanding.
        """
        anc_expanding = False
        while n.up:
            n = n.up
            if n.expanding:
                anc_expanding = True
        return anc_expanding

    def child_is_expanding(tree, n):

        if n.is_leaf():
            return True

        for c in n.children:
            if c.expanding:
                return True

        return False

    N = len(tree.get_leaves())
    node2leaves = tree.get_cached_content()
    # determine which nodes pass the desired expansion
    # threshold
    for n in tree.traverse():
        if n.is_root():
            n.add_features(expanding=False)
            continue
        
        size_of_clade = len(node2leaves[n.up])
        if node_to_exp[n.name] < (pval+EPSILON):
            n.add_features(expanding=True)
        else:
            n.add_features(expanding=False)


    # find original expanding clone.
    expanding_nodes = {}
    
    if not _last:
        for n in tree.traverse():
            if n.expanding:
                if not ancestors_are_expanding(tree, n):
                    kn = len(node2leaves[n])
                    pn = len(node2leaves[n.up])
                    expanding_nodes[n.name] = (kn / pn, kn / N, node_to_exp[n.name], kn , N)

    else:

        expanding_node_names = []
        for n in tree.traverse():
            if n.expanding:

                if not child_is_expanding(tree, n):
                    expanding_node_names.append(n.name)
                else:
                    n.expanding=False

        # find original expanding clones.
        parallel_evolution = True
        while parallel_evolution:

            # detect parallel evolution events
            encountered_parallel_evolution = False
            for i, j in itertools.combinations(expanding_node_names, 2):
                
                ancestor = tree.get_common_ancestor(i, j)
                if ancestor.name != j and ancestor.name != i and ancestor.name in expanding_node_names:
                    ancestor.expanding = False
                    encountered_parallel_evolution = True
                    expanding_node_names.remove(ancestor.name)
                    break
                    
            if not encountered_parallel_evolution:
                parallel_evolution = False
        
        
        # now get top events
        expanding_nodes = {}
        for n in tree.traverse():
            if n.expanding:
                kn = len(node2leaves[n])
                pn = len(node2leaves[n.up])
                expanding_nodes[n.name] = (kn / pn, kn / N, node_to_exp[n.name], kn, N)
                
                for v in n.traverse():
                    v.expanding=False

    # remove temporary annotations
    for n in tree.traverse():
        if n.name not in expanding_nodes.keys():
            n.add_features(expanding=False)

    return tree, expanding_nodes


def detect_expansion(tree, pval = 0.05, _first=True, min_clade_prop=0.1, min_depth=0):

    tree = solver_utilities.collapse_unifurcations(tree)

    N = len(tree.get_leaves())

    tree, node_to_prop = annotate_tree(
        tree, min_clade_size=N * min_clade_prop, min_depth=min_depth
    )

    tree, expanding_nodes = find_expanding_clones(
        tree, node_to_prop, pval = pval, _last=(not _first)
    )

    expansion_df = pd.DataFrame.from_dict(
        expanding_nodes, orient="index", columns=["SubProp", "TotalProp", "Probability", "LeavesInExpansion", "TotalLeaves"]
    )
    return tree, expansion_df


def create_expansion_file_for_itol(tree, expansions, outfp=None):

    _leaves = tree.get_leaf_names()

    out = ""
    header = ["TREE_COLORS", "SEPARATOR SPACE", "DATA"]
    for line in header:
        out += line + "\n"
    for row in expansions.iterrows():
        out += row[0] + " clade #ff0000 normal 1\n"

    if outfp:
        with open(outfp, "w") as fOut:
            fOut.write(out)
        return outfp
    else:
        return out
