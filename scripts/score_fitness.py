import sys
import os
import time

import ete3
import networkx as nx
import numpy as np
import pandas as pd
import pickle as pic
import scipy
from tqdm import tqdm

sys.path.append("/data/yosef2/users/mattjones/projects/kptc/jungle/")  # specify path to jungle
import jungle as jg


def convert_network_to_newick_format(graph):
    """
	Given a networkx network, converts to proper Newick format.

	:param graph:
		Networkx graph object
	:return: String in newick format representing the above graph
	"""

    def _to_newick_str(g, node):
        is_leaf = g.out_degree(node) == 0
        is_root = g.in_degree(node) == 0

        if is_root:
            g.nodes[node]["length"] = 0

        if node.name == "internal" or node.name == "state-node":
            _name = node.get_character_string()
        else:
            _name = node.name

        return (
            "%s" % (_name,) + ":" + str(g.nodes[node]["length"])
            if is_leaf
            else (
                "("
                + ",".join(_to_newick_str(g, child) for child in g.successors(node))
                + "):"
                + str(g.nodes[node]["length"])
            )
        )

    def to_newick_str(g, root=0):  # 0 assumed to be the root
        return _to_newick_str(g, root) + ";"

    return to_newick_str(
        graph, [node for node in graph if graph.in_degree(node) == 0][0]
    )


infiles = []
names_trees = []

target_trees = []  # populate this list if you'd like to restrict your analysis

tumor_to_model = pd.read_csv(
    "/data/yosef2/users/mattjones/projects/kptc/trees/tumor_model.txt", sep="\t", index_col=0
)

home_dir = "/data/yosef2/users/mattjones/projects/kptc/trees/"
for tumor in ['3434_NT_T2']:
# for tumor in tqdm(tumor_to_model.index):

    fp = os.path.join(home_dir, tumor, tumor_to_model.loc[tumor, "Networkx"])

    if not os.path.exists(fp):
        continue
    if tumor not in target_trees and len(target_trees):
        continue

    tree = pic.load(open(fp, "rb")).network
    root = [n for n in tree if tree.in_degree(n) == 0][0]
    for e in nx.dfs_edges(tree, source=root):

        tree.nodes[e[1]]["length"] = min(1, e[0].get_modified_hamming_dist(e[1]))

    treenwk = convert_network_to_newick_format(tree)

    infiles.append(treenwk)
    names_trees.append(tumor)

F_empirical = jg.Forest.from_newick(infiles)

print(str(len(F_empirical)) + " trees")

print("Size of trees:")
for name, tree in zip(names_trees, F_empirical.trees):
    print(name, str(len(tree)) + " leaves")

F_empirical.annotate_standard_node_features()

F_empirical.infer_fitness(params={})

for tree, name in zip(F_empirical.trees, names_trees):

    fitnesses = pd.DataFrame()
    fitnesses["mean_fitness"] = None
    for node in tree.T.traverse():
        if node.is_leaf():
            fitnesses.loc[node.name, "mean_fitness"] = node.mean_fitness

    print(name)
    outfp = os.path.join(
        "/data/yosef2/users/mattjones/projects/kptc/trees", name, f"mean_fitness.{name}.txt"
    )
    fitnesses.to_csv(outfp, sep="\t")
