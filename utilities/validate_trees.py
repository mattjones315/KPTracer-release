from ete3 import Tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy as sp
import scipy.io as spio

import scipy.stats as st
import pickle as pic
import networkx as nx

from cassiopeia.TreeSolver.Node import Node
from cassiopeia.TreeSolver.Cassiopeia_Tree import Cassiopeia_Tree


from tqdm import tqdm

import functools

import numba
import argparse

import seaborn as sns
import colorcet as cc


def dist_plotter(tree_dists, edit_dists, plot_type, diam, n_targets, out_fp=None):

	plt.figure(1, figsize=(6, 6))
	if plot_type == "2D-Density":
		kde = sns.kdeplot(
			tree_dists, edit_dists, cmap=cc.cm.CET_L19, shade=True, bw=0.02, n_levels=50
		)
	elif plot_type == "2D-Hist":
		hist = plt.hist2d(
			tree_dists, edit_dists, bins=[diam - 1, n_targets], cmap=cc.cm.CET_L19
		)
	elif plot_type == "Scatter":
		scat = plt.scatter(tree_dists, edit_dists, c="r", marker=".", alpha=0.01)
	xlab = plt.xlabel("Phylogenetic Distance")
	ylab = plt.ylabel("Allele Distance")
	ylim = plt.ylim(0, 1)
	xlim = plt.xlim(0, 1)

	if out_fp is not None:
		plt.savefig(out_fp)


def assign_edge_lengths(tree):

	for e in tqdm(tree.edges(), desc="assigning edge lengths"):

		tree[e[0]][e[1]]["length"] = e[0].get_mut_length(e[1])

	return tree


def find_phy_neighbors(g, query, K=1, dist_mat=None):

	if dist_mat is not None:

		phydist = dist_mat.loc[query].min()
		idx = np.where(dist_mat.loc[query].values == phydist)[0]
		neighbor = dist_mat.columns[idx]

	else:
		root = [n for n in g if g.in_degree(n) == 0][0]
		DIST_TO_ROOT = nx.single_source_dijkstra_path_length(g, root, weight="length")

		_leaves = [n for n in g if g.out_degree(n) == 0]

		pairs = [(query, l) for l in _leaves if l != query]
		lcas = nx.algorithms.lowest_common_ancestors.all_pairs_lowest_common_ancestor(
			g, pairs=pairs
		)

		min_dist, neighbor = np.inf, None
		for _lca in lcas:

			n_pair, mrca = _lca[0], _lca[1]

			dA = DIST_TO_ROOT[n_pair[0]]
			dB = DIST_TO_ROOT[n_pair[1]]
			dC = DIST_TO_ROOT[mrca]

			phydist = dA + dB - 2 * dC
			if phydist < min_dist:
				neighbor = n_pair[1]

	return neighbor, phydist


def compute_pairwise_edit_dists(
	g, compare_method=None, meta_item=None, nodes=None, verbose=True
):

	edit_dist = []
	if nodes is not None:
		_leaves = nodes
		n_targets = len(_leaves)
	else:
		root = [n for n in g if g.in_degree(n) == 0][0]
		_leaves = [n for n in g if g.out_degree(n) == 0]

	n_targets = len(_leaves[0].get_character_vec())

	all_pairs = []
	pair_names = []
	for i1 in tqdm(range(len(_leaves)), desc="Creating pairs to compare"):
		l1 = _leaves[i1]
		for i2 in range(i1 + 1, len(_leaves)):
			l2 = _leaves[i2]

			if compare_method == "inter":
				if meta_item.loc[l1.name] != meta_item.loc[l2.name]:
					all_pairs.append((l1, l2))
					pair_names.append((l1.name, l2.name))
			elif compare_method == "intra":
				if meta_item.loc[l1.name] == meta_item.loc[l2.name]:
					all_pairs.append((l1, l2))
					pair_names.append((l1.name, l2.name))

			else:
				all_pairs.append((l1, l2))
				pair_names.append((l1.name, l2.name))

	for p in all_pairs:
		edit_dist.append(p[0].get_modified_hamming_dist(p[1]))

	return np.array(edit_dist), pair_names

def compute_pairwise_phylo_dist_nx(
	g, compare_method=None, pair_names=None, meta_item=None, subset=None, verbose=True
):
	
	tree_dist = []

	root = [n for n in g if g.in_degree(n) == 0][0]

	DIST_TO_ROOT = nx.single_source_dijkstra_path_length(g, root, weight="length")

	if subset:
		_leaves = subset
	else:
		_leaves = [n for n in g if g.out_degree(n) == 0]

	all_pairs = []
	if pair_names is None:
		pair_names = []
		for i1 in tqdm(range(len(_leaves)), desc="Creating pairs to compare"):
			l1 = _leaves[i1]
			for i2 in range(i1 + 1, len(_leaves)):
				l2 = _leaves[i2]

				if compare_method == "inter":
					if meta_item.loc[l1.name] != meta_item.loc[l2.name]:
						all_pairs.append((l1, l2))
						pair_names.append((l1.name, l2.name))
				elif compare_method == "intra":
					if meta_item.loc[l1.name] == meta_item.loc[l2.name]:
						all_pairs.append((l1, l2))
						pair_names.append((l1.name, l2.name))

				else:
					all_pairs.append((l1, l2))
					pair_names.append((l1.name, l2.name))
	else:

		for i1 in tqdm(range(len(_leaves)), desc="Creating pairs to compare from list"):
			l1 = _leaves[i1]
			for i2 in range(i1 + 1, len(_leaves)):
				l2 = _leaves[i2]

				if (l1.name, l2.name) in pair_names or (l2.name, l1.name) in pair_names:
					all_pairs.append((l1, l2))

	if verbose:
		print("Finding LCAs for all pairs...")
	lcas = nx.algorithms.lowest_common_ancestors.all_pairs_lowest_common_ancestor(
		g, pairs=all_pairs
	)

	if verbose:
		print("Computing pairwise distances...")

	for _lca in lcas:

		n_pair, mrca = _lca[0], _lca[1]

		dA = DIST_TO_ROOT[n_pair[0]]
		dB = DIST_TO_ROOT[n_pair[1]]
		dC = DIST_TO_ROOT[mrca]

		tree_dist.append(dA + dB - 2 * dC)

	diam = np.max(tree_dist)

	return np.array(tree_dist), pair_names, diam

def compute_pairwise_dist_nx(
	g, compare_method=None, pair_names=None, meta_item=None, subset=None, verbose=True
):
	"""Computes the phylogenetic and allelic distances of all cells in a tree.

	Computes the distances between all cells as implied by the tree structure (i.e. phylogenetic distance)
	and as implied by the mutation data (i.e. allelic distance). 

	:param g: Networkx tree
	:param compare_method: one of ["inter", "intra" or None]. Tells the algorithm how to create pairs of cells
	to compare with respect to a given meta data item. If "inter" is provided, we'll only look across groups; if 
	"intra" is provided we'll only look within groups. 
	:param pair_names: a list of tuples of cellBC's to compare. E.g. [(cell1, cell2), (cell5, cell6)]
	:param meta_item: a pd.Series mapping each cellBC to a given group.
	:param subset: restrict the analysis to the set of cellBCs specified.
	:param verbose: print out help statements.

	:returns: a list of 4 items - the phylogenetic distances (a np.array), the edit distances (a np.array), 
	the pair names, the diameter (maximum phylogenetic distance), and number of target sites in this tree.
	"""

	tree_dist = []
	edit_dist = []

	root = [n for n in g if g.in_degree(n) == 0][0]

	DIST_TO_ROOT = nx.single_source_dijkstra_path_length(g, root, weight="length")
	n_targets = len([n for n in g][0].get_character_vec())

	if subset:
		_leaves = subset
	else:
		_leaves = [n for n in g if g.out_degree(n) == 0]

	all_pairs = []
	if pair_names is None:
		pair_names = []
		for i1 in tqdm(range(len(_leaves)), desc="Creating pairs to compare"):
			l1 = _leaves[i1]
			for i2 in range(i1 + 1, len(_leaves)):
				l2 = _leaves[i2]

				if compare_method == "inter":
					if meta_item.loc[l1.name] != meta_item.loc[l2.name]:
						all_pairs.append((l1, l2))
						pair_names.append((l1.name, l2.name))
				elif compare_method == "intra":
					if meta_item.loc[l1.name] == meta_item.loc[l2.name]:
						all_pairs.append((l1, l2))
						pair_names.append((l1.name, l2.name))

				else:
					all_pairs.append((l1, l2))
					pair_names.append((l1.name, l2.name))
	else:

		for i1 in tqdm(range(len(_leaves)), desc="Creating pairs to compare from list"):
			l1 = _leaves[i1]
			for i2 in range(i1 + 1, len(_leaves)):
				l2 = _leaves[i2]

				if (l1.name, l2.name) in pair_names or (l2.name, l1.name) in pair_names:
					all_pairs.append((l1, l2))

	if verbose:
		print("Finding LCAs for all pairs...")
	lcas = nx.algorithms.lowest_common_ancestors.all_pairs_lowest_common_ancestor(
		g, pairs=all_pairs
	)

	if verbose:
		print("Computing pairwise distances...")

	for _lca in lcas:

		n_pair, mrca = _lca[0], _lca[1]

		dA = DIST_TO_ROOT[n_pair[0]]
		dB = DIST_TO_ROOT[n_pair[1]]
		dC = DIST_TO_ROOT[mrca]

		tree_dist.append(dA + dB - 2 * dC)
		edit_dist.append(n_pair[0].get_modified_hamming_dist(n_pair[1]))

	diam = np.max(tree_dist)

	return np.array(tree_dist), np.array(edit_dist), pair_names, diam, n_targets


def compute_RNA_corr(counts, cells, _method="euclidean"):

	entries = counts.loc[cells].values

	if _method == "frobenius":

		corrs = sp.spatial.distance.pdist(entries, metric="minkowski", p=2.0)

	elif _method == "euclidean":

		corrs = sp.spatial.distance.pdist(entries, metric="euclidean")

	elif _method == "pearson":

		corrs = sp.spatial.distance.pdist(entries, metric="correlation")

	else:
		raise Exception("Method not recognized: " + _method)

	return corrs


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"tree",
		help="Tree to process. This should be a processed Cassiopeia_Tree object, saved as a pickle",
	)
	parser.add_argument(
		"--expr",
		type=str,
		help="Expression matrix for RNA correlatoins. Assumed to be Cells x Components (genes, PCs, etc.)",
	)
	parser.add_argument("--meta", type=str, help="Meta file")
	parser.add_argument(
		"--meta_item", type=str, help="Column in the meta file to assign to cells."
	)
	parser.add_argument(
		"--inter_meta",
		action="store_true",
		default=False,
		help="compute correlations between cells of different meta assignment",
	)
	parser.add_argument(
		"--intra_meta",
		action="store_true",
		default=False,
		help="compute correlations between cells of the same meta assignment",
	)
	parser.add_argument(
		"--out_fp",
		default=None,
		type=str,
		help="Write out correlations to sparse matrix format at the specified file location.",
	)
	parser.add_argument(
		"--rna_comp_method",
		default="euclidean",
		type=str,
		help="Choose from euclidean, frobenius, or pearson",
	)

	args = parser.parse_args()

	treefp = args.tree
	expr_fp = args.expr
	meta_fp = args.meta
	meta_col = args.meta_item
	out_fp = args.out_fp
	_method = args.rna_comp_method

	tree = pic.load(open(treefp, "rb")).get_network()
	root = [n for n in tree if tree.in_degree(n) == 0][0]
	_leaves = [n for n in tree if tree.out_degree(n) == 0]

	subset = None
	if meta_fp:
		meta = pd.read_csv(meta_fp, sep="\t", index_col=0)

	if expr_fp:
		expr = pd.read_csv(expr_fp, sep="\t", index_col=0)
		subset = [l for l in _leaves if l.name in expr.index.values]

	tree = assign_edge_lengths(tree)

	if args.intra_meta:
		tree_dists, edit_dists, pairs = compute_pairwise_dist_nx(
			tree, compare_method="intra", meta_item=meta[meta_col], subset=subset
		)

	elif args.inter_meta:
		tree_dists, edit_dists, pairs = compute_pairwise_dist_nx(
			tree, compare_method="inter", meta_item=meta[meta_col], subset=subset
		)

	else:
		tree_dists, edit_dists, pairs = compute_pairwise_dist_nx(tree, subset=subset)

	rna_dists = None
	if expr_fp:

		rna_dists = compute_RNA_corr(expr, [n.name for n in subset], _method=_method)

	if out_fp:

		out_stem = ".".join(out_fp.split(".")[:-1])

		# ds_phylo = sp.sparse.csc_matrix(sp.spatial.distance.squareform(tree_dists))
		# ds_hamming = sp.sparse.csc_matrix(sp.spatial.distance.squareform(edit_dists))

		# spio.mmwrite(out_stem + ".phylodist.mtx", ds_phylo)
		# spio.mmwrite(out_stem + ".modifiedhamming.mtx", ds_hamming)

		with open(out_stem + ".phylodist.txt", "wb") as f:
			f.write(tree_dists)

		with open(out_stem + ".editdist.txt", "wb") as f:
			f.write(edit_dists)

		fig = plt.figure(figsize=(7, 7))
		plt.scatter(tree_dists, edit_dists, alpha=0.5)
		plt.xlabel("Phylogenetic Distance")
		plt.ylabel("Allele Distance")
		plt.title("Pearson Corr: " + str(st.stats.pearsonr(tree_dists, edit_dists)[0]))
		plt.savefig(out_stem + ".phylo_vs_allele.png")

		if expr_fp:

			# ds_rna = sp.sparse.csc_matrix(sp.spatial.distance.squareform(rna_dists))
			# spio.mmwrite(out_stem + ".rna_corr.mtx", ds_rna)

			with open(out_stem + "rna_corr.txt", "wb") as f:
				f.write(rna_dists)

			fig = plt.figure(figsize=(7, 7))
			plt.scatter(tree_dists, rna_dists, alpha=0.5)
			plt.xlabel("Phylogenetic Distance")
			plt.ylabel("RNA Distance, " + _method)
			plt.title(
				"Pearson Corr: " + str(st.stats.pearsonr(tree_dists, rna_dists)[0])
			)
			plt.savefig(out_stem + ".phylo_vs_rna.png")

			fig = plt.figure(figsize=(7, 7))
			plt.scatter(edit_dists, rna_dists, alpha=0.5)
			plt.xlabel("Allele Distance")
			plt.ylabel("RNA Distance, " + _method)
			plt.title(
				"Pearson Corr: " + str(st.stats.pearsonr(edit_dists, rna_dists)[0])
			)
			plt.savefig(out_stem + ".alleles_vs_rna.png")
