import os
import sys

from collections import defaultdict
import numpy as np
import pandas as pd
import random
import scanpy as sc
from tqdm.auto import tqdm

import tree_utilities

HOME = "/data/yosef2/users/mattjones/projects/KPTracer-release/reproducibility/Figure5_S5/data"

tumor2model = pd.read_csv(
    os.path.join(HOME, "trees/tumor_model.txt"), sep="\t", index_col=0
)
adata_all = sc.read_h5ad(
    os.path.join(
        HOME, "RNA/ALL/adata_processed.all_filtered_combined_assignments.h5ad"
    )
)

FILTER_PROP = 0.025  # minimum state proportion to be considered in evo coupling
B = 2000  # number of random couplings to sample
K = 10  # number of neighbors for KNN evolutionary coupling
SEED = 1232135 # random seed

# set random seeds
np.random.seed(SEED)
random.seed(SEED)

cluster_column = "Combined_Clusters"

uniq_states = adata_all.obs[cluster_column].unique()

tumor_z_df = {}
observed_df = {}

tumors = [
    t
    for t in tumor2model.index
    if "Fam" not in t
    and "Met" not in t
    and "All" not in t
    and t.split("_")[2].startswith("T")
]
for tumor in tqdm(tumors):

    print(tumor)

    graph = tree_utilities.prepare_tumor_tree(
        tumor,
        adata_all,
        tumor2model,
        column=cluster_column,
        FILTER_PROP=FILTER_PROP,
        preprocess=False,
    )

    phylogenetic_distance_matrix = pd.read_csv(
        os.path.join(HOME, f"{tumor}_phylogenetic_distance_matrix.tsv"),
        sep="\t",
        index_col=0,
    )
    edit_distance_matrix = pd.read_csv(
        os.path.join(HOME, f"{tumor}_edit_distance_matrix.tsv"),
        sep="\t",
        index_col=0,
    )

    _leaves = [n for n in graph if graph.out_degree(n) == 0]
    _leaves = np.intersect1d(phylogenetic_distance_matrix.index.values, _leaves)

    phylogenetic_distance_matrix = phylogenetic_distance_matrix.loc[_leaves, _leaves]
    edit_distance_matrix = edit_distance_matrix.loc[_leaves, _leaves]

    leaf_states = adata_all.obs.loc[_leaves, "Combined_Clusters"]

    phylo_inter_cluster_df = tree_utilities.get_inter_cluster_df(
        leaf_states, phylogenetic_distance_matrix
    )
    edit_inter_cluster_df = tree_utilities.get_inter_cluster_df(
        leaf_states, edit_distance_matrix
    )
    kth_inter_cluster_df = tree_utilities.get_inter_cluster_df(
        leaf_states,
        phylogenetic_distance_matrix,
        tree_utilities.average_nn_dist,
        k=K,
    )

    phylo_background = defaultdict(list)
    edit_background = defaultdict(list)
    kth_background = defaultdict(list)

    for _ in tqdm(range(B)):
        permuted_assignments = leaf_states.copy()
        permuted_assignments.index = np.random.permutation(
            leaf_states.index.values
        )
        bg_df = tree_utilities.get_inter_cluster_df(
            permuted_assignments, phylogenetic_distance_matrix
        )

        for s1 in bg_df.index:
            for s2 in bg_df.index:
                phylo_background[(s1, s2)].append(bg_df.loc[s1, s2])

        bg_df = tree_utilities.get_inter_cluster_df(
            permuted_assignments, edit_distance_matrix
        )

        for s1 in bg_df.index:
            for s2 in bg_df.index:
                edit_background[(s1, s2)].append(bg_df.loc[s1, s2])

        bg_df = tree_utilities.get_inter_cluster_df(
            permuted_assignments,
            phylogenetic_distance_matrix,
            func=tree_utilities.average_nn_dist,
            k=K,
        )

        for s1 in bg_df.index:
            for s2 in bg_df.index:
                kth_background[(s1, s2)].append(bg_df.loc[s1, s2])

    phylo_null_means = phylo_inter_cluster_df.copy()
    phylo_null_sds = phylo_inter_cluster_df.copy()

    edit_null_means = edit_inter_cluster_df.copy()
    edit_null_sds = edit_inter_cluster_df.copy()

    kth_null_means = kth_inter_cluster_df.copy()
    kth_null_sds = kth_inter_cluster_df.copy()

    for s1 in phylo_null_means.index:
        for s2 in phylo_null_means.columns:
            phylo_null_means.loc[s1, s2] = np.mean(phylo_background[(s1, s2)])
            phylo_null_sds.loc[s1, s2] = np.std(phylo_background[(s1, s2)])

            edit_null_means.loc[s1, s2] = np.mean(edit_background[(s1, s2)])
            edit_null_sds.loc[s1, s2] = np.std(edit_background[(s1, s2)])

            kth_null_means.loc[s1, s2] = np.mean(kth_background[(s1, s2)])
            kth_null_sds.loc[s1, s2] = np.std(kth_background[(s1, s2)])

    phylo_zscores = phylo_inter_cluster_df.copy()
    edit_zscores = edit_inter_cluster_df.copy()
    kth_zscores = kth_inter_cluster_df.copy()

    delta = 0.001
    for ind in phylo_zscores.index:
        for col in phylo_zscores.columns:
            phylo_zscores.loc[ind, col] = (
                delta + phylo_zscores.loc[ind, col] - phylo_null_means.loc[ind, col]
            ) / (delta+phylo_null_sds.loc[ind, col])
            edit_zscores.loc[ind, col] = (
                delta+edit_zscores.loc[ind, col] - edit_null_means.loc[ind, col]
            ) / (delta+edit_null_sds.loc[ind, col])
            kth_zscores.loc[ind, col] = (
                delta+kth_zscores.loc[ind, col] - kth_null_means.loc[ind, col]
            ) / (delta+kth_null_sds.loc[ind, col])

    phylo_zscores.fillna(0, inplace=True)
    edit_zscores.fillna(0, inplace=True)
    kth_zscores.fillna(0, inplace=True)

    mu = np.mean(phylo_zscores.values.ravel())
    sigma = np.std(phylo_zscores.values.ravel())
    phylo_zscores_norm = phylo_zscores.apply(lambda x: (x - mu) / sigma, axis=1)

    mu = np.mean(edit_zscores.values.ravel())
    sigma = np.std(edit_zscores.values.ravel())
    edit_zscores_norm = edit_zscores.apply(lambda x: (x - mu) / sigma, axis=1)

    mu = np.mean(kth_zscores.values.ravel())
    sigma = np.std(kth_zscores.values.ravel())
    kth_zscores_norm = kth_zscores.apply(lambda x: (x - mu) / sigma, axis=1)

    phylo_zscores_norm.to_csv(
        os.path.join(
            HOME, f"evolutionary_coupling.{tumor}.txt"
        ),
        sep="\t",
    )

    edit_zscores_norm.to_csv(
        os.path.join(
            HOME, f"evolutionary_coupling.{tumor}.edit_preprocessed.txt"
        ),
        sep="\t",
    )

    kth_zscores_norm.to_csv(
        os.path.join(
            HOME, f"evolutionary_coupling.{tumor}.knn_preprocessed.txt"
        ),
        sep="\t",
    )
