import os
import sys

import ete3
import networkx as nx
import numba
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm.auto import tqdm

import tree_utilities

adata = sc.read_h5ad(
    '/data/yosef2/users/mattjones/projects/kptc/KPTracer-Data/expression/adata_processed.nt.h5ad'
)
tumor2model = pd.read_csv("/data/yosef2/users/mattjones/projects/kptc/trees/tumor_model.txt", sep='\t', index_col = 0)

FILTER_PROP = 0.025
PREPROCESS = True

cluster_column = "leiden_sub"

uniq_states = adata.obs[cluster_column].unique()

tumor_z_df = {}
observed_df = {}

tumors = [
    t
    for t in tumor2model.index
    if "Fam" not in t
    and "Met" not in t
    and "All" not in t
    and 'NT' in t
    and t.split("_")[2].startswith("T")
]
for tumor in tqdm(tumors):

    print(tumor)

    graph = tree_utilities.prepare_tumor_tree(
        tumor,
        adata,
        tree_dir = "/data/yosef2/users/mattjones/projects/kptc/KPTracer-Data/trees",
        column=cluster_column,
        preprocess=PREPROCESS,
    )

    if PREPROCESS:
        for u, v in graph.edges():
            _length = graph[u][v]["length"]
            if _length > 0:
                _length = 1
            graph[u][v]["length"] = _length

    (
        phylogenetic_distance_matrix,
        leaf_pairs,
        edit_distance_matrix,
        tree_diameter,
        n_targets,
    ) = tree_utilities.compute_pairwise_dist_nx(graph)

    np.fill_diagonal(phylogenetic_distance_matrix.values, 0)
    np.fill_diagonal(edit_distance_matrix.values, 0)

    phylogenetic_distance_matrix.to_csv(
        f"/data/yosef2/users/mattjones/projects/kptc/KPTracer-release/reproducibility/Figure5_S5/data/{tumor}_phylogenetic_distance_matrix.tsv",
        sep="\t",
    )

    edit_distance_matrix.to_csv(
        f"/data/yosef2/users/mattjones/projects/kptc/KPTracer-release/reproducibility/Figure5_S5/data/{tumor}_edit_distance_matrix.tsv",
        sep="\t",
    )
