import os
import sys

import ete3
import networkx as nx
import numba
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm.auto import tqdm

sys.path.append("/data/yosef2/users/mattjones/projects/kptc/KPTracer/notebooks")
from utilities import tree_utilities

from cassiopeia.data import utilities as data_utilities
from cassiopeia.solver import solver_utilities

HOME = "/data/yosef2/users/mattjones/projects/kptc"

tumor2model = pd.read_csv(
    os.path.join(HOME, "trees/tumor_model.txt"), sep="\t", index_col=0
)

adata_all = sc.read_h5ad(
    os.path.join(
        HOME, "RNA/ALL/adata_processed.all_filtered_combined_assignments.h5ad"
    )
)

FILTER_PROP = 0.025
PREPROCESS = False

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
        os.path.join(
            HOME,
            "trees",
            tumor,
            f"phylogenetic_distance_matrix.tsv",
        ),
        sep="\t",
    )

    edit_distance_matrix.to_csv(
        os.path.join(HOME, "trees", tumor, f"edit_distance_matrix.tsv"),
        sep="\t",
    )
