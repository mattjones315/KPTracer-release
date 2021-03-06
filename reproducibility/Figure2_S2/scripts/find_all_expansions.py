import sys
import os

import numpy as np
import pandas as pd

from ete3 import Tree

import pickle as pic
from tqdm import tqdm

sys.path.append("/path/to/utilities")
from utilities import clonal_expansions

import cassiopeia as cas
from cassiopeia.solver import solver_utilities

data_directory = "/path/to/KPTracer-Data"
base_dir = f"{data_directory}/trees/"
save_dir = "/path/to/savedir"
tumor_list = pd.read_csv(f"{data_directory}/trees/tumor_list.txt", sep='\t')

# set parameters for clonal expansion detection
PVAL = 0.01
MIN_CLADE_PROP = 0.15
MIN_DEPTH=1
FIRST_EXPANSION = False
GENOTYPE = 'NT'

for tumor in tqdm(tumor_list['Tumor'].values):

    if 'All' in tumor or 'Met' in tumor or 'Fam' in tumor:
        continue

    if GENOTYPE not in tumor:
        continue

    # tree_fp = f"{data_directory}/trees/{tumor}_tree.nwk"
    tree_fp = f"{data_directory}/trees/nj/{tumor}_tree_nj.processed.tree"
    if not os.path.exists(tree_fp):
        continue

    # print(tumor)
    # tree = Tree(tree_fp, format=1)

    character_matrix = pd.read_csv(f"{data_directory}/{tumor}_character_matrix.txt", sep='\t', index_col = 0)
    print(tumor)

    character_matrix = character_matrix.replace("-", "-1").astype(int)
    tree = Tree(tree_fp, 1)

    tree = solver_utilities.collapse_unifurcations(tree)

    # i = 0
    # for n in tree.traverse():
    #     if n.is_leaf():
    #         continue
    #     else:
    #         n.name = f'internal{i}'
    #         i += 1

    # cas_tree = cas.data.CassiopeiaTree(tree=tree, character_matrix=character_matrix)

    # cas_tree.collapse_mutationless_edges(infer_ancestral_characters=True)

    # tree = Tree(cas_tree.get_newick())

    # i = 0
    # for n in tree.traverse():
    #     if n.is_leaf():
    #         continue
    #     else:
    #         n.name = f'internal{i}'
    #         i += 1

    tree, expansions = clonal_expansions.detect_expansion(tree,
                        pval=PVAL,
                        min_depth=MIN_DEPTH,
                        _first=FIRST_EXPANSION,
                        min_clade_prop = MIN_CLADE_PROP)

    outfp = os.path.join(save_dir, tumor, f"clonal_expansions.{tumor}.nj.txt")
    # outfp = os.path.join(save_dir, tumor, f"clonal_expansions.{tumor}.txt")
    expansions.to_csv(outfp, sep = '\t')

    # j = 1
    # for n in expansions.index:
    #    pd.DataFrame((tree&n).get_leaf_names()).to_csv(os.path.join(base_dir, tumor, f"clonal_expansions.{tumor}.expansion{j}.txt"), sep='\t')
    #    j += 1
