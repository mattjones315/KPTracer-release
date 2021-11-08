import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ete3 import Tree
from tqdm.auto import tqdm
from scipy import stats


all_expanding_counts = {}
all_non_expanding_counts = {}
for target in sorted(targets):
    print(target)
    threshold = '0.2'
    out_dir = f'{target}_grouped_{threshold}'  # inferCNV output directory
    
    # Load CNVs
    df_regions = pd.read_csv(f'{out_dir}/HMM_CNV_predictions.HMMi6.rand_trees.hmm_mode-subclusters.Pnorm_{threshold}.pred_cnv_regions.dat', sep='\t')
    df_cells = pd.read_csv(f'{out_dir}/17_HMM_predHMMi6.rand_trees.hmm_mode-subclusters.cell_groupings', sep='\t')
    
    # Load Cassiopeia tree
    df_model = pd.read_csv('/lab/solexa_weissman/mgjones/projects/kptc/trees/tumor_model.txt', sep='\t', index_col=0)
    tree_cass = Tree(f'/lab/solexa_weissman/mgjones/projects/kptc/trees/{target}/{df_model.loc[target]["Newick"]}', format=1)
    leaves = tree_cass.get_leaf_names()
    
    # Load expansions
    expansions = []
    for expansion_path in sorted(glob.glob(f'/lab/solexa_weissman/mgjones/projects/kptc/trees/{target}/clonal_expansions.{target}.expansion*.txt')):
        expansions.append(list(pd.read_csv(expansion_path, sep='\t', index_col=0)['0']))
    
    # Cells present in both trees
    cells = set(df_cells['cell']).intersection(leaves)

    cnv_counts = dict(pd.merge(df_regions, df_cells, on='cell_group_name')[['cell', 'cnv_name']].groupby('cell').size())
    all_counts = np.array([cnv_counts.get(cell, 0) for cell in leaves])
    
    if len(expansions) > 0:
        all_expanding = set.union(*[set(expansion) for expansion in expansions])
        all_non_expanding = set(leaves) - all_expanding
        
        non_expanding_counts = np.array([cnv_counts.get(cell, 0) for cell in all_non_expanding])
        filtered_non_expanding_counts = non_expanding_counts[
            np.abs(non_expanding_counts - np.mean(all_counts)) <= 3 * np.std(all_counts)
        ]
        all_non_expanding_counts[target] = filtered_non_expanding_counts
        for i, expansion in enumerate(expansions):
            expanding_counts = np.array([cnv_counts.get(cell, 0) for cell in leaves if cell in expansion])
            filtered_expanding_counts = expanding_counts[
                np.abs(expanding_counts - np.mean(all_counts)) <= 3 * np.std(all_counts)
            ]
            all_expanding_counts.setdefault(target, []).append(filtered_expanding_counts)
    else:
        non_expanding_counts = np.array([cnv_counts.get(cell, 0) for cell in leaves])
        filtered_non_expanding_counts = non_expanding_counts[
            np.abs(non_expanding_counts - np.mean(all_counts)) <= 3 * np.std(all_counts)
        ]
        all_non_expanding_counts[target] = filtered_non_expanding_counts

rows = []
for tumor in targets:
    non_expanding_counts = all_non_expanding_counts[tumor]
    rows.extend([[tumor, np.nan, 'non-expanding', count] for count in non_expanding_counts])
    if tumor not in all_expanding_counts:
        continue
    for i, expanding_counts in enumerate(all_expanding_counts[tumor]):
        rows.extend([[tumor, i+1, 'expanding', count] for count in expanding_counts])
pd.DataFrame(rows, columns=['tumor', 'expansion', 'type', 'count']).to_csv('../data/cnvs_per_cell.csv', index=False)