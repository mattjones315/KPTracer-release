import glob
import gzip
import os
import tempfile
import sys
import pickle

import anndata
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from ete3 import Tree

# Read chromosome sizes = 0 to end of last gene
df_genes = pd.read_csv('../data/mm10_gene_ordering.tsv', names=['gene', 'chr', 'start', 'end'], sep='\t', index_col=0)
chromosome_sizes = dict(df_genes.groupby('chr')['end'].max())
# Matplotlib can't plot very large images correctly, so we have to bin
# https://github.com/matplotlib/matplotlib/issues/19276
bin_size = 100
chromosome_order = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
    '16', '17', '18', '19', 'X', 'Y'
]
gene_positions = {chrom: {gene: ((row['start']-1) // bin_size, row['end'] // bin_size) for gene, row in df_chrom.iterrows()} for chrom, df_chrom in df_genes.groupby('chr')}

with open('./data/tumors.txt', 'r') as f:
    tumors = [line.strip() for line in f if not line.isspace()]
    
threshold = 0.1
n_expansions = 0
expansion_cnv_gains = {
    chrom: np.zeros((chromosome_sizes[chrom] - 1) // bin_size, dtype=np.int32)
    for chrom in chromosome_order
}
expansion_cnv_losses = {
    chrom: np.zeros((chromosome_sizes[chrom] - 1) // bin_size, dtype=np.int32)
    for chrom in chromosome_order
}
df_model = pd.read_csv('/lab/solexa_weissman/mgjones/projects/kptc/trees/tumor_model.txt', sep='\t', index_col=0)
for tumor in tumors:
    print(tumor)
    out_dir = f'{tumor}_grouped_0.2'
    
#     tree_cass = Tree(f'/lab/solexa_weissman/mattjones/projects/kptc/trees/{tumor}/{df_model.loc[tumor]["Newick"]}', format=1)
    
    # Load expansions
    expansions = []
    for expansion_path in glob.glob(f'/lab/solexa_weissman/mgjones/projects/kptc/trees/{tumor}/clonal_expansions.{tumor}.expansion*.txt'):
        expansions.append(list(pd.read_csv(expansion_path, sep='\t', index_col=0)['0']))
        n_expansions += 1
        
    if not expansions:
        continue
    
    df_regions = pd.read_csv(f'{out_dir}/HMM_CNV_predictions.HMMi6.rand_trees.hmm_mode-subclusters.Pnorm_0.2.pred_cnv_regions.dat', sep='\t')
    df_cells = pd.read_csv(f'{out_dir}/17_HMM_predHMMi6.rand_trees.hmm_mode-subclusters.cell_groupings', sep='\t')
    df_merged = pd.merge(df_regions, df_cells[~df_cells['cell_group_name'].str.contains('Normal')], on='cell_group_name')
    
    for expansion in expansions:
        gains = {}
        losses = {}
        for chrom, start, end, state in df_merged[df_merged['cell'].isin(expansion)][['chr', 'start', 'end', 'state']].values:
            start = start // bin_size
            end = (end - 1) // bin_size
            chrom = str(chrom)
            if state > 3:
                gains.setdefault(chrom, np.zeros((chromosome_sizes[chrom] - 1) // bin_size, dtype=np.uint16))[start:end] += 1
            elif state < 3:
                losses.setdefault(chrom, np.zeros((chromosome_sizes[chrom] - 1) // bin_size, dtype=np.uint16))[start:end] += 1
                
        for chrom, gain in gains.items():
            expansion_cnv_gains[chrom] += gain > len(expansion) * threshold
        for chrom, loss in losses.items():
            expansion_cnv_losses[chrom] += loss > len(expansion) * threshold
            
with open('../data/expansion_cnv_regions.pkl.gz', 'wb') as f:
    pickle.dump({
        'gains': expansion_cnv_gains,
        'losses': expansion_cnv_losses,
        'bin_size': bin_size,
        'n_expansions': n_expansions,
    }, f)