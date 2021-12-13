import pickle
from collections import Counter

import os
import numpy as np
import pandas as pd
from ete3 import Tree
from scipy import stats
from scipy.cluster import hierarchy

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

from tqdm import tqdm

def collapse_unifurcations(tree: Tree):
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

def find_nearest_neighbors(leaf):
    node = leaf.up
    while len(node.get_leaves()) < 2:
        node = node.up

    min_distance = np.inf
    neighbors = []
    for l in node.iter_leaves():
        if l.name == leaf.name:
            continue

        distance = leaf.get_distance(l)
        assert int(distance) == distance
        distance = int(distance)

        if distance < min_distance:
            neighbors = [l.name]
            min_distance = distance
        elif distance == min_distance:
            neighbors.append(l.name)
    return neighbors, min_distance

def permute(cells, nearest_neighbors, labels, rng=None):
    rng = np.random.default_rng() if rng is None else rng

    bootstrap_labels = labels.copy()
    rng.shuffle(bootstrap_labels)
    bootstrap_cell_labels = {cell: label for cell, label in zip(cells, bootstrap_labels)}
    count = 0
    for leaf, neighbors in nearest_neighbors.items():
        count += sum(bootstrap_cell_labels[neighbor] == bootstrap_cell_labels[leaf] for neighbor in neighbors) / len(neighbors)
    return count / len(nearest_neighbors)

plot_dir = 'plots'
save_dir = 'frozen/nearest_neighbor_bootstrap'

with open('../data/tumors.txt', 'r') as f:
    targets = [line.strip() for line in f if not line.isspace()]
print(targets)

# Read chromosome sizes = 0 to end of last gene
df_genes = pd.read_csv('../data/mm10_gene_ordering_reordered.tsv', names=['gene', 'chr', 'start', 'end'], sep='\t', index_col=0)
chromosome_sizes = dict(df_genes.groupby('chr')['end'].max())
# Matplotlib can't plot very large images correctly, so we have to bin
# https://github.com/matplotlib/matplotlib/issues/19276
gene_positions = {chrom: {gene: (row['start']-1, row['end']) for gene, row in df_chrom.iterrows()} for chrom, df_chrom in df_genes.groupby('chr')}

n_bootstraps = 1000
ps = {}
bootstraps = {}
df_model = pd.read_csv('/lab/solexa_weissman/mgjones/projects/kptc/trees/tumor_model.txt', sep='\t', index_col=0)
for target in targets:
    threshold = '0.2'
    out_dir = f'{target}_grouped{"_0.2" if threshold == "0.2" else ""}'

    # Load Cassiopeia tree
    tree_cass = Tree(f'/lab/solexa_weissman/mgjones/projects/kptc/trees/{target}/{df_model.loc[target]["Newick"]}', format=1)
    tree_cass = collapse_unifurcations(tree_cass)
    leaves = tree_cass.get_leaf_names()

    # Load CNVs
    df_regions = pd.read_csv(f'{out_dir}/HMM_CNV_predictions.HMMi6.rand_trees.hmm_mode-subclusters.Pnorm_0.2.pred_cnv_regions.dat', sep='\t')
    df_cells = pd.read_csv(f'{out_dir}/17_HMM_predHMMi6.rand_trees.hmm_mode-subclusters.cell_groupings', sep='\t')
    df_merged = pd.merge(df_regions, df_cells[~df_cells['cell_group_name'].str.contains('Normal')], on='cell_group_name')

    gene_states = {}
    for cell, chrom, start, end, state in df_merged[['cell', 'chr', 'start', 'end', 'state']].values:
        interval_genes = []
        for gene, (gene_start, gene_end) in gene_positions[str(chrom)].items():
            if start - 1 <= gene_start and end >= gene_end:
                gene_states.setdefault(cell, {})[gene] = state
    for cell in leaves:
        if cell not in gene_states:
            gene_states[cell] = {'placeholder': 3}
    df_counts = pd.DataFrame.from_dict(gene_states, orient='index').fillna(3).astype(int)
    if 'placeholder' in df_counts.columns:
        df_counts.drop(columns='placeholder', inplace=True)

    # Subset CNV counts to only cells in the tree
    df_counts = df_counts[df_counts.index.isin(leaves)]
    cells = list(df_counts.index)

    # Calculate linkage for hierarchical clustering
    linkage = hierarchy.linkage(df_counts.values, method='ward', metric='euclidean')

    # Find nearest neighbors in cassiopeia tree
    pkl_path = os.path.join(save_dir, f'{target}_neighbors.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            nearest_neighbors = pickle.load(f)
    else:
        nearest_neighbors = {}
        for leaf in tree_cass.iter_leaves():
            nearest_neighbors[leaf.name] = find_nearest_neighbors(leaf)[0]
        with open(pkl_path, 'wb') as f:
            pickle.dump(nearest_neighbors, f)

    for t in sorted(np.unique(linkage[:,2])):
        labels = hierarchy.fcluster(linkage, criterion='distance', t=t)
        n_clusters = len(np.unique(labels))

        if n_clusters < 2:
            continue
        print(f'{target} {n_clusters}')

        cell_labels = {cell: label for cell, label in zip(cells, labels)}

        # Count number of nearest neighbors that are in the same cluster
        count = 0
        for leaf, neighbors in nearest_neighbors.items():
            count += sum(cell_labels[neighbor] == cell_labels[leaf] for neighbor in neighbors) / len(neighbors)
        p = count / len(leaves)
        ps.setdefault(target, {})[n_clusters] = p

        bootstrap_ps = Parallel(n_jobs=32, verbose=5)(delayed(permute)(cells, nearest_neighbors, labels, rng=np.random.default_rng(np.random.randint(1e10))) for _ in range(n_bootstraps))
        bootstraps.setdefault(target, {})[n_clusters] = bootstrap_ps



rows = []
for tumor, p_dict in ps.items():
    for n_clusters, p in p_dict.items():
        rows.append([tumor, n_clusters, p, sum(bootstraps[tumor][n_clusters] >= p) / n_bootstraps])
df = pd.DataFrame(rows, columns=['tumor', 'n_clusters', 'nearest_neighbor_probability', 'p'])
df.to_csv('../data/nearest_neighbor_probabilities.csv', index=False)
