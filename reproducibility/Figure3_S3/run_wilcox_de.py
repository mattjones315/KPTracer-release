import numpy as np
import pandas as pd
from sklearn import metrics
import scanpy as sc

from tqdm.auto import tqdm

cluster_labels = 'Combined_Clusters'

adata = sc.read_h5ad("/data/yosef2/users/mattjones/projects/kptc/RNA/ALL/adata_processed.all_filtered_combined_assignments.h5ad")

adata_raw = adata.raw.to_adata()

adata_raw.layers['logged'] = adata_raw.X.copy()

print(adata_raw.obs[cluster_labels].nunique(), adata_raw.obs[cluster_labels].unique())

scale_factor = np.median(np.array(adata_raw.X.sum(axis=1)))
sc.pp.normalize_total(adata_raw, target_sum=scale_factor, layers=['logged'])

sc.pp.log1p(adata_raw, layer='logged')

sc.tl.rank_genes_groups(adata_raw, cluster_labels, method='wilcoxon', layer = 'logged', use_raw=False, n_genes=2000, pos_only=True)

# calculate % expressed
result = adata_raw.uns['rank_genes_groups']
groups = result['names'].dtype.names

result['percent_expressed'] = {}
for group in tqdm(groups):
    
    cells = adata_raw.obs[adata_raw.obs[cluster_labels] == group].index.values
    names = result['names'][group]
    pexpr = []
    
    mask = adata_raw[:,names][cells,:].X.todense()
    mask[mask > 0] = 1
    
    sums = np.squeeze(np.asarray(np.sum(mask, axis=0)))
    result['percent_expressed'][group] = (sums / len(cells))
    
result['percent_expressed'][groups[0]]

result['auroc'] = {}
for group in tqdm(groups):
    
    labels = (adata_raw.obs[cluster_labels] == group).values
    
    name_list = result['names'][group]
    
    aurocs = []
    for gene in tqdm(name_list):
        vec=adata_raw[:,[gene]].X
        y_score = vec.todense()
        aurocs.append(metrics.roc_auc_score(labels, y_score))
    result['auroc'][group] = aurocs

result = adata_raw.uns['rank_genes_groups']
groups = result['names'].dtype.names
de_res = pd.DataFrame(
    {'leiden' + group + '_' + key: result[key][group]
    for group in groups for key in ['names', 'logfoldchanges', 'pvals_adj', 'percent_expressed', 'auroc']})

de_res.to_csv('all_de_results.combined_clusters.tsv', sep='\t')