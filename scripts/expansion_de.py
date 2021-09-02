import os

import ete3
import pandas as pd
import scanpy as sc
import numpy as np
import numba

import matplotlib.pyplot as plt 
import seaborn as sns

import scipy as sp
import scipy.stats as scs
import statsmodels.stats.multitest as multi

from tqdm.auto import tqdm

@numba.jit(nopython = True)
def get_log2fc(g1, g2):
    
    fc = (np.mean(g1) + 0.001) / (np.mean(g2) + 0.001)
    return np.log2(fc)

@numba.jit(forceobj=True)
def run_wilcoxon_de(group1, group2, var_names):
    
    res = {}
    for i in tqdm(range(group1.shape[1])):
        pvalue = scs.ranksums(group1.X[:,i].toarray(), group2.X[:,i].toarray())[1]
        res[var_names[i]] = [pvalue, get_log2fc(group1.X[:,i].toarray(), group2.X[:,i].toarray())]
    
    return res


print('reading in data')

adata_raw = sc.read('/data/yosef2/users/mattjones/projects/kptc/RNA/NT/adata_processed.nt.h5ad', cache=False)
adata_raw = adata_raw[(adata_raw.obs['genotype'] != 'SECC_NT') & (adata_raw.obs['genotype'] != 'None')]

print('reading meta data & filtering')
tumor_to_model = pd.read_csv(
    "/data/yosef2/users/mattjones/projects/kptc/trees/tumor_model.txt", sep="\t", index_col=0
)
meta = pd.read_csv("/data/yosef2/users/mattjones/projects/kptc/L6-37_ALL_META_052720.txt", sep='\t', index_col = 0)

print('preprocessing expression data')
# mito_genes = [name for name in adata_raw.var_names if name.startswith('MT-')]
# adata_raw.obs['percent_mito'] = np.sum(adata_raw[:,mito_genes].X, axis=1).A1 / np.sum(adata_raw.X, axis=1).A1
# adata_raw.obs['nUMI'] = adata_raw.X.sum(axis=1).A1
# adata_raw.obs['n_genes'] = np.sum(adata_raw.X > 0, axis=1).A1

adata_raw.obs = adata_raw.obs.merge(meta, left_index = True, right_index=True, how="left")

scale_factor = np.median(np.array(adata_raw.X.sum(axis=1)))
sc.pp.normalize_per_cell(adata_raw, counts_per_cell_after = scale_factor)

thresh = 11
sc.pp.filter_genes(adata_raw, min_cells=thresh)

filtered = ['3430_NT_T1', '3434_NT_T1', '3513_NT_T1', '3703_NT_T1', '3703_NT_T3', '3434_NT_T3', '3451_Lkb1_T1', "3451_Lkb1_T2", '3451_Lkb1_T3',
            '3454_Lkb1_T3', '3457_ApcT3', '3457_Apc_T4', '3458_Apc_T1', '3460_Lkb1_T2', '3464_Lkb1_T1', '3465_Lkb1_T1', '3505_Lkb1_T1', 
           '3508_Apc_T1', '3519_Lkb1_T3', '3732_Lkb1_T1']

cell_to_expansion = {}

base_dir = "/data/yosef2/users/mattjones/projects/kptc/trees/"
for tumor in tqdm(tumor_to_model.index):
    print(tumor)

    out_path = os.path.join(base_dir, tumor, f"expansion_de.{tumor}.tsv")

    if os.path.exists(out_path):
        continue

    if 'NT' not in tumor or "Fam" in tumor or "Met" in tumor or tumor in filtered:
        continue

    tree = ete3.Tree(os.path.join(base_dir, tumor, tumor_to_model.loc[tumor, 'Newick']), 1)

    expansions = pd.read_csv(os.path.join(base_dir, tumor, f"clonal_expansions.{tumor}.txt"), sep='\t', index_col = 0)

    leaves = tree.get_leaf_names()
    expanding_cells = []
    for ind, row in expansions.iterrows():
        
        expanding_cells += (tree & ind).get_leaf_names()
        
    nonexpanding_cells = np.setdiff1d(leaves, expanding_cells)

    # add cells to dictionary
    for e in expanding_cells:
        cell_to_expansion[e] = 'expanding'
    for n in nonexpanding_cells:
        cell_to_expansion[n] = 'nonexpanding'

    # perform DE
    leaves = np.intersect1d(leaves, adata_raw.obs_names)

    expanding_cells = np.intersect1d(expanding_cells, adata_raw.obs_names)
    nonexpanding_cells = np.intersect1d(nonexpanding_cells, adata_raw.obs_names)

    if len(nonexpanding_cells) == 0 or len(expanding_cells) == 0:
        continue

    adata_filt = adata_raw[np.intersect1d(leaves, adata_raw.obs_names),:]

    adata_filt.obs['status'] = None
    adata_filt.obs.loc[expanding_cells, 'status'] = 'expanding'
    adata_filt.obs.loc[nonexpanding_cells, 'status'] = 'nonexpanding'
    adata_filt.obs['status'] = adata_filt.obs['status'].astype('category')

    thresh = 11
    sc.pp.filter_genes(adata_filt, min_cells=thresh)

    gene_variances = np.var(adata_filt.X.toarray(), axis=0)
    adata_filt = adata_filt[:,gene_variances > 0]

    g1, g2 = adata_filt[adata_filt.obs['status'] == 'expanding',:], adata_filt[adata_filt.obs['status'] == 'nonexpanding',:]

    de_res = run_wilcoxon_de(g1, g2, adata_filt.var_names)
    de_res = pd.DataFrame.from_dict(de_res, orient='index')

    de_res.columns = ['Pval', 'log2fc']

    de_res['FDR'] = multi.multipletests(de_res["Pval"], alpha=0.05, method='fdr_bh')[1]

    de_res.to_csv(os.path.join(base_dir, tumor, f"expansion_de.{tumor}.tsv"), sep='\t')

# adata_filt = adata_raw[np.intersect1d(adata_raw.obs_names, list(cell_to_expansion.keys())),:]
# adata_filt.obs['expansion_status'] = adata_filt.obs.index.map(cell_to_expansion)
# g1, g2 = adata_filt[adata_filt.obs['expansion_status'] == 'expanding',:], adata_filt[adata_filt.obs['expansion_status'] == 'nonexpanding',:]

# de_res = run_wilcoxon_de(g1, g2, adata_filt.var_names)
# de_res = pd.DataFrame.from_dict(de_res, orient='index')

# de_res.columns = ['Pval', 'log2fc']

# de_res['FDR'] = multi.multipletests(de_res["Pval"], alpha=0.05, method='fdr_bh')[1]

# de_res.to_csv(f"/data/yosef2/users/mattjones/projects/kptc/KPTracer/expansion_de.all.tsv", sep='\t')

# genedf.to_csv("/home/mattjones/projects/kptc/KPTracer/linregress.NT.txt", sep='\t')
