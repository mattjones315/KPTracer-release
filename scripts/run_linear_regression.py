import pandas as pd
import scanpy as sc
import numpy as np
import os

import matplotlib.pyplot as plt 
import seaborn as sns

import scipy as sp
import scipy.stats as scs
import statsmodels.stats.multitest as multi

from tqdm.auto import tqdm

def run_lin_reg_ALL_GENES(counts, meta_var, covars = []):

    pvals = {}
    betas = {}
    corrs = {}
    scorrs = {}

    meta_scores = counts.obs[meta_var]

    for gene in tqdm(counts.var_names):
        ex = counts.X[:, counts.var_names == gene].toarray()[:,0]
        slope, intercept, r_value, p_value, std_err = scs.linregress(meta_scores, ex)

        pvals[gene] = p_value
        betas[gene] = slope
        corrs[gene] = r_value

        scorrs[gene] = scs.spearmanr(meta_scores, ex)[0]

    return pvals, betas, corrs, scorrs

print('reading in data')
path = "/data/yosef2/users/mattjones/projects/kptc/RNA/ALL/mm10/"
adata = sc.read(path + "matrix.mtx").T
genes = pd.read_csv(path + "genes.tsv", header=None, sep='\t')
adata.var_names = genes[1]
adata.var['gene_ids'] = genes[0]  # add the gene ids as annotation of the variables/genes
adata.obs_names = pd.read_csv(path + 'barcodes.tsv', header=None)[0]
adata.var_names_make_unique()
all_genes = adata.var_names

print('reading meta data & filtering')
tumor_to_model = pd.read_csv(
    "/data/yosef2/users/mattjones/projects/kptc/trees/tumor_model.txt", sep="\t", index_col=0
)
meta = pd.read_csv("/data/yosef2/users/mattjones/projects/kptc/L6-37_ALL_META_052720.txt", sep='\t', index_col = 0)

print('preprocessing expression data')
adata.var['mt'] = adata.var_names.str.startswith('mt-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

adata.obs = adata.obs.merge(meta, left_index = True, right_index=True, how="left")

scale_factor = np.median(np.array(adata.X.sum(axis=1)))
sc.pp.normalize_per_cell(adata, counts_per_cell_after = scale_factor)

adata.raw = adata

sc.pp.log1p(adata)

print('performing linear regressions')

home_dir = "/data/yosef2/users/mattjones/projects/kptc/trees/"

for tumor in tqdm(tumor_to_model.index):

    fitness_fp = "/data/yosef2/users/mattjones/projects/kptc/trees/{}/mean_fitness.{}.txt".format(tumor, tumor)
    if not os.path.exists(fitness_fp):
        continue
    
    fitness = pd.read_csv(fitness_fp, sep='\t', index_col = 0)
        
    ts_rna_overlap = np.intersect1d(adata.obs_names, fitness.index)
    fitness = fitness.loc[ts_rna_overlap]
    adata_sub = adata[ts_rna_overlap,:]
    adata_sub.var_names_make_unique()

    adata_sub.obs['Fitness']= fitness.mean_fitness
    
    thresh = 11
    sc.pp.filter_genes(adata_sub, min_cells=thresh)
    
    print(tumor, adata_sub.shape)

    pvals, betas, corrs, scorrs = run_lin_reg_ALL_GENES(adata_sub, 'Fitness', covars = ['n_genes', 'percent_mito'])
    
    genedf = pd.DataFrame.from_dict(pvals, orient="index", columns=['pvalues'])
    genedf["betas"] = list(map(lambda x: betas[x], pvals.keys()))
    genedf["Corr"] = list(map(lambda x: corrs[x], pvals.keys()))
    genedf["SpearmanCorr"] = list(map(lambda x: scorrs[x], pvals.keys()))
    genedf["FDR"] = multi.multipletests(genedf["pvalues"], alpha=0.05, method='fdr_bh')[1]
        
    genedf.to_csv("/data/yosef2/users/mattjones/projects/kptc/trees/{}/linregress.{}.txt".format(tumor, tumor), sep='\t')
