library(ape)
library(reticulate)
library(VISION)
library(Matrix)
suppressMessages(suppressWarnings(require(dplyr, quietly=T)))

# load in scanpy
scanpy = import("scanpy") 

# read in arguments
args = commandArgs(trailingOnly=TRUE)
tumor_name = args[[1]]
path_to_kptracer_data = args[[2]]
signature_directory = args[[3]]
save_path = args[[4]]

# read in expression data
message(">>> Reading in data.")
adata = scanpy$read_h5ad(paste0(path_to_kptracer_data, "/expression/adata_processed.combined.h5ad"))

counts = adata$raw$X
rownames(counts) <- as.character(adata$raw$obs_names$to_list())
colnames(counts) <- as.character(adata$raw$var_names$to_list())

# read in tree
tree = read.tree(paste0(path_to_kptracer_data, "trees/", tumor_name, "_tree.nwk"))
tree$node.label <- unlist(lapply(1:tree$Nnode, function(x) paste0('internal', x)))
kii = intersect(tree$tip.label, rownames(counts))
tree = keep.tip(tree, kii)
tree = collapse.singles(tree)
counts = counts[tree$tip.label,]

# read in meta
zero_one_normalize <- function(array) {
        max.val = max(array)
        min.val = min(array)

        norm.vals = unlist(lapply(array, function(x) {
                (x - min.val) / (max.val - min.val)
        }))

        return(norm.vals)
}

message(">>> Reading in meta data.")
plasticity = read.table(paste0(path_to_kptracer_data, "plasticity_scores.tsv"), sep='\t', header=T, row.names=1)
fitness = read.table(paste0(path_to_kptracer_data, "/fitnesses/mean_fitness.", tumor_name, ".txt"), sep='\t', header=T, row.names=1)

meta = as.data.frame(adata$obs)
meta <- meta[rownames(counts),]
meta <- meta[, c('PercentUncut', 'n_genes_by_counts', 'total_counts',
                'pct_counts_mt', 'n_genes', 'Cluster')]

# subset to plasticity index
kii = intersect(rownames(meta), rownames(plasticity))
meta = meta[kii,]
counts = counts[kii,]
plasticity = plasticity[kii,]
tree = keep.tip(tree, kii)
meta[kii, 'scFitness'] = fitness[kii, 'mean_fitness']
meta[rownames(plasticity), colnames(plasticity)] = plasticity
meta$Cluster <- factor(as.character(meta$Cluster), levels=unique(as.character(meta$Cluster)))

# normalize
meta$scPlasticity <- zero_one_normalize(meta$scPlasticity)
meta$scPlasticityL2 <- zero_one_normalize(meta$scPlasticityL2)
meta$scPlasticityAllele <- zero_one_normalize(meta$scPlasticityAllele)
meta$scFitness <- zero_one_normalize(meta$scFitness)


# specify signatures
sigs = c()
for (file in list.files(signature_directory)) {
    sigs <- c(sigs, paste0(signature_directory, "/", file))
}

createSigFromList <- function(lst, sign=1) {

    data = list()
    for (g in lst) {
        data[[g]] = sign
    }

    return(data)
}
fitness.up = read.table(paste0(path_to_kptracer_data, "/expression/majority_vote.up_genes.sgNT.txt"),
            sep='\t', header=T, row.names=1, stringsAsFactors=F)[,1]
fitness.down = read.table(paste0(path_to_kptracer_data, "/expression/majority_vote.down_genes.sgNT.txt"),
            sep='\t', header=T, row.names=1, stringsAsFactors=F)[,1]
sigs = c(sigs, createGeneSignature(name = 'FitnessSignature',
            sigData=c(createSigFromList(fitness.up),
                createSigFromList(fitness.down, sign=-1))
            )
        )

# library-size normalize
message(">>> Normalizing.")
counts <- t(as.matrix(counts))
norm.factor = median(colSums(counts))
counts <- t( t(counts) / colSums(counts)) * norm.factor

# get embedding
umap <- as.data.frame(adata$obsm['X_umap'])
rownames(umap) <- as.character(adata$raw$obs_names$to_list())
umap <- umap[tree$tip.label,]

scvi <- as.data.frame(adata$obsm['X_scVI'])
rownames(scvi) <- as.character(adata$raw$obs_names$to_list())
scvi <- scvi[tree$tip.label,]

counts <- Matrix(counts)
f.genes = VISION:::filterGenesFano(counts)

vis <- PhyloVision(tree=tree, data=counts, signatures=sigs,
              meta=meta, num_neighbors=15, projection_genes = f.genes,
              projection_methods=c("tSNE30", 'UMAP'), latentSpace=scvi)
vis <- addProjection(vis, 'scanpyUMAP', umap)

vis = phyloAnalyze(vis)
saveRDS(vis, save_path)


