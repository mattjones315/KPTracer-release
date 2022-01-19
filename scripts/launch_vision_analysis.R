require(Matrix)
require(VISION)
require(biomaRt)
library(reticulate)

scanpy = import("scanpy") 

human = useMart("ensembl", dataset = "hsapiens_gene_ensembl")
mouse = useMart("ensembl", dataset = "mmusculus_gene_ensembl")

options(mc.cores=15)

# read in data -------------------------------------
adata = scanpy$read_h5ad("/data/yosef2/users/mattjones/projects/kptc/KPTracer-Data/expression/adata_processed.nt.h5ad")

counts = adata$raw$X
rownames(counts) <- as.character(adata$raw$obs_names$to_list())
colnames(counts) <- as.character(adata$raw$var_names$to_list())

# read in meta information -------------------------------------
# read in meta
zero_one_normalize <- function(array) {
        max.val = max(array)
        min.val = min(array)

        norm.vals = unlist(lapply(array, function(x) {
                (x - min.val) / (max.val - min.val)
        }))

        return(norm.vals)
}

plasticity = read.table("/data/yosef2/users/mattjones/projects/kptc/KPTracer-release/reproducibility/Figure4_S4/data/plasticity_scores.tsv", sep='\t', header=T, row.names=1)
meta = as.data.frame(adata$obs)
meta <- meta[rownames(counts),]
meta <- meta[, c('Lane', 'Tumor', 'SubTumor', 'ES_clone', 'Mouse',
       'Batch_Library', 'Aging_Time', 'MetFamily', 'PercentUncut', 'n_counts',
       'leiden_sub')]

# subset to plasticity index
kii = intersect(rownames(meta), rownames(plasticity))
meta = meta[kii,]
counts = counts[kii,]
plasticity = plasticity[kii,]
tree = keep.tip(tree, kii)
meta[rownames(plasticity), colnames(plasticity)] = plasticity
meta$scPlasticity <- zero_one_normalize(meta$scPlasticity)
meta$scPlasticityL2 <- zero_one_normalize(meta$scPlasticityL2)
meta$scPlasticityAllele <- zero_one_normalize(meta$scPlasticityAllele)

meta$leiden <- as.factor(meta$leiden_sub, levels=uniquemeta($leiden_sub))
meta$Batch_Library <- as.factor(meta$Batch_Library, levels=uniquemeta($Batch_Library))
meta$Tumor <- as.factor(meta$Tumor, levels=uniquemeta($Tumor))
meta$SubTumor <- as.factor(meta$SubTumor, levels=uniquemeta($SubTumor))
meta$ES_clone <- as.factor(meta$ES_clone, levels=uniquemeta($ES_clone))
meta$Mouse <- as.factor(meta$Mouse, levels=uniquemeta($Mouse))

print(str(meta))

# specify signatures
sigs = c("/data/yosef2/users/mattjones/data/h.all.mm10.gmt",
        "/data/yosef2/users/mattjones/data/c5.bp.mm10.gmt")

# read in metastatic signatures
createSigFromList <- function(lst, sign=1) {
    data = list()
    for (g in lst) {
        data[[g]] = sign
    }

    return(data)
}

# add fitness gene sets
gene.cluster1 = read.table("/data/yosef2/users/mattjones/projects/kptc/KPTracer-release/reproducibility/Figure3_S3/data/hotspot_module1.txt",
                            sep='\t', row.names = 1, header=T, stringsAsFactors=F)[,1]
gene.cluster2 = read.table("/data/yosef2/users/mattjones/projects/kptc/KPTracer-release/reproducibility/Figure3_S3/data/hotspot_module2.txt",
                            sep='\t', row.names = 1, header=T, stringsAsFactors=F)[,1]
gene.cluster3 = read.table("/data/yosef2/users/mattjones/projects/kptc/KPTracer-release/reproducibility/Figure3_S3/data/hotspot_module3.txt",
                            sep='\t', row.names = 1, header=T, stringsAsFactors=F)[,1]

sigs = c(sigs, createGeneSignature(name = 'FitnessMod1', sigData=createSigFromList(gene.cluster1)))
sigs = c(sigs, createGeneSignature(name = 'FitnessMod2', sigData=createSigFromList(gene.cluster2)))
sigs = c(sigs, createGeneSignature(name = 'FitnessMod3', sigData=createSigFromList(gene.cluster3)))

fitness.up = read.table("/data/yosef2/users/mattjones/projects/kptc/KPTracer-release/reproducibility/Figure3_S3/data/majority_vote.up_genes.sgNT.txt", sep='\t', header=T, row.names=1, stringsAsFactors=F)[,1]
fitness.down = read.table("/data/yosef2/users/mattjones/projects/kptc/KPTracer-release/reproducibility/Figure3_S3/data/majority_vote.down_genes.sgNT.txt", sep='\t', header=T, row.names=1, stringsAsFactors=F)[,1]

sigs = c(sigs, createGeneSignature(name = 'FitnessDE_Up', sigData=createSigFromList(fitness.up)))
sigs = c(sigs, createGeneSignature(name = 'FitnessDE_Down', sigData=createSigFromList(fitness.down)))
sigs = c(sigs, createGeneSignature(name = 'FitnessSignature', sigData=c(createSigFromList(fitness.up), createSigFromList(fitness.down, sign=-1))))

# library-size normalize 
counts <- t(as.matrix(counts))
norm.factor = median(colSums(counts))
counts <- t( t(counts) / colSums(counts)) * norm.factor

# get embedding
umap <- as.data.frame(adata$obsm['X_umap'])
rownames(umap) <- as.character(adata$raw$obs_names$to_list())

scvi <- as.data.frame(adata$obsm['X_scVI'])
rownames(scvi) <- as.character(adata$raw$obs_names$to_list())

counts <- Matrix(counts)
f.genes = VISION:::filterGenesFano(counts)

# create vision object
vis <- Vision(counts, sigs, projection_genes = f.genes, meta=meta, pool=F, latentSpace=scvi,
                    projection_methods=c("tSNE30", "UMAP"), sig_gene_threshold=0.01, num_neighbors=30)
vis <- addProjection(vis, 'scanpyUMAP', umap)
vis = analyze(vis)
