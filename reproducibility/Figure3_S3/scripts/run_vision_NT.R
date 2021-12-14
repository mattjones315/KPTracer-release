require(Matrix)
require(VISION)
require(biomaRt)

human = useMart("ensembl", dataset = "hsapiens_gene_ensembl")
mouse = useMart("ensembl", dataset = "mmusculus_gene_ensembl")

options(mc.cores=15)

# read in data -------------------------------------
home.dir = "/data/yosef2/users/mattjones/projects/kptc/RNA/NT/"
data = readMM(paste0(home.dir, "mm10/matrix.mtx"))
genes = read.table(paste0(home.dir, "mm10/genes.tsv"), sep='\t', row.names=1, header=F, stringsAsFactors=F)[,1]
bc = read.table(paste0(home.dir,"mm10/barcodes.tsv"), sep='\t', header=F, stringsAsFactors=F)[,1]

rownames(data) = genes
colnames(data) = bc

f.genes = read.table(paste0(home.dir, "filtered_genes.txt"), sep='\t', header=F, stringsAsFactors=F)[,1]


latent = read.table(paste0(home.dir, "NT_latent.txt"), sep='\t', header=T, row.names=1)
kii = intersect(bc, rownames(latent))

latent = latent[kii,]
data = data[,kii]

# read in meta information -------------------------------------
meta = read.table("/data/yosef2/users/mattjones/projects/kptc/L6-37_ALL_META_052720.txt", sep='\t',
                    stringsAsFactors = F, header = T, row.names = 1)
meta[is.na(meta$Tumor), 'Tumor'] = 'None'
meta = meta[meta$Tumor != 'None',]
# meta = meta[meta$genotype == 'sgNT',]
meta = meta[meta$genotype != 'SECC_NT',]
meta = meta[!unlist(lapply(meta$Tumor, function(x) grepl('Im', x, fixed=T))),]

meta$Mouse = as.factor(meta$Mouse)
meta$MULTI = as.factor(meta$MULTI)
meta$Lane = as.factor(meta$Lane)
meta$Tumor = as.factor(meta$Tumor)
meta$MetFamily = as.factor(meta$MetFamily)
meta$lentiBC = as.factor(meta$lentiBC)
meta$SubTumor = as.factor(meta$SubTumor)
meta$Aging_time = as.factor(meta$Aging_time)
meta$Aging_Month = as.factor(meta$Aging_Month)
meta$Batch_Library = as.factor(meta$Batch_Library)
meta$Batch_Harvest = as.factor(meta$Batch_Harvest)
meta$genotype = as.factor(meta$genotype)

# intersect by meta ---------------------------
kii = intersect(rownames(meta), colnames(data))
meta = meta[kii,]
data = data[,kii]
latent = latent[kii,]

meta$nUMI = colSums(data)
meta$nGenes = colSums(data > 0)

# normalize
norm.factor = median(colSums(data))
data <- t( t(data) / colSums(data)) * norm.factor

# read in metastatic signatures
createSigFromList <- function(lst, sign=1) {
    data = list()
    for (g in lst) {
        data[[g]] = sign
    }

    return(data)
}


# add cell cycle genes
g1_s = read.table("/data/yosef2/users/mattjones/data/g1_s.txt", sep='\t', stringsAsFactors=F)[,1]
g2_m = read.table("/data/yosef2/users/mattjones/data/g2_m.txt", sep='\t', stringsAsFactors=F)[,1]

mm10_g1s = getLDS(attributes=c("hgnc_symbol"), filters="hgnc_symbol", values=g1_s, mart=human,
                attributesL=c("mgi_symbol"), martL=mouse, uniqueRows=T)[["MGI.symbol"]]
mm10_g2m = getLDS(attributes=c("hgnc_symbol"), filters="hgnc_symbol", values=g2_m, mart=human,
                attributesL=c("mgi_symbol"), martL=mouse, uniqueRows=T)[["MGI.symbol"]]

cluster_sigs = c(cluster_sigs, createGeneSignature(name = 'G1/S', sigData=createSigFromList(mm10_g1s)))
cluster_sigs = c(cluster_sigs, createGeneSignature(name = 'G2/M', sigData=createSigFromList(mm10_g2m)))
cluster_sigs = c(cluster_sigs, createGeneSignature(name = 'CellCycle', sigData=createSigFromList(c(mm10_g1s, mm10_g2m))))

# add fitness gene sets
gene.cluster1 = read.table("/data/yosef2/users/mattjones/projects/kptc/KPTracer/notebooks/DE/NT/fitness_hotspot_module1.txt",
                            sep='\t', row.names = 1, header=T, stringsAsFactors=F)[,1]
gene.cluster2 = read.table("/data/yosef2/users/mattjones/projects/kptc/KPTracer/notebooks/DE/NT/fitness_hotspot_module2.txt",
                            sep='\t', row.names = 1, header=T, stringsAsFactors=F)[,1]
gene.cluster3 = read.table("/data/yosef2/users/mattjones/projects/kptc/KPTracer/notebooks/DE/NT/fitness_hotspot_module3.txt",
                            sep='\t', row.names = 1, header=T, stringsAsFactors=F)[,1]

cluster_sigs = c(cluster_sigs, createGeneSignature(name = 'FitnessMod1', sigData=createSigFromList(gene.cluster1)))
cluster_sigs = c(cluster_sigs, createGeneSignature(name = 'FitnessMod2', sigData=createSigFromList(gene.cluster2)))
cluster_sigs = c(cluster_sigs, createGeneSignature(name = 'FitnessMod3', sigData=createSigFromList(gene.cluster3)))

fitness.up = read.table("/data/yosef2/users/mattjones/projects/kptc/KPTracer/notebooks/DE/NT/majority_vote.up_genes.txt", sep='\t', header=T, row.names=1, stringsAsFactors=F)[,1]
fitness.down = read.table("/data/yosef2/users/mattjones/projects/kptc/KPTracer/notebooks/DE/NT/majority_vote.down_genes.txt", sep='\t', header=T, row.names=1, stringsAsFactors=F)[,1]

cluster_sigs = c(cluster_sigs, createGeneSignature(name = 'FitnessDE_Up', sigData=createSigFromList(fitness.up)))
cluster_sigs = c(cluster_sigs, createGeneSignature(name = 'FitnessDE_Down', sigData=createSigFromList(fitness.down)))
cluster_sigs = c(cluster_sigs, createGeneSignature(name = 'FitnessSignature', sigData=c(createSigFromList(fitness.up), createSigFromList(fitness.down, sign=-1))))

my.sigs = c(cluster_sigs, "/data/yosef2/users/mattjones/data/h.all.mm10.gmt",
						"/data/yosef2/users/mattjones/data/c2.reactome.mm10.gmt",
                        "/data/yosef2/users/mattjones/data/c2.cgp.mm10.gmt",
                        "/data/yosef2/users/mattjones/data/c5.bp.mm10.gmt",
                        "/data/yosef2/users/mattjones/data/c6.all.mm10.gmt")

scanpy_umap = read.table(paste0(home.dir, 'NT_scanpy_umap.txt'), sep='\t', row.names=1, header=T)
scanpy_clusters = read.table(paste0(home.dir, 'NT_leiden_clusters.txt'), sep='\t', row.names=1, header=T)

kii = intersect(rownames(scanpy_clusters), colnames(data))
meta[kii, 'leiden_clusters'] = as.factor(scanpy_clusters[kii, 1])
data = data[,kii]
latent = latent[kii,]

# create vision object
vis <- Vision(data, my.sigs, projection_genes = f.genes, meta=meta, pool=F, latentSpace=latent,
                    projection_methods=c("tSNE30", "UMAP"), sig_gene_threshold=0.01)
vis = addProjection(vis, 'ScanpyUMAP', scanpy_umap[intersect(colnames(data), rownames(scanpy_umap)),])
vis = analyze(vis)

print("Writing data...")
saveRDS(vis, paste0(home.dir, 'NT_vision.rds'))
write.table(vis@SigScores, paste0(home.dir, "NT_sigscores.tsv"), sep='\t')
viewResults(vis, host='0.0.0.0', port=8113, browser=F)
