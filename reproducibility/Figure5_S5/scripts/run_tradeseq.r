#' Run Tradeseq DE for Phylotimes in each Fate Cluster

require(Matrix)
require(VISION)
library(tradeSeq)
library(RColorBrewer)
library(SingleCellExperiment)
library(slingshot)
library(clusterExperiment)
library(ggplot2)
library(enrichR)
library(biomaRt)
library(stringr)
library(pheatmap)

set.seed(12312301)

setEnrichrSite("Enrichr") # Human genes

human = useMart("ensembl", dataset = "hsapiens_gene_ensembl")
mouse = useMart("ensembl", dataset = "mmusculus_gene_ensembl")

# set number of threads
BPPARAM <- BiocParallel::bpparam()
BPPARAM$workers <- 4

# control whether or not to save output
WRITE = FALSE

# read in data -------------------------------------
home.dir = paste0("/path/to/KPTracer-Data", "/expression/raw/NT/")
data = readMM(paste0(home.dir, "mm10/matrix.mtx"))
genes = read.table(paste0(home.dir, "mm10/genes.tsv"), sep='\t', row.names=1, header=F, stringsAsFactors=F)[,1]
bc = read.table(paste0(home.dir,"mm10/barcodes.tsv"), sep='\t', header=F, stringsAsFactors=F)[,1]

rownames(data) = genes
colnames(data) = bc

# normalize
norm.factor = median(colSums(data))
data <- t( t(data) / colSums(data)) * norm.factor

# read in fate times ---------------------------------
fate_times1 = read.table("../data/fate_cluster1_treetime.tsv", sep='\t', row.names=1, header=T)
fate_times2 = read.table("../data/fate_cluster2_treetime.tsv", sep='\t', row.names=1, header=T)

colnames(fate_times1) <- c("Fate1")
colnames(fate_times2) <- c("Fate2")

cell_weights1 <- fate_times1
cell_weights1[rownames(fate_times1), "Fate1"] <- 1

cell_weights2 <- fate_times2
cell_weights2[rownames(fate_times2), "Fate2"] <- 1

sub_data1 = data[,rownames(fate_times1)]
f.genes <- VISION:::filterGenesThreshold(sub_data1, as.integer(0.1*ncol(sub_data1)))
sub_data1 <- sub_data1[f.genes,]

sub_data2 = data[,rownames(fate_times2)]
f.genes <- VISION:::filterGenesThreshold(sub_data2, as.integer(0.1*ncol(sub_data2)))
sub_data2 <- sub_data2[f.genes,]


# choose optimal K
icMat1 <- evaluateK(counts = as.matrix(sub_data1), k = 3:15, pseudotime = fate_times1, cellWeights = cell_weights1,
                   nGenes = 300, verbose = T, parallel=T, BPPARAM = BPPARAM)

icMat2 <- evaluateK(counts = as.matrix(sub_data2), k = 3:10, pseudotime = fate_times2, cellWeights = cell_weights2,
                    nGenes = 200, verbose = T, parallel=T, BPPARAM = BPPARAM)

# fit NB models (# knots determined with the evaluateK procedure above)
sce1 <- fitGAM(counts = as.matrix(sub_data1), pseudotime = fate_times1, cellWeights = cell_weights1,
                            nknots = 4, verbose = T, parallel=T, BPPARAM = BPPARAM)

sce2 <- fitGAM(counts = as.matrix(sub_data2), pseudotime = fate_times2, cellWeights = cell_weights2,
               nknots = 5, verbose = T, parallel=T, BPPARAM = BPPARAM)

assoRes1 <-  associationTest(sce1, l2fc = log2(1.5))
assoRes2 <-  associationTest(sce2, l2fc = log2(1.5))

assoRes1 <- assoRes1[!is.na(assoRes1$waldStat),]
assoRes2 <- assoRes2[!is.na(assoRes2$waldStat),]

assoRes1['FDR'] = p.adjust(assoRes1$pvalue, method='fdr')
assoRes2['FDR'] = p.adjust(assoRes2$pvalue, method='fdr')

signif.genes1 <- rownames(assoRes1[which( (assoRes1$FDR < 0.05) & (assoRes1$meanLogFC > 0.5)),])
signif.genes2 <- rownames(assoRes2[which( (assoRes2$FDR < 0.05) & (assoRes2$meanLogFC > 0.5)),])

assoResSignif1 <- assoRes1[signif.genes1,]
assoResSignif2 <- assoRes2[signif.genes2,]

smoothed.profiles.lineage1 = t(as.matrix(c(rep(0, 50)), nrow=1, ncol=50))
smoothed.profiles.lineage2 = t(as.matrix(c(rep(0, 50)), nrow=1, ncol=50))
for (gene in rownames(assoResSignif1)) {
  ysmoothed = predictSmooth(models = sce1, gene = gene, nPoints = 50)
  smoothed.profiles.lineage1 = rbind(smoothed.profiles.lineage1, ysmoothed[, 'yhat'])
}
smoothed.profiles.lineage1 = smoothed.profiles.lineage1[-1,]
rownames(smoothed.profiles.lineage1) <- signif.genes1

for (gene in rownames(assoResSignif2)) {
  ysmoothed = predictSmooth(models = sce2, gene = gene, nPoints = 50)
  smoothed.profiles.lineage2 = rbind(smoothed.profiles.lineage2, ysmoothed[, 'yhat'])
}
smoothed.profiles.lineage2 = smoothed.profiles.lineage2[-1,]
rownames(smoothed.profiles.lineage2) <- signif.genes2

heatmap1 = pheatmap(smoothed.profiles.lineage1, cluster_cols=F, scale='row', show_rownames=F, clustering_method = 'ward')
heatmap2 = pheatmap(smoothed.profiles.lineage2, cluster_cols=F, scale='row', show_rownames = F, clustering_method='ward')

clusters1 = cutree(heatmap1$tree_row,k=2)
clusters2 = cutree(heatmap2$tree_row,k=2)

heatmap1 = pheatmap(smoothed.profiles.lineage1, cluster_cols=F, scale='row', show_rownames=F, clustering_method = 'ward', cutree_rows = 2,
                    annotation_row = as.data.frame(as.factor(clusters1)))

heatmap2 = pheatmap(smoothed.profiles.lineage2, cluster_cols=F, scale='row', show_rownames = F, clustering_method='ward', cutree_rows = 2,
                    annotation_row = as.data.frame(as.factor(clusters2)))

#cluster_to_name = list("1"= "Intermediate", "2" = "Late", "3" = "Early")
cluster_to_name = list("1"= "Early", "2" = "Late")
assoResSignif1[,'Cluster'] = sapply(rownames(assoResSignif1), function(x) cluster_to_name[[clusters1[[x]]]])

cluster_to_name = list("1"= "Early", "2" = "Late")
assoResSignif2[,'Cluster'] = sapply(rownames(assoResSignif2), function(x) cluster_to_name[[clusters2[[x]]]])

if (WRITE) {
  write.table(assoResSignif1, "tradeseqDE_fate1.tsv", sep='\t')
  write.table(assoResSignif2, "tradeseqDE_fate2.tsv", sep='\t')
}

dbs = c("GO_Biological_Process_2018", "ChEA_2016", "MSigDB_Hallmark_2020")
cluster_to_enrichment_fate1 = list()
cluster_to_enrichment_fate2 = list()
for (cluster in unique(assoResSignif1[,'Cluster'])) {
  genes <- rownames(assoResSignif1[assoResSignif1$Cluster == cluster,])
  genes.human <- getLDS(attributes=c("mgi_symbol"), filters="mgi_symbol", values=genes, mart=mouse,
                        attributesL=c("hgnc_symbol"), martL=human, uniqueRows=T)[["HGNC.symbol"]]
  enrichment = enrichr(genes, dbs)
  
  if (WRITE) {
    write.table(enrichment[[1]], paste0("BP_fate1-", cluster, '.tsv'), sep='\t')
    write.table(enrichment[[2]], paste0("ChEA_fate1-", cluster, '.tsv'), sep='\t')
    write.table(enrichment[[3]], paste0("Hallmark_fate1-", cluster, '.tsv'), sep='\t')
  }
}

for (cluster in unique(assoResSignif2[,'Cluster'])) {
  genes <- rownames(assoResSignif2[assoResSignif2$Cluster == cluster,])
  genes.human <- getLDS(attributes=c("mgi_symbol"), filters="mgi_symbol", values=genes, mart=mouse,
                        attributesL=c("hgnc_symbol"), martL=human, uniqueRows=T)[["HGNC.symbol"]]
  enrichment = enrichr(genes, dbs)
  
  if (WRITE) {
    write.table(enrichment[[1]], paste0("BP_fate2-", cluster, '.tsv'), sep='\t')
    write.table(enrichment[[2]], paste0("ChEA_fate2-", cluster, '.tsv'), sep='\t')
    write.table(enrichment[[3]], paste0("Hallmark_fate2-", cluster, '.tsv'), sep='\t')
  }
}


############# Plot marker gene heatmaps ########################

early.genes = c("Lyz2", "Lamp3", "Sftpa1", "Sftpc", "Cd74", "Cxcl15",  "B2m")
fate1.genes = c("Gkn2", "Gdf15", "Yy1", "Flrt3",  "Vim", "Fabp5", "Marcks", "Acta2", "Emp2")
fate2.genes = c("Scgb1a1",  "Pax9", "Prom1", "Sox2", "Itgb4", "Cd24a",  "Gsto1", "Plac8",  "Cldn4", "Tff1")

smoothed.profiles.lineage1 = t(as.matrix(c(rep(0, 100)), nrow=1, ncol=100))
smoothed.profiles.lineage2 = t(as.matrix(c(rep(0, 100)), nrow=1, ncol=100))

gene.names <- c()
for (gene in unique(c(early.genes, fate1.genes, fate2.genes))) {
  if (gene %in% rownames(sub_data1)) {
    ysmoothed = predictSmooth(models = sce1, gene = gene, nPoints = 100)
    smoothed.profiles.lineage1 = rbind(smoothed.profiles.lineage1, ysmoothed[, 'yhat'])
    gene.names <- c(gene.names, gene)
  }
}

smoothed.profiles.lineage1 = smoothed.profiles.lineage1[-1,]
rownames(smoothed.profiles.lineage1) <- gene.names

gene.names <- c()
for (gene in unique(c(early.genes, fate1.genes, fate2.genes))) {
  if (gene %in% rownames(sub_data2)) {
    ysmoothed = predictSmooth(models = sce2, gene = gene, nPoints = 100)
    smoothed.profiles.lineage2 = rbind(smoothed.profiles.lineage2, ysmoothed[, 'yhat'])
    gene.names <- c(gene.names, gene)
  }
}
smoothed.profiles.lineage2 = smoothed.profiles.lineage2[-1,]
rownames(smoothed.profiles.lineage2) <- gene.names

fate.gene.annotations = data.frame(cluster = rep('Early', length(early.genes)))
rownames(fate.gene.annotations) <- early.genes

uniq.fate1 = setdiff(fate1.genes, fate2.genes)
uniq.fate2 = setdiff(fate2.genes, fate1.genes)
fate.overlapping = intersect(fate1.genes, fate2.genes)
fate.gene.annotations = rbind(fate.gene.annotations, data.frame(cluster = rep('Fate1', length(uniq.fate1)), row.names=uniq.fate1))
fate.gene.annotations = rbind(fate.gene.annotations, data.frame(cluster = rep('Fate2', length(uniq.fate2)), row.names=uniq.fate2))
fate.gene.annotations = rbind(fate.gene.annotations, data.frame(cluster = rep('Overlapping', length(fate.overlapping)), row.names=fate.overlapping))

total.smoothed.profiles = merge(smoothed.profiles.lineage1, smoothed.profiles.lineage2,by=0, all=TRUE)
total.smoothed.profiles[is.na(total.smoothed.profiles)] = 0
rownames(total.smoothed.profiles) <- total.smoothed.profiles$Row.names
total.smoothed.profiles <- total.smoothed.profiles[,-1]

heatmap.total = pheatmap(total.smoothed.profiles[rownames(fate.gene.annotations),], cluster_cols=F, cluster_rows=F, scale='row', show_rownames=T, clustering_method = 'ward', show_colnames = F,
                         annotation_row = fate.gene.annotations, main='Total', gaps_col=100, gaps_row=7)
  


