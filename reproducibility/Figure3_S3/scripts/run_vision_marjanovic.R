require(Matrix)
require(VISION)
options(mc.cores=15)

# Read in RNA counts
RNA.dir = "/data/yosef2/users/mattjones/projects/kptc/RNA/NT_and_SS2/mm10/"
data = readMM(paste0(RNA.dir, "collated_matrix.mtx"))
genes = read.table(paste0(RNA.dir, "genes.tsv"), sep='\t', header=F, stringsAsFactors=F)[,1]
bc = read.table(paste0(RNA.dir,"barcodes.tsv"), sep='\t', header=F, stringsAsFactors=F)[,1]
rownames(data) = genes
colnames(data) = bc

# Read in meta data
meta = read.table("/data/yosef2/users/mattjones/projects/kptc/RNA/KP_SS2/prepared/mmLungPlate_meta.tsv", sep='\t', row.names=1, header=T)
meta = apply(meta, 2, as.factor)

# subset data
data = data[,rownames(meta)]

# filter genes
f.genes = VISION:::filterGenesFano(data)

# read in signatures
createSigFromList <- function(lst, sign=1) {
    data = list()
    for (g in lst) {
        data[[g]] = sign
    }

    return(data)
}

cluster_sigs = c()

fitness.up = read.table("/data/yosef2/users/mattjones/projects/kptc/KPTracer/notebooks/DE/NT/majority_vote.up_genes.txt", sep='\t', row.names=1, stringsAsFactors=F)[,1]
fitness.down = read.table("/data/yosef2/users/mattjones/projects/kptc/KPTracer/notebooks/DE/NT/majority_vote.down_genes.txt", sep='\t', row.names=1, stringsAsFactors=F)[,1]

cluster_sigs = c(cluster_sigs, createGeneSignature(name = 'FitnessDE_Up', sigData=createSigFromList(fitness.up)))
cluster_sigs = c(cluster_sigs, createGeneSignature(name = 'FitnessDE_Down', sigData=createSigFromList(fitness.down)))
cluster_sigs = c(cluster_sigs, createGeneSignature(name = 'FitnessSignature', sigData=c(createSigFromList(fitness.up), createSigFromList(fitness.down, sign=-1))))

my.sigs = c(cluster_sigs, "/data/yosef2/users/mattjones/data/h.all.mm10.gmt",
						"/data/yosef2/users/mattjones/data/c2.reactome.mm10.gmt",
                        "/data/yosef2/users/mattjones/data/c2.cgp.mm10.gmt",
                        "/data/yosef2/users/mattjones/data/c5.bp.mm10.gmt",
                        "/data/yosef2/users/mattjones/data/c6.all.mm10.gmt")

vis <- Vision(data, my.sigs, projection_genes = f.genes, meta=meta, pool=F,
                    projection_methods=c("tSNE30", "UMAP"), sig_gene_threshold=0.01)
vis = analyze(vis, hotspot=F)
