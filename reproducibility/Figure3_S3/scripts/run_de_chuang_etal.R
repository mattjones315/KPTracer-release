require(limma)
require(edgeR)
require(biomaRt)

mouse = useMart("ensembl", dataset = "mmusculus_gene_ensembl")

# read in tpm data
tpm.data <- read.table("chuang_tpm.txt", sep='\t', header=T, row.names=1, check.names=F)
tpm.data <- tpm.data[!is.na(tpm.data$GeneSymbol),]

# make design matrix
tumor.meta <- read.table("tumor_meta.tsv", sep='\t', row.names=1, header=T)

# DE TEST 1: KPT-E vs Tnonmet and Tmet
tumor.meta.timing <- tumor.meta[which(tumor.meta$TumorType %in% c("KPT-E", "KT", "TnonMet", "TMet")),,drop=F]
tumor.meta.timing = tumor.meta.timing[-which(tumor.meta.timing$TumorType == 'KT'),]
tumor.meta.timing$Group <- apply(tumor.meta.timing, 1, function(x) ifelse(x[['TumorType']] == 'KPT-E', 1, 2))
group <- factor(tumor.meta.timing$Group)
design <- model.matrix(~group)
# subset
keep.cols <- row.names(tumor.meta.timing)
tpm.data.subset <- tpm.data[,keep.cols]
# fit limma model
dge <- DGEList(counts=tpm.data.subset)
dge <- calcNormFactors(dge)
logCPM <- cpm(dge, log=TRUE, prior.count=3)
fit <- lmFit(logCPM, design)
# fit <- treat(fit, lfc=log2(1.2), trend=TRUE)
fit <- eBayes(fit)
res <- topTable(fit, coef='group2', number=20000)
mapping = getBM(filters="ensembl_gene_id", attributes=c("ensembl_gene_id", "mgi_symbol"), values=rownames(res), mart=mouse)
res$ens = rownames(res)
res$gene = apply(res, 1, function(x) mapping[which(mapping$ensembl_gene_id == x[['ens']]), "mgi_symbol"])
res$gene <- as.character(res$gene)

write.table(res, "chuang_de_results.early_vs_late.tsv", sep='\t')

# DE TEST 2: Tnonmet vs Tmet
tumor.meta.mets = tumor.meta[-which((tumor.meta$TumorType == "KT") | (tumor.meta$TumorType == "KPT-E") | (tumor.meta$TumorType == "Normal Lung") | (tumor.meta$TumorType == 'TMet')),]
tumor.meta.mets$Group <- apply(tumor.meta.mets, 1, function(x) ifelse(x[['TumorType']] == 'TnonMet' | x[['TumorType']] == 'KPT-E', 1, 2))
group <- factor(tumor.meta.mets$Group)
design <- model.matrix(~group)
# subset
keep.cols <- row.names(tumor.meta.mets)
tpm.data.subset <- tpm.data[,keep.cols]
# fit limma model
dge <- DGEList(counts=tpm.data.subset)
# keep <- filterByExpr(dge, design)
# dge <- dge[keep,,keep.lib.sizes=FALSE]
dge <- calcNormFactors(dge)
logCPM <- cpm(dge, log=TRUE, prior.count=3)
fit <- lmFit(logCPM, design)
fit <- eBayes(fit)
res <- topTable(fit, coef='group2', number=20000)

mapping = getBM(filters="ensembl_gene_id", attributes=c("ensembl_gene_id", "mgi_symbol"), values=rownames(res), mart=mouse)
res$ens = rownames(res)
res$gene = apply(res, 1, function(x) mapping[which(mapping$ensembl_gene_id == x[['ens']]), "mgi_symbol"])
res$gene <- as.character(res$gene)

write.table(res, "chuang_de_results.tnonmet_vs_met.tsv", sep='\t')



