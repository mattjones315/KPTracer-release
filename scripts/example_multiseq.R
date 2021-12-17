require(deMULTIplex)

bar.ref = read.table("/home/mattjones/projects/kptc/MULTI/LMOlist.csv", sep=',', header=F, stringsAsFactors=F)[,1]
cell.id.vec = read.table("/home/dian/10x_data/20190510_TRCR4_novaseq/L8_scRNAseq_result_all/outs/filtered_gene_bc_matrices/mm10/barcodes.tsv", stringsAsFactors = F)[,1]

# if your cells come from an RNA-seq library, make sure to cleave off the gem group
cell.id.vec = unlist(lapply(cell.id.vec, function(x) unlist(strsplit(x, "-", fixed=T))[[1]]))

R1 = '/home/dian/10x_data/20190403_TRCR3_novaseq/raw/L8_Multi_merged_R1_001.fastq.gz'
R2 = '/home/dian/10x_data/20190403_TRCR3_novaseq/raw/L8_Multi_merged_R2_001.fastq.gz'

cell.pos = c(1,16)
umi.pos = c(17,26)
tag.pos = c(1,8)

whitelist = read.table("../lane_multi_whitelist.txt", sep='\t', header=T, stringsAsFactors=F)
good.bars = whitelist[which(whitelist$Sample == 'L8'),'BC']
print(good.bars)

final.calls = analyze_multi_sample(bar.ref, cell.id.vec, R1, R2, exp.name = "L8_multi")
print(table(final.calls))