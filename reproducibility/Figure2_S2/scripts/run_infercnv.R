#!/usr/bin/Rscript

args <- commandArgs(trailingOnly = TRUE)
counts_path <- args[1]          # TSV containing gene expression counts
annotation_path <- args[2]      # TSV containing sample annotations
ordering_path <- args[3]        # TSV containing gene ordering (chromosome, start & end positions)
controls <- readLines(args[4])  # TXT file containing which samples should be used as controls
out_dir <- args[5]              # Output directory
n_threads <- strtoi(args[6])    # Number of threads to use
resume <- args[7]               # Whether to overwrite or resume using existing files

library('infercnv')

infercnv_obj = infercnv::CreateInfercnvObject(
    raw_counts_matrix=counts_path,
    annotations_file=annotation_path,
    gene_order_file=ordering_path,
    ref_group_names=controls,
    delim='\t',
)
infercnv_obj = infercnv::run(
    infercnv_obj,
    min_cells_per_gene=1,
    analysis_mode='subclusters',
    hclust_method='ward.D2',
    tumor_subcluster_partition_method='random_trees',
    max_centered_threshold=NA,
    cutoff=0.1,
    out_dir=out_dir,
    cluster_by_groups=TRUE,
    denoise=TRUE,
    HMM=TRUE,
    BayesMaxPNormal=0.2,
    num_threads=n_threads,
    resume_mode=(resume == "T"),
    no_plot=TRUE
)
