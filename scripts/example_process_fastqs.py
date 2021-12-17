import sys
import os

import Cassiopeia.ProcessingPipeline.process as process
import Cassiopeia.ProcessingPipeline.process.sequencing as sequencing
import pandas as pd 

import Cassiopeia as sclt
import numpy as np

#samples = ["L6", "L7"]
samples = ["L10"]

for sample in samples:

    home_dir = sample + "_TS/"
    
    print("PROCESSING " + sample + " out of " + home_dir)

    print("Collapsing")
    process.collapseUMIs(".", home_dir + "possorted_genome_bam.bam", force_sort=True)
    process.collapseBam2DF(home_dir + "possorted_genome_bam_sorted.collapsed.bam", home_dir + "possorted_genome_bam_sorted.collapsed.txt")
    print(home_dir + sample + "_possorted_genome_bam_sorted.collapsed.bam")

    print("Picking Sequences")
    process.pickSeq(home_dir + "possorted_genome_bam_sorted.collapsed.txt", home_dir + sample + "_possorted_genome_bam_sorted.picked.txt", home_dir, cell_umi_thresh=2, avg_reads_per_UMI_thresh=2, save_output=True)

    # switch to line below if you are using this pipeline for the LENTI-BC preprocessing
    process.align_sequences("../TS_ref.fa", home_dir + sample + "_possorted_genome_bam_sorted.picked.txt", home_dir + sample + "_sw_aligned.sam")
    #process.align_sequences("../lentiBC.fa", home_dir + sample + "_possorted_genome_bam_sorted.picked.txt", home_dir + sample + "_sw_aligned.sam")

    print("Calling indels")
    process.call_indels(home_dir + sample + "_sw_aligned.sam", "../TS_ref.fa", home_dir + sample + "_umi_table.sam")

    print("Error Correcting UMIs")
    process.errorCorrectUMIs(home_dir + sample + "_umi_table.bam", sample, "ec_log.txt")

    print("Filtering Molecule Tables")
    process.filter_molecule_table(home_dir + sample + "_umi_table_sorted_ec.moleculeTable.txt", sample + ".moleculeTable.filtered.txt", home_dir, doublet_threshold=0.1)
    #process.filter_molecule_table(home_dir + sample + "_umi_table_sorted_ec.moleculeTable.txt", sample + ".moleculeTable.filtered.txt", home_dir, detect_intra_doublets=False)

    #print("Calling Lineage Groups")
    #process.call_lineage_groups(home_dir + sample + ".moleculeTable.filtered.txt",sample + ".alleleTable.txt", home_dir, cell_umi_filter=10, verbose=True, detect_doublets_inter=True, min_intbc_thresh = 0.2, kinship_thresh=0.25) 

