Launching KPTracer VISION Reports
================

# Introduction

This vignette serves as a tutorial for downloaiding and launching interactive VISION reports for the KP-Tracer dataset. More specifically, this dataset consists of paired scRNA-seq and inferred single-cell phylogenies for tumors from the KP mouse model of non-small-cell lung cancer (as described in [Yang et al, *bioRxiv* 2021](https://www.biorxiv.org/content/10.1101/2021.10.12.464111v1)). Below, we demonstrate how to download pre-run reports, as well as launch your own.

## Prerequisites

Before getting started, you'll have to make sure you have a working version of `VISION` installed. You can install `VISION` directly from [Github](https://github.com/YosefLab/VISION). You can do this in two ways -- either by first cloning the repository and installing from source, or using `devtools` as described below.

``` r{eval=false}
require(devtools)
install_github("YosefLab/VISION")
```

Also, if you'd like to run your own PhyloVision analyses using [the script](https://github.com/mattjones315/KPTracer-release/blob/main/scripts/run_phylovision.R) in our Github repository, you'll have to make sure you have the following packages installed:

``` r{eval=false}
install.packages('reticulate')
install.packages('ape')
install.packages('Matrix')
```

## Downloading existing reports

Downloading an existing report is easy as we've hosted these on Zenodo, and R can download data from an existing URL. You can view the reports available on our VISION reports repository [here](https://zenodo.org/record/5888896#.YfmuQPXMLYU). There are two types of reports:

1.  Basic scRNA-seq VISION reports providing for the sgNT dataset and full dataset; and
2.  PhyloVision reports for individual clones which allow you to probe the phylogenies and scRNA-seq for a specific tumor.

After you pick a report to view (for example 3513\_NT\_T2 PhyloVision), you can download the report and view the report as so:

``` r{eval=false}
url_location = 'https://zenodo.org/record/5888896/files/3513_NT_T2_phylovision.rds?download=1'
vision_report = readRDS(url(url_location))

viewResults(vision_report, port=8119)
```

The `viewResults` function will launch a report locally at port 8119. If you are trying to launch a report from a remote server, try using the following call to `viewResults`:

``` r{eval=false}
viewResults(vision_report, host='0.00.0', port=8119, browser=F)
```

This will launch a report on your remote server, accessible at `https://<servername>:8119/Results.html`.

## Performing new VISION analyses

If there's a particular clone you'd like to evaluate, you can download raw data from our Zenodo repository (currently under embargoed access) [here](https://zenodo.org/record/5847462#.Yfm4nPXMLYV).

Once you've downloaded this data, you can use [the script](https://github.com/mattjones315/KPTracer-release/blob/main/scripts/run_phylovision.R) available on our reproducibility repository for doing this analysis. To use this script, you must provide four arguments:

-   A tumor name that exists in the `KPTracer-Data/trees` directory
-   The path to the `KPTracer-Data` directory, downloaded from Zenodo.
-   The path to the directory holding the signatures you'd like to use. The script will use all the signatures stored in this folder, so it is advisable to create a new directory specifically for this analysis. Note: Make sure these are signatures with gene names from mm10! We provide two example signature files for mm10 in our [reproducibility repository](https://github.com/mattjones315/KPTracer-release/tree/main/data).
-   The file path at which you'd like to save the analyzed object.

For example, the following can be used to launch the script from command line:

    ~$ Rscript run_phylovision.R 3513_NT_T2 /path/to/KPTracer-Data/ /path/to/kptracer-signatures/ test_vision.rds

## Session Info

``` r
sessionInfo()
```

    ## R version 3.6.2 (2019-12-12)
    ## Platform: x86_64-pc-linux-gnu (64-bit)
    ## Running under: Ubuntu 16.04.6 LTS
    ## 
    ## Matrix products: default
    ## BLAS:   /usr/lib/atlas-base/atlas/libblas.so.3.0
    ## LAPACK: /usr/lib/atlas-base/atlas/liblapack.so.3.0
    ## 
    ## locale:
    ##  [1] LC_CTYPE=en_US.UTF-8       LC_NUMERIC=C              
    ##  [3] LC_TIME=en_US.UTF-8        LC_COLLATE=en_US.UTF-8    
    ##  [5] LC_MONETARY=en_US.UTF-8    LC_MESSAGES=en_US.UTF-8   
    ##  [7] LC_PAPER=en_US.UTF-8       LC_NAME=C                 
    ##  [9] LC_ADDRESS=C               LC_TELEPHONE=C            
    ## [11] LC_MEASUREMENT=en_US.UTF-8 LC_IDENTIFICATION=C       
    ## 
    ## attached base packages:
    ## [1] stats     graphics  grDevices utils     datasets  methods   base     
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] BiocManager_1.30.10 compiler_3.6.2      BiocStyle_2.22.0   
    ##  [4] magrittr_1.5        tools_3.6.2         htmltools_0.4.0    
    ##  [7] yaml_2.2.1          Rcpp_1.0.5          stringi_1.4.6      
    ## [10] rmarkdown_2.11      knitr_1.37          stringr_1.4.0      
    ## [13] xfun_0.29           digest_0.6.20       rlang_0.4.6        
    ## [16] evaluate_0.14
