using MatrixMarket;
using GLM;
using DataFrames;
using CSV;
using Statistics;
using Distributions;
using Tables;
using StatsBase;


### Many functions below taken from https://github.com/ThomsonMatt/CompBioClass#loading-mtx-files-in-julia
function read_csc(pathM::String)
    X = MatrixMarket.mmread(string(path10x, "matrix.mtx"));
    Float64.(X)
end

function read_barcodes(tsvPath::String)
    f=open(tsvPath)
    lines=readlines(f)
    a=String[]
    for l in lines
        push!(a,l)
    end
    close(f)
    return a
end

function read_genes(tsvPath::String)
    f=open(tsvPath)
    lines=readlines(f)
    a=String[] #Array{}
    for l in lines
        push!(a,split(l,"\t")[2])
    end
    close(f)
    return a
end

function sort_array_by_list(arr, _list)

    order = [];
    for bc in _list
        i = findall(x->x == bc, arr)[1]
        push!(order, i)
    end
    order
end

function calc_log2fc(up, dwn)

    fc = (0.01 + mean(up)) / (0.01 + mean(dwn));
    log2(fc)

end

function find_log2fc_ci(up, dwn, B=1000)

    fcs = [];
    fc_est = calc_log2fc(up, dwn);

    for i in 1:B
        up_p = sample(up, length(up), replace=true);
        down_p = sample(dwn, length(dwn), replace=true);
        push!(fcs, calc_log2fc(up_p, down_p));
    end

    fc_est, percentile(fcs, 2.5), percentile(fcs,97.5)
end 

count_nnz(x) = count(i -> (i>0), x)
genotype = "NT"

println("reading in data")

path10x = "/data/yosef2/users/mattjones/projects/kptc/RNA/ALL/mm10/";
meta_fp = "/data/yosef2/users/mattjones/projects/kptc/L6-37_ALL_META_052720.txt"
tumor2modelfp = "/data/yosef2/users/mattjones/projects/kptc/trees/tumor_model.txt"

expr = Array(read_csc(string(path10x, "matrix.mtx")));
barcodes = read_barcodes(string(path10x, "barcodes.tsv"));
genes = read_genes(string(path10x, "genes.tsv"));
println(string("read in a gene expression matrix with ", length(barcodes), " cells and ", length(genes), " genes"))

println("filtering out apoptotic cells")
mito_genes = [name for name in genes if startswith(name, "MT-")];
mito_inds = findall(i -> i in mito_genes, genes);
mito_prop = mapslices(i -> sum(i[mito_inds,:]) / sum(i), expr, dims=1);
keep_cells = [i <= 0.2 for i in mito_prop][1,:];

println(string("filtering out ", size(expr)[2] - sum(keep_cells), " cells"))
expr = expr[:, keep_cells];
barcodes = barcodes[keep_cells];

# define UMI-normalization factor
norm_factor = median(mapslices(i -> sum(i), expr, dims=1));

println("reading in and filtering by meta data entries")
meta_data = CSV.read(meta_fp, delim='\t');

println("reading in tumor to model mapping")
tumor2model = CSV.read(tumor2modelfp, delim='\t')

# drop out rows that have missing values
# meta_data = meta_data[completecases(meta_data),:];

tumors = unique(tumor2model.Tumor)

tumors = ["3434_NT_T3"];
for tumor in tumors

    if !occursin(genotype, tumor)
        continue;
    end

    fp = string("/data/yosef2/users/mattjones/projects/kptc/trees/", tumor, "/mean_fitness.", tumor, ".txt");
    if !isfile(fp)
        continue;
    end 

    println(string("Analyzing tumor ", tumor))
    fitness = CSV.read(fp, delim='\t');

    keep_cells = intersect(fitness.Column1, barcodes);
    fitness = filter(row->row.Column1 in keep_cells, fitness);

    _order = sort_array_by_list(barcodes, fitness.Column1);
    expr_tumor = expr[:, _order];

    # calculate size factors
    size_factors = [i for i in mapslices(i -> sum([x > 0 for x in i]) / length(genes), expr_tumor, dims=1)[1, :]];

    # println("normalizing library counts")
    expr_tumor_norm = mapslices(i -> (i * norm_factor) / sum(i), expr_tumor, dims=1);

    println("filtering out lowly expressed genes")
    nnz = mapslices(count_nnz, expr_tumor, dims=2);
	  threshold = 10;
    keep_ii = map(i->i[1], findall(x -> x>threshold, nnz));

    expr_tumor_filt = expr_tumor[keep_ii,:];
    expr_tumor_norm_filt = expr_tumor_norm[keep_ii,:];

    _genes = genes[keep_ii];

    # center dynamic met DynamicMetScore
    #mu = mean(tumor_meta.mean_fitness);
    #tumor_meta.mean_fitness = [(i - mu) for i in tumor_meta.mean_fitness];

    refactor_met_score(x) = ifelse(x >= 0, 1, 0);
    # y = [refactor_met_score(i) for i in tumor_meta.mean_fitness];
    y = [i for i in fitness.mean_fitness];
    sz = [i for i in size_factors];
    X = transpose(expr_tumor_filt);
    Xn = transpose(expr_tumor_norm_filt);
    # X = transpose(expr_tumor_norm_filt);

    up_cells = Xn[y .>= percentile(y, 90),:];
    dwn_cells = Xn[y .< percentile(y, 10),:];

    betas = [];
    pvalues = [];
    genes_tested = [];
    test_type = [];
    fold_changes = [];
    fold_changes_95up = [];
    fold_changes_95lo = [];

    println(string("continuing with model of ", size(X)[1], " cells and ", size(X)[2], " genes for tumor ", tumor, "."))

    for i in 1:size(X)[2]

        # df = DataFrame(x = (X[:, i] .+ 1), y = string.(y),  sz = sz);
                        
        fc, lower95, upper95 = find_log2fc_ci(up_cells[:,i], dwn_cells[:,i]);

        push!(genes_tested, _genes[i]);
        push!(fold_changes, fc)
        push!(fold_changes_95up, upper95)
        push!(fold_changes_95lo, lower95)

    end 
    
    res_df = DataFrame(genes = genes_tested, log2fc = fold_changes, CI_up = fold_changes_95up, CI_down = fold_changes_95lo)
    CSV.write(open(string("/data/yosef2/users/mattjones/projects/kptc/trees/", tumor, "/fitness_log2fc.", tumor, ".txt"), "w"), res_df, delim = "\t");

end
