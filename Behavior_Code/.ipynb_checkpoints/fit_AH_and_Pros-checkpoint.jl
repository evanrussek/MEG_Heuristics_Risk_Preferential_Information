### em options
parallel = true # Run on multiple CPUs.
full = true    # Maintain full covariance matrix (vs a diagional one) a the group level
emtol = 1e-3    # stopping condition (relative change) for EM
first_run = false;
split_on_rt = false;

# are we at work computer or laptop?
work_computer = false;

# set up parallel if using
using Distributed
if (parallel & first_run)
    println("adding processes")
    # only run this once
    addprocs()
end

# add packages and other scripts for model fitting on every core
println("adding packages")
@everywhere using CSV
@everywhere using DataFrames
@everywhere using DataFramesMeta
@everywhere using SharedArrays
@everywhere using ForwardDiff
@everywhere using Optim
@everywhere using LinearAlgebra       # for tr, diagonal
@everywhere using StatsFuns           # logsumexp
@everywhere using SpecialFunctions    # for erf
@everywhere using Statistics          # for mean
@everywhere using Distributions
@everywhere using GLM
@everywhere using JLD
@everywhere using LineSearches

# for reading in data

#@everywhere project_folder = "/Users/evanrussek/meg_behavior";


if work_computer
    @everywhere project_folder = "C:\\Users\\erussek\\Documents\\GitHub\\meg_behavior";
    @everywhere em_dir =  "$project_folder/model_fitting/em";
else
    @everywhere project_folder = "/Users/erussek/Dropbox/meg_behavior";
    @everywhere em_dir =  "$project_folder/model_fitting/em";
end


# import other functions
@everywhere include("$project_folder/descriptive/descriptive.jl")
@everywhere include("$project_folder/model_fitting/likfuns2.jl")
@everywhere include("$project_folder/model_fitting/model_fitting_functions.jl")

@everywhere include("$em_dir/em.jl");
@everywhere include("$em_dir/common.jl");

# read in data and clean
data_raw = CSV.read("$project_folder/data/meg_behavior.csv", DataFrame);
data = clean_data(data_raw);

#### build models
to_save_folder = "$project_folder/paper_model_fit_results";

# EV, pros, additive, 
use_additive_list = [false, false, true];
warp_prob_list = [0, 2, 0];
warp_fun_list = ["none", "LOL", "none"];
warp_util_list = [0, 2, 0]; # how many utils to fit - use 1 or 2... 
n_accept_bias_list = [0, 0, 2];
n_delta_list = [0,1,0];
rt_cond = "none"

n_models = length(use_additive_list);

for i in 1:n_models

    this_model = Dict();
    (param_names, model_name) = build_names(use_additive_list[i],warp_prob_list[i],warp_util_list[i],learn_prob_list[i],sub_baseline_list[i],n_accept_bias_list[i], rt_cond, use_additive2_list[i];  fix_beta = false, warp_fun = warp_fun_list[i], n_delta = n_delta_list[i]);

    lik_fun = (params_in, sub_data) -> build_lik(params_in, sub_data, use_additive_list[i],warp_prob_list[i],warp_util_list[i],learn_prob_list[i],sub_baseline_list[i],n_accept_bias_list[i], rt_cond, use_additive2_list[i]; simulate = false,  warp_fun = warp_fun_list[i], n_delta = n_delta_list[i]);

    this_model["param_names"] = param_names;
    this_model["model_name"] = model_name;
    this_model["likfun"] = lik_fun;
    
    println(model_name)
    println(param_names)

    fit_model_em(this_model, data,to_save_folder, full = full, parallel = parallel, emtol = emtol, run_loo = true);
end

