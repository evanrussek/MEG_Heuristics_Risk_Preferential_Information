### em options
parallel = true # Run on multiple CPUs.
full = false    # Maintain full covariance matrix (vs a diagional one) a the group level
emtol = 1e-3    # stopping condition (relative change) for EM
first_run = true;

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

# for reading in data

#@everywhere project_folder = "/Users/evanrussek/meg_behavior";

println("defining folders")
if work_computer
    @everywhere project_folder = "C:\\Users\\erussek\\Documents\\GitHub\\meg_behavior";
    @everywhere em_dir =  "$project_folder/model_fitting/em";
else
    @everywhere project_folder = "/Users/erussek/Dropbox/meg_behavior";
    @everywhere em_dir = "$project_folder/model_fitting/em";
end


# import other functions
@everywhere include("$project_folder/descriptive/descriptive.jl")
@everywhere include("$project_folder/model_fitting/likfuns_sample2.jl")
@everywhere include("$project_folder/model_fitting/model_fitting_functions.jl")

@everywhere include("$em_dir/em.jl");
@everywhere include("$em_dir/common.jl");

# read in data and clean
data_raw = CSV.read("$project_folder/data/meg_behavior.csv", DataFrame);
data = clean_data(data_raw)


#### build models
to_save_folder = "$project_folder/paper_model_fit_results";

# fit the UWS model
param_names = ["choice_noise", "ns_p", "gain_power", "loss_power"];
model_name = "UWS_Sample";
lik_fun = uws_lik_mults;
this_model = Dict();
this_model["param_names"] = param_names;
this_model["model_name"] = model_name;
this_model["likfun"] = lik_fun;

# Fit the prob sample model
param_names = ["choice_noise", "ns_p", "gain_power", "loss_power"];
model_name = "Prob_Sample";
lik_fun = prob_lik_mults;
this_model = Dict();
this_model["param_names"] = param_names;
this_model["model_name"] = model_name;
this_model["likfun"] = lik_fun;
fit_model_em(this_model, data,to_save_folder, parallel = parallel, emtol = emtol, run_loo = true);