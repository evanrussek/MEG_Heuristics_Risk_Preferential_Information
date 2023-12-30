using CSV
using DataFrames
using DataFramesMeta
using CategoricalArrays
using Gadfly
using Statistics
using Distributions
using SpecialFunctions
using StatsFuns
using Optim
using ForwardDiff
using Cairo
using Fontconfig

cd("/Users/evanrussek/foraging/")

# basic real data clearning, etc, functions...
include("/Users/evanrussek/lockin_data/lockin_analysis/forage_data_funs.jl")
include("sim_lag_functions.jl")
include("sim_learn_funcs.jl")


# read in the data...
data = CSV.read("/Users/evanrussek/forage_jsp/analysis/data/run5_data.csv");

# check the travel keys are correctly labeled...
travel_keys = unique(data.travel_key)
travel_key_easy = travel_keys[1]
travel_key_hard = travel_keys[2]
travel_keys_he = [travel_key_hard travel_key_easy];
cdata, n_subj = clean_group_data(data,travel_keys_he);

show(first(cdata,2), allcols = true)



## data cols needed
# sub, trial_num, lag, choice, phase, start_reward...

# do you need to scale the lags in some way????????

s_data = @where(cdata, :s_num .== 1);
s_data[!,:sub] = s_data[!,:s_num];
s_data[!,:lag_scale] = s_data[!,:lag];
s_data[!,:lag] = s_data[!,:lag_scale] ./ 100;

s_data[ismissing.(s_data[!,:reward_obs]),:reward_obs] .= 0;
s_data[ismissing.(s_data[!,:exit]),:exit] .= 1;
s_data[!,:choice] = s_data[!,:exit] .+ 1;
s_data[!,:choice] = convert(Array{Int,1}, s_data[!,:choice]);




# should we filter out extreme responses????? - no for now,
# because it would probably remove

# harvest is 1, exit is 2...

plot(s_data, x = :press_num,y = :choice, xgroup = :travel_key_cond,ygroup = :start_reward,
    Geom.subplot_grid(Geom.line()))
plot(s_data, x = :press_num,y = :lag, xgroup = :travel_key_cond,ygroup = :start_reward,
        Geom.subplot_grid(Geom.line()))

## pass high / low lag filter into lik fun... # also don't model first press lag...
#

plot(s_data, y = :exit,  Geom.line())

# this function orders the reward levels wrong...
make_exit_plot(s_data)
make_lag_plot(s_data)

## fit this subject...

param_names = ["travel_cost_easy", "travel_cost_hard", "lr_R_hat_pre", "harvest_cost"];

start_p = generate_start_vals((x) -> forage_learn_lik3(x,s_data,"both"), length(param_names))


function forage_learn_lik3(param_vals, data, which_data)

    #println(param_vals)

    # get rid of lookup
    #params = params .+ .0001;
    #lag_beta = param_vals[1];
    travel_cost_easy = param_vals[1]*10;
    travel_cost_hard = param_vals[2]*10;
    lr_R_hat_pre = param_vals[3];
    #println(param_vals[3])
    #println(string("lr_R_hat_pre: ", lr_R_hat_pre))

    harvest_cost = param_vals[4]*10;
    choice_beta = 1.;#param_vals[5];
    lag_beta = 1.;
    #println(string("lr_R_hat_pre: ", lr_R_hat_pre))

    lr_R_hat = (.5 + .5*erf(lr_R_hat_pre/5))/10; #0.5 + 0.5 * erf(lr_R_hat_pre / sqrt(2));
    # get unique trials...
    travel_costs = Dict();
    travel_costs["HARD"] = travel_cost_hard;
    travel_costs["EASY"] = travel_cost_easy;

    ###### for picking a lag -- what numbers to evaluate......
    travel_success_prob = .8;
    harvest_success_prob = .5;
    decay = .98;

    lag_ll = 0;
    choice_ll = 0;

    trial_list = unique(data[!,:trial_num])

    for trial_idx in trial_list

        # is this the problem???

        trial_data = data[data[!,:trial_num] .== trial_idx,:];

        lag = trial_data[!,:lag];
        choice = trial_data[!,:choice];
        phase = trial_data[!,:phase];
        reward_obs = trial_data[!,:reward_obs];

        trial_travel_key = trial_data[1,:travel_key_cond]
        travel_cost = travel_costs[trial_travel_key]
        trial_start_reward = trial_data[1,:start_reward];

        last_reward_obs = copy(trial_start_reward);

        R_hat = convert(typeof(lag_beta),5.); # maybe you need to fit this...

        # go through each measure in trial_data
        n_presses = size(lag,1);
        for press_idx in 1:n_presses

            if phase[press_idx] == "HARVEST"

                if val_fail(R_hat,harvest_cost, lag_beta)
                    return 1e9
                end


                current_optimal_lag = sqrt(harvest_cost / R_hat)
                E_next_reward_harvest = last_reward_obs*decay*harvest_success_prob - (harvest_cost / current_optimal_lag);
                E_opportunity_cost_harvest = R_hat*current_optimal_lag;

                theta = [choice_beta*E_next_reward_harvest, choice_beta*E_opportunity_cost_harvest];

                if !first_round_harvest
                    choice_ll = choice_ll + theta[choice[press_idx]] - logsumexp(theta);
                end

                if (choice[press_idx] == 1) # choose harvest

                    this_lag = lag[press_idx];

                    # don't contribute first press to likelihood
                    if !first_round_harvest
                        lag_ll = lag_ll + log_lik_lag(R_hat,harvest_cost,lag_beta,this_lag);
                    else
                        first_round_harvest = false;
                    end

                    this_cost = harvest_cost / this_lag;
                    this_reward = reward_obs[press_idx];

                    if this_reward > .0000001
                        last_reward_obs = this_reward;
                    end

                else # choose travel
                    if val_fail(R_hat,travel_cost, lag_beta)
                        return 1e9
                    end

                    this_lag = lag[press_idx];
                    # don't add first press to lag...
                    # lag_ll = lag_ll + log_lik_lag(R_hat,travel_cost,lag_beta,this_lag);

                    this_cost = travel_cost / this_lag;
                    this_reward = reward_obs[press_idx];

                    # reset the last_reward_obs for the next harvest
                    last_reward_obs = copy(trial_start_reward);
                    first_round_harvest = true;
                end
            else # travel session - just pick a lag...

                if val_fail(R_hat,travel_cost, lag_beta)
                    return 1e9
                end
                this_lag = lag[press_idx];
                lag_ll = lag_ll + log_lik_lag(R_hat,travel_cost,lag_beta,this_lag);
                this_cost = travel_cost / this_lag;
                this_reward = reward_obs[press_idx];
            end

            # update R_hat...
            this_move_rr = (this_reward - this_cost) / this_lag;
            R_hat = (1 - (1 - lr_R_hat)^this_lag)*this_move_rr + ((1 - lr_R_hat)^this_lag)*R_hat;

        end
    end

    if which_data == "choice"
        return -1*choice_ll
    elseif which_data == "lag"
        return -1*lag_ll
    else
        return -1*(lag_ll + choice_ll);
    end

end

#


# can we plot the RTs?

!true
