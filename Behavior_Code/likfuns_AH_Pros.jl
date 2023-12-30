
using SpecialFunctions    # for erf
using StatsFuns
using DataFrames
using DataFramesMeta
using Distributions

#####################################################
#####################################################
# HELPER FUNCTIONS TO COMPUTE LIKELIHOOD #########
#####################################################
#####################################################


####################################################
# Compute unnormalized choice log probabilities
###################################################

# multiplicitive evaluation
function evaluate_mult(p_o1, p_o2, o1_val, o2_val, safe_val, choice_beta, accept_bias)
    accept_val =  o1_val*p_o1 + o2_val*p_o2;
    reject_val = safe_val;
    return [choice_beta*reject_val, accept_bias + choice_beta*accept_val];
end

# additive evaluation
function evaluate_add(p_o1, o1_val, o2_val, safe_val, prob_beta, rew_beta, accept_bias)
    
    # beta_reward*trig
    if o1_val > o2_val 
        win_prob = p_o1 - (1 - p_o1);
    else
        win_prob = (1 - p_o1) - p_o1;
    end
    
    accept_val = accept_bias + prob_beta*win_prob + rew_beta*(mean([o1_val , o2_val])); ## does this make sense???? 
    reject_val = rew_beta*safe_val;
    
    return [reject_val, accept_val];
end

# additive evaluation
function evaluate_add2(p_o1, o1_val, o2_val, safe_val, prob_beta, rew_beta, accept_bias)
    
    # beta_reward*trig
    if o1_val > o2_val 
        win_prob = p_o1 - (1 - p_o1);
    else
        win_prob = (1 - p_o1) - p_o1;
    end
    
    if abs(o1_val) > abs(o2_val)
        rew_part = o1_val/2;
    else
        rew_part = o2_val/2;
    end
    
    accept_val = accept_bias + prob_beta*win_prob + rew_beta*rew_part; ## does this make sense???? 
    reject_val = rew_beta*safe_val;
    
    return [reject_val, accept_val];
end


########################################
# Probability Distortion
#######################################

# Prelec- 2 parameter version, set delta to 1 for 1 parameter version
function warp_prob_Prelec(p_o1, o1_val, o2_val, gamma,delta)
    p_o2 = 1 - p_o1;
    # warp the one which is the trigger and take the other to be 1 minus that
    
    if abs(o1_val) > abs(o2_val)
        if p_o1 < 1e-20
            p_o1_new = 0;
        else
            p_o1_new = exp(-delta*(-log(p_o1))^gamma)
        end
    else
        if p_o2 < 1e-20
            p_o2_new = 0;
        else
            p_o2_new = exp(-delta*(-log(p_o2))^gamma)
        end
        p_o1_new = 1 - p_o2_new;
    end
    return p_o1_new;
end


function warp_prob_Prelec2(p_o1, o1_val, o2_val, gamma,delta)
    p_o2 = 1 - p_o1;
    # warp the one which is the trigger and take the other to be 1 minus that
    
    p_o1_new = exp(-delta*(-log(p_o1))^gamma)
    p_o2_new = exp(-delta*(-log(p_o2))^gamma)


    return (p_o1_new, p_o2_new)
end

function warp_prob_power(p_o1, o1_val, o2_val, gamma,delta)
    p_o2 = 1 - p_o1;
    
    p_o1_new = p_o1^gamma;
    p_o2_new = p_o2^gamma;
    
    return (p_o1_new, p_o2_new)
    
end
    

################################## L
function warp_prob_lol(p_o1, o1_val, o2_val, gamma, delta)
    # "Gamma conrols curvature, delta controls elevation
    
    p_o2 = 1 - p_o1;
    
    p_o1_new = delta*(p_o1^gamma) / (delta*(p_o1^gamma) + (1-p_o1)^gamma);
    p_o2_new = delta*(p_o2^gamma) / (delta*(p_o2^gamma) + (1-p_o2)^gamma);
    
    return (p_o1_new, p_o2_new)
end
        


#############################################
# Utility Function
#############################################

# Take power as per Kahneman and Tversky
# Optional argument to subtract the baseline which was added to each option
function util_power(o1_val, o2_val, safe_val, gain_power, loss_power; sub_baseline = true)
    if safe_val > 0
        if sub_baseline
            min_val = minimum([o1_val, o2_val, safe_val]);
            o1_val = o1_val - min_val;
            safe_val = safe_val - min_val;
            o2_val = o2_val - min_val;
        end
        return (o1_val^gain_power, o2_val^gain_power, safe_val^gain_power);
    else
        if sub_baseline
            max_val = maximum([o1_val, o2_val, safe_val]);
            o1_val = o1_val - max_val;
            safe_val = safe_val - max_val;
            o2_val = o2_val - max_val;
        end
        return (-(abs(o1_val)^loss_power), -(abs(o2_val)^loss_power), -(abs(safe_val)^loss_power));
    end
end

#############################################################################################
#############################################################################################
# LOG LIKELIHOOD FUNCTION 
#############################################################################################

# compute log likelihood of sequence of choices -- or compute choice probabilities for sequence of choicse
function MEG_val_lik(params, sub_data; warp_prob = 0, sub_baseline = true, use_additive = false, use_additive2 = false, rt_cond = "none", simulate =  false, transform_params = true, add_choice_noise = true, warp_fun = "LOL")
            
    if transform_params
        lr = 0.5 + 0.5 * erf(params[2] / sqrt(2)) # learning rate... transform it...
        gamma = exp(params[3]/50); # # trying to change this...
        delta_gain = exp(params[4]/50); # for prelect warp...
        delta_loss = exp(params[11]/50);
        gain_power = exp(params[5]/50);
        loss_power = exp(params[6]/50);
    else
        lr = params[2];
        gamma = params[3];
        delta_gain = params[4]; # for prelect warp...
        gain_power = params[5];
        loss_power = params[6];
    end
    
    if use_additive | use_additive2
        choice_beta = params[1];
        prob_beta = params[1]; # also prob beta, but for the additive model
        rew_beta = params[7]/10; # 
    else
        choice_beta = params[1];
    end
    accept_bias_gain = params[8];
    accept_bias_loss = params[9];
    
    # 
    if add_choice_noise
        choice_noise = 0.5 + 0.5 * erf(params[10] / sqrt(2));
    end
    
    
    n_trials = size(sub_data,1);
    choice_o1_prob = typeof(choice_beta)[.2, .4, .6, .8];

    # transform these to starting points
    point_scale = 1/15;
    
    # simulate or likelihood
    if simulate
        accept_prob = zeros(Float64,n_trials); # if we simulate, return accept_prob prob for each trial... 
    else
        lik = 0;
    end

    for trial_idx in 1:n_trials
        
       # println(trial_idx)

        choice_idx = Int.(sub_data[trial_idx, :choice_number]);
        p_o1 = choice_o1_prob[choice_idx];
        o1_val = sub_data[trial_idx, :o1_val]*point_scale;
        o2_val = sub_data[trial_idx, :o2_val]*point_scale;
        safe_val = sub_data[trial_idx, :safe_val]*point_scale;
        accept = Int.(sub_data[trial_idx, :accept]);
        outcome_reached = Int.(sub_data[trial_idx, :outcome_reached]);
        #high_rt = sub_data[trial_idx, :high_rt];
        high_rt = 1
        
        
        if safe_val > 0
            accept_bias = accept_bias_gain;
            delta = delta_gain;
        else
            accept_bias = accept_bias_loss;
            delta = delta_loss;
        end
                
        # warp probs
        if warp_prob > 0
            #p_o1_warp = warp_prob_Prelec(p_o1, o1_val, o2_val, gamma, delta)
            #(p_o1_warp, p_o2_warp) = warp_prob_Prelec2(p_o1, o1_val, o2_val, gamma, delta)
            
            if warp_fun == "Prelec"
                (p_o1_warp, p_o2_warp) = warp_prob_Prelec2(p_o1, o1_val, o2_val, gamma, delta)
            elseif warp_fun == "LOL"
                (p_o1_warp, p_o2_warp) = warp_prob_lol(p_o1, o1_val, o2_val, gamma, delta)
            elseif warp_fun == "Power"
                (p_o1_warp, p_o2_warp) = warp_prob_power(p_o1, o1_val, o2_val, gamma,delta)
            end
        else
            p_o1_warp = p_o1;
            p_o2_warp = 1 - p_o1;
        end
        
        # get outcome utility
        (o1_u, o2_u, safe_u) = util_power(o1_val, o2_val, safe_val, gain_power, loss_power; sub_baseline = sub_baseline);
        
        if use_additive
            option_vals = evaluate_add(p_o1, o1_val, o2_val, safe_val, prob_beta, rew_beta, accept_bias);
        elseif use_additive2
            option_vals = evaluate_add2(p_o1, o1_val, o2_val, safe_val, prob_beta, rew_beta, accept_bias);
        else
            option_vals = evaluate_mult(p_o1_warp, p_o2_warp, o1_u, o2_u, safe_u, choice_beta, accept_bias);
        end
        
        if simulate
            #println(option_vals[2]);
            accept_prob[trial_idx] = exp(option_vals[2])/sum(exp.(option_vals));
        else
            if (rt_cond == "none") | ((rt_cond == "high") &  (high_rt == 1)) | ((rt_cond == "low") & (high_rt == 0))
               # println("here")
                if add_choice_noise
                    prob_accept = (1 - choice_noise)*logistic(option_vals[2] - option_vals[1]) + .5*choice_noise;
                    if accept == 1
                        prob_choice = prob_accept;
                    else
                        prob_choice = 1 - prob_accept;
                    end
                    lik += log(prob_choice);
                else
                    lik += option_vals[accept + 1] - logsumexp(option_vals);
                end
            end                
        end
        
        # update probability rep
        if (accept == 1)
            choice_o1_prob[choice_idx] = (1 - lr)*choice_o1_prob[choice_idx] + lr*(outcome_reached == 1);
        end
    end
        
    if simulate
        return accept_prob
    else
        return -1*lik;
    end

end


# want to define a likelihood function and name for each model-type...
#### LOOP through these 6 things... 
######################
# build the param_names
function build_names(use_additive,warp_prob,warp_util,learn_prob,sub_baseline,n_accept_bias, rt_cond, use_additive2; add_choice_noise = true, fix_beta = false, warp_fun = "", n_delta = 1)
    
    param_names = [];
    title_str = "";
    
    if use_additive
        push!(param_names,"beta_prob");
        title_str = title_str*"Add";
    elseif use_additive2
        push!(param_names,"beta_prob");
        title_str = title_str*"Add2";
    else
        if !fix_beta
            push!(param_names, "choice_beta");
            title_str = title_str*"Mult";
        else
            title_str = title_str*"Mult_fb";
        end

    end
    
    if learn_prob
        push!(param_names, "lr");
        title_str = title_str*"_Learn";
    end
    if warp_prob == 2
        push!(param_names,"gamma"); push!(param_names, "delta");
        title_str = title_str*"_WarpProb"*warp_fun;
    elseif warp_prob == 1
        push!(param_names,"gamma");
        title_str = title_str*"_WarpProb"*warp_fun*"1";
    end
    
    if warp_util == 2
        push!(param_names,"gain_power"); push!(param_names, "loss_power");
        title_str = title_str*"_WarpUtil";
    elseif warp_util == 1
        push!(param_names,"u_power");
        title_str = title_str*"_WarpUtil1";
    end
    
    if use_additive | use_additive2
        push!(param_names, "beta_rew");
    end
    
    if n_accept_bias == 1
        push!(param_names,"accept_bias")
        title_str = title_str*"_AccBias";
    elseif n_accept_bias == 2
        push!(param_names,"accept_bias_gain")
        push!(param_names,"accept_bias_loss")
        title_str = title_str*"_2AccBias";
    end
    
    if sub_baseline
        title_str = title_str*"_SubBase";
    end
    
    if rt_cond != "none"
        title_str = title_str*"_$rt_cond";
    end
    
    if add_choice_noise
        title_str = title_str*"_pchoice_noise";
        push!(param_names, "choice_noise")
    end
    
    if n_delta == 2
        title_str = title_str*"_2delt";
        push!(param_names, "delta_loss")
    end

    
    return (param_names, title_str)
end

function build_lik(params_in, sub_data, use_additive,warp_prob,warp_util,learn_prob,sub_baseline,n_accept_bias, rt_cond, use_additive2; simulate = false, transform_params = true, add_choice_noise = false, fix_beta = false, warp_fun = "", n_delta = 1);

    p_idx = 1;
    
    if fix_beta
        beta = 1
    else
        beta = params_in[p_idx];
        p_idx = p_idx+1;
    end
    
    if learn_prob
        lr = params_in[p_idx];
        p_idx = p_idx+1;
    else
        lr = -1000;
    end
    
    if warp_prob == 2
        gamma = params_in[p_idx];
        p_idx = p_idx+1;
        delta_gain = params_in[p_idx];
        p_idx = p_idx+1;
    elseif warp_prob == 1
        gamma = params_in[p_idx];
        p_idx = p_idx+1;
        delta = 0; #??
        delta_gain = delta;
    else
        gamma = 0; delta = 0;
        delta_gain = delta;
    end
    
    if warp_util == 2
        gain_power = params_in[p_idx];
        p_idx = p_idx+1;
        loss_power = params_in[p_idx];
        p_idx = p_idx+1;
    elseif warp_util == 1
        gain_power = params_in[p_idx];
        loss_power = params_in[p_idx];
        p_idx = p_idx+1;
    else
        gain_power = 1; loss_power = 1;
    end
    
    if use_additive | use_additive2
        rew_beta = params_in[p_idx];
        p_idx = p_idx+1;
    else
        rew_beta = 0;
    end
    
    if n_accept_bias == 1 # should this split by gain / loss?
        accept_bias_gain = params_in[p_idx];
        accept_bias_loss = params_in[p_idx];
        p_idx = p_idx+1;
    elseif n_accept_bias == 2
        accept_bias_gain = params_in[p_idx];
        p_idx = p_idx+1;
        accept_bias_loss = params_in[p_idx];
        p_idx = p_idx+1;
    else
        accept_bias_gain = 0;
        accept_bias_loss = 0;
    end
    
    if add_choice_noise
        choice_noise = params_in[p_idx]; p_idx += 1;
    else
        choice_noise = -1000;
    end
    
    if n_delta == 2
        delta_loss = params_in[p_idx];
    else
        delta_loss = delta_gain;
    end
    
    
    params = [beta; lr; gamma; delta_gain; gain_power; loss_power; rew_beta; accept_bias_gain; accept_bias_loss; choice_noise; delta_loss];
    
    return MEG_val_lik(params, sub_data; warp_prob = warp_prob, sub_baseline = sub_baseline, use_additive = use_additive, use_additive2 = use_additive2, rt_cond = rt_cond, simulate =  simulate, transform_params = transform_params, add_choice_noise = add_choice_noise, warp_fun = warp_fun)
end


#### functon to simulate group recovery dataset
function recover_data_set(this_model_res,data)
    
    sim_data = DataFrame();

    for s_idx in unique(data[!,:sub])
        s_data = @where(data,:sub .== s_idx);
        s_params = this_model_res["x"][s_idx,:];
        s_sim_accept = this_model_res["simfun"](s_params,s_data);

        s_sim_data = s_data;
        s_sim_data[!,:accept] = s_sim_accept;

        sim_data = [sim_data; s_sim_data];
    end
    
    return sim_data
end



