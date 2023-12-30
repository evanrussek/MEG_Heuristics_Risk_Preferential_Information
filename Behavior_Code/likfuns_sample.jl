### want to use this to do the UWS and the Probability Sampling...


include("$project_folder/model_fitting/likfuns2.jl")
using Distributions
using StatsFuns

function binomial_pdf(n,N, p)
    #print(n)
    return binomial(N, n)*(p^n)*(1-p)^(N-n)
end

function ordered_probit_pmf(k, n_in, c_in, beta_in)
    
   # println(beta)
    
    # warp this by the scale....
    c = beta_in .* c_in;
    n = beta_in .* n_in;
    
    ϕ = (x) -> cdf(Normal(), x);
    K = length(c) + 1;

    if k == 1
        return 1 - ϕ(n - c[1]);
    elseif 1 < k < K
        return ϕ(n - c[k-1]) - ϕ(n - c[k])
    elseif k == K
        return ϕ(n - c[K-1]);
    else
        error("k is out of domain")
    end 
end

function make_n_sample_pmf(max_n_samples, ns_p, scale_p)
    #scale_p = 3;
    
    # define evenly spaced cuts for probit distr
    c = range(1.,stop = max_n_samples,length = max_n_samples + 1)[2:end-1];
    
    # get probability mass for each discrete number of samples we might take
    return  [ordered_probit_pmf(i,ns_p,c,scale_p) for i in 1:max_n_samples];
end


####
function make_qmeta_FL(p_o1, o1_u, o2_u, safe_u)
    outcome_u_diff = [o1_u, o2_u].- safe_u;
    outcome_probs = [p_o1, (1 - p_o1)];
    qmeta = [outcome_probs[outcome_idx]*abs(outcome_u_diff[outcome_idx])
                for outcome_idx in 1:length(outcome_probs)];
    
    qmeta_new = qmeta ./ sum(qmeta);
    
   # if isnan(qmeta_new[1])
   #     println("ufail: ", o1_u, o2_u)
   # end
    return qmeta_new
end


function make_qmeta_prob(p_o1)
    return [p_o1, (1 - p_o1)]
end

function make_qmeta_ard(p_o1, o1_u, o2_u, safe_u, n_samples)
    baseline_u = 0;
    outcome_u_diff = [o1_u, o2_u] .- safe_u;
    outcome_probs = [p_o1, (1 - p_o1)];
    qmeta = [outcome_probs[outcome_idx] * 
                abs(outcome_u_diff[outcome_idx])*
                    (sqrt( (1 + abs(outcome_u_diff[outcome_idx] - baseline_u) *
                            sqrt(n_samples))/(abs(outcome_u_diff[outcome_idx] - baseline_u)*sqrt(n_samples))))
                for outcome_idx in 1:length(outcome_probs)];
    
    qmeta_new = qmeta ./ sum(qmeta);
    
   # if isnan(qmeta_new[1])
   #     println("ufail: ", o1_u, o2_u)
   # end
    return qmeta_new
end

# sample distribution
function make_qmeta_glm(p_o1, o1_u, o2_u, safe_u, int_beta, better_beta, prob_beta, rew_beta, extr_beta)
    p_o2 = 1 - p_o1;
    outcome_u_diff = [o1_u, o2_u] .- safe_u;
    o1_better = 2*(o1_u > o2_u) - 1.; # weird... 
    dv = int_beta + better_beta*o1_better + prob_beta*(p_o1 - p_o2) + rew_beta*(abs(o1_u) - abs(o2_u)) + extr_beta*(abs(outcome_u_diff[1]) - abs(outcome_u_diff[2]));
    qm_o1 = logistic(dv);
    return [qm_o1, (1 - qm_o1)];
end

# what if you skip the even number of samples? 

function sample_glm_lik2(params,sub_data; max_n_samples = 2, simulate = false, odd_samples_only = false, transform_params = true, use_w = true, do_UWS = false, do_Prob = false);
    
    # start likelihood count
    lik = 0;
    
    # initialize parameters
    int_beta = params[1];
    better_beta = params[2];
    prob_beta = params[3];
    rew_beta = params[4];
    extr_beta = params[5];
    
    if transform_params
        gain_power = 1;#exp(params[7]/50); # think you don't need these...
        loss_power = 1;#exp(params[8]/50);
        choice_noise = 0.5 + 0.5 * erf(params[6] / sqrt(2));

    else
        gain_power = 1;#params[7]; # think you don't need these...
        loss_power = 1;#params[8];
        choice_noise = params[6];
    end
    
    if use_w
        sample_weight = 0.5 + 0.5 * erf(params[7] / sqrt(2));
    else
        ns_p = params[7]; # parameter to control number of samples; 
    end
    
    # this is w if use_w is true and will balance btw 1 and 2 samples.
    scale_p = exp(params[8]/2) + .1;
    
    if do_UWS
        gain_power = exp(params[9]/50);
        loss_power = exp(params[10]/50);
    end
    
    # set n trials, etc.
    n_trials = size(sub_data,1);
    if simulate
        accept_prob = zeros(typeof(int_beta),n_trials);
    end
    
    choice_o1_prob = typeof(int_beta)[.2, .4, .6, .8];

    # transform these to starting points
    point_scale = 1/15;
    
    if odd_samples_only
        possible_n_samples = 1:2:max_n_samples
    else
        possible_n_samples = 1:max_n_samples;
    end
    
    # 
    # make pdf over number of samples agent draws
    if !use_w
        n_sample_pmf = make_n_sample_pmf(length(possible_n_samples), ns_p, scale_p);
    end
    
   # println(ns_p)
   # println(n_sample_pmf)
    
    for trial_idx in 1:n_trials
        
        # this trial info
        choice_idx = Int(sub_data[trial_idx, :choice_number]);
        p_o1 = choice_o1_prob[choice_idx];
        o1_val = sub_data[trial_idx, :o1_val]*point_scale;
        o2_val = sub_data[trial_idx, :o2_val]*point_scale;
        safe_val = sub_data[trial_idx, :safe_val]*point_scale;
        accept = Int(sub_data[trial_idx, :accept]);
        outcome_reached = Int(sub_data[trial_idx, :outcome_reached]);
        high_rt = 0#sub_data[trial_idx, :high_rt];
        
        # get outcome utility for each outcome / safe value ... / ... / ... / ...
        (o1_u, o2_u, safe_u) = util_power(o1_val, o2_val, safe_val, gain_power, loss_power; sub_baseline = true);
    
        if !do_UWS
            qmeta = make_qmeta_glm(p_o1, o1_u, o2_u, safe_u, int_beta, better_beta, prob_beta, rew_beta, extr_beta);
        end
        
        outcome_u_diff = [o1_u, o2_u] .- safe_u;


        
        # what's the probability we would accept for each number of samples...
        accept_prob_for_nsample = zeros(typeof(int_beta), length(possible_n_samples));
        
        ######### Here, loop through the number of samples agent takes
        ## we need to marginalize accept probability over these...
        

        
        
        # try to only take an odd number of samples
        for n_samples_idx = 1:length(possible_n_samples)
            
            #n_samples_o1_idx = typeof(gain_power)[collect(0:1:n_samples);];
            n_samples = possible_n_samples[n_samples_idx];
            
            if do_UWS
               # qmeta = make_qmeta_ard(p_o1, o1_u, o2_u, safe_u,n_samples);
                qmeta = make_qmeta_FL(p_o1, o1_u, o2_u, safe_u)
            elseif do_Prob
                qmeta = make_qmeta_prob(p_o1);
            end
                
                    # importance weights
            w = [p_o1, (1 - p_o1)] ./ qmeta;

            
            # possible number of samples which came out as O1 (one index for each possibility)
            n_samples_o1_idx = 0:1:n_samples;
    
            # probability of each of these possibilities
            
            
            #this_distr = Distributions.Binomial{typeof(gain_power)}(n_samples,qmeta[1])
            
            #this_distr = convert(typeof(Binomial(1,.2)), this_distr)
            
            
            prob_draw_n_o1 = [binomial_pdf(n_samples_o1_idx[i], n_samples,qmeta[1]) for i in 1:length(n_samples_o1_idx)]
            
            
            #Distributions.pdf(this_distr, n_samples_o1_idx); # probability of drawing (index - 1) number of o1

            # compute the sum of weights for each sample (for each of these cases)
            sum_w_vec = n_samples_o1_idx.*w[1] + (n_samples .- n_samples_o1_idx).*w[2];
            
            # get sum of weight * u_diff for each sample drawn (for each possible number of samples we might draw)
            sum_weighted_u_diff = n_samples_o1_idx.*w[1].*outcome_u_diff[1] + (n_samples .- n_samples_o1_idx).*w[2].*outcome_u_diff[2];
            
            # this is our estimate of the difference between gamble and safe, for each case
            estimate_vec = sum_weighted_u_diff ./ sum_w_vec; # what would i choose for each of these?
            
            # accept probability for each of these cases 
            # deterministic if > or < 0. .5 if == 0
            
            # try this which might be more differentiable...
            accept_vec = logistic.(50 .* estimate_vec);#typeof(int_beta)[1. .* (estimate_vec .>= 0);];
            #accept_vec[estimate_vec .== 0] .= .5;
            # average each accept prob by the prob we'd draw that number of 01 samples
            accept_prob_for_nsample[n_samples_idx] = prob_draw_n_o1'*accept_vec;
        end
#        println("acc_prob_n_samples: ", accept_prob_for_nsample)

        if use_w
            prob_accept_pre_noise = (1 - sample_weight)*accept_prob_for_nsample[1] + sample_weight*accept_prob_for_nsample[2];
        else
            prob_accept_pre_noise = n_sample_pmf'*accept_prob_for_nsample;
        end
            
        
 #       println("prob_accept: ", prob_accept_pre_noise)
                
        # add the choice_noise
        prob_accept = (1 - choice_noise)*prob_accept_pre_noise + choice_noise*(1/2);
        
        if simulate
            accept_prob[trial_idx] = prob_accept;
        else
            if accept == 1
                prob_choice = prob_accept;
            else
                prob_choice = 1 - prob_accept;
            end
            
            if prob_choice .< 1e-10
                prob_choice = 1e-10;
            end
            lik = lik + log(prob_choice);
        end 
    end
    if simulate 
        return accept_prob
    else
        return -1*lik;
    end
end

# returns expected nubmer of points, for each decision.
function compute_expected_points(subj_accept_probs, sub_data)
    p_o1 = sub_data[!,:choice_number] ./ 5;
    gamble_ev = p_o1.*sub_data[!,:o1_val] + (1 .- p_o1).*sub_data[!,:o2_val];
    safe_ev = sub_data[!,:safe_val];
    expected_points = subj_accept_probs.*gamble_ev + (1 .- subj_accept_probs).*safe_ev;
    return sum(expected_points)
end

function comp_mean_points(sim_fun,params,data)
    group_points = 0;
    for s_idx in unique(data[!,:sub])
        sub_data = @where(data, :sub .== s_idx);
        subj_accept_probs = sim_fun(params, sub_data);
        group_points += compute_expected_points(subj_accept_probs, sub_data);
    end
    mean_points = group_points / length(unique(data[!,:sub]));
    return mean_points
end

function build_names_sglm2(use_int, use_better, use_prob, use_rew, use_extr, max_n_samples; use_w = true, odd_samples_only = false, add_choice_noise = true)
    
    param_names = [];
    title_str = "sglm_max_$max_n_samples";
    
    if odd_samples_only
        title_str = title_str*"_OO_";
    end
    
    if use_w
        title_str = title_str*"_w_";
    end
    
    if use_int
        push!(param_names, "beta_int");
        title_str = title_str*"Int";
    end
    
    if use_better
        push!(param_names, "better_int");
        title_str = title_str*"Better";
    end
    
    if use_prob
        push!(param_names, "beta_prob")
        title_str = title_str*"Prob";
    end
    
    if use_rew
        push!(param_names, "beta_rew")
        title_str = title_str*"Rew";
    end
    
    if use_extr
        push!(param_names, "beta_extr")
        title_str = title_str*"Extr";
    end
    
    if add_choice_noise
        push!(param_names, "choice_noise")
    end
    
    if use_w
        push!(param_names, "w")
    else
        push!(param_names, "ns_p")
    end
    #push!(param_names, "scale_p")
    
    return (param_names, title_str)

end

function build_lik_sglm2(params_in, sub_data, use_int, use_better, use_prob, use_rew, use_extr; max_n_samples = 2, simulate = false, transform_params = true, odd_samples_only = false, use_w = true, add_choice_noise = true)
    
    p_idx = 1;
    
    if use_int
        int_beta = params_in[p_idx]; p_idx += 1;
    else
        int_beta = 0;
    end
    
    if use_better
        better_beta = params_in[p_idx]; p_idx += 1;
    else
        better_beta = 0;
    end
    
    if use_prob
        prob_beta = params_in[p_idx]; p_idx += 1;
    else
        prob_beta = 0;
    end
    
    if use_rew
        rew_beta = params_in[p_idx]; p_idx += 1;
    else
        rew_beta = 0;
    end
    
    if use_extr
        extr_beta = params_in[p_idx]; p_idx += 1;
    else
        extr_beta = 0;
    end
    
    if add_choice_noise
        choice_noise = params_in[p_idx]; p_idx += 1;
    else
        choice_noise = -1000;#params_in[p_idx]; p_idx += 1;
    end
    ns_p = params_in[p_idx]; p_idx += 1;
    scale_p = 2;#params_in[p_idx];
    
    params = [int_beta, better_beta, prob_beta, rew_beta, extr_beta, choice_noise, ns_p, scale_p];
    
    return sample_glm_lik2(params,sub_data; max_n_samples = max_n_samples, simulate = simulate, odd_samples_only = odd_samples_only, transform_params = true, use_w = use_w);
    
end


function uws_lik_mults(params_in, sub_data; max_n_samples = 7, simulate = false, transform_params = true, odd_samples_only = false, use_w = true, add_choice_noise = true, do_UWS = true)
    
    # initialize parameters
    int_beta = 0.;
    better_beta = 0.;
    prob_beta = 0.;
    rew_beta = 0.;
    extr_beta = 0.;
    
    choice_noise = params_in[1];
    ns_p = params_in[2];
    gain_power = params_in[3]; # think you don't need these...
    loss_power = params_in[4];
    
    scale_p = 2;
    
    params = [int_beta, better_beta, prob_beta, rew_beta, extr_beta, choice_noise, ns_p, scale_p, gain_power, loss_power];
    
    return sample_glm_lik2(params,sub_data; max_n_samples = max_n_samples, simulate = simulate, odd_samples_only = odd_samples_only, transform_params = true, use_w = use_w, do_UWS = true, do_Prob = false);
end

function prob_lik_mults(params_in, sub_data; max_n_samples = 7, simulate = false, transform_params = true, odd_samples_only = false, use_w = true, add_choice_noise = true, do_UWS = false)
    
    # initialize parameters
    int_beta = 0.;
    better_beta = 0.;
    prob_beta = 0.;
    rew_beta = 0.;
    extr_beta = 0.;
    
    choice_noise = params_in[1];
    ns_p = params_in[2];
    gain_power = params_in[3]; # think you don't need these...
    loss_power = params_in[4];
    
    scale_p = 2;
    
    params = [int_beta, better_beta, prob_beta, rew_beta, extr_beta, choice_noise, ns_p, scale_p, gain_power, loss_power];
    
    return sample_glm_lik2(params,sub_data; max_n_samples = max_n_samples, simulate = simulate, odd_samples_only = odd_samples_only, transform_params = true, use_w = use_w, do_Prob = true, do_UWS = false);
end
    