
# fit the em model in this model dict to data, using experiment parmas
# if parallel, make sure you already loaded toolbox and cores, etc.
function fit_model_em(this_model,data, to_save_folder; full = false, parallel = false, emtol = 1e-3, maxiter = 100, run_loo = false, save_res = true, return_res = false)
    
    # setup up input for EM
    param_names = this_model["param_names"];
    
    # input for em
    NP = length(param_names);
    NS = length(unique(data[!,:sub]));
    println("NS: ", NS)
    subs = 1:NS;
    X = ones(NS);
    
    # starting parameters for seach
    betas_start = 1. * ones(1,NP);
    sigma_start = 2. * ones(NP);

    
    # define the likelihood function
    likfun_model = this_model["likfun"];
    likfun_task = (x,data) -> likfun_model(x,data);

    # fit the model using em
    println("calling em")
    (betas,sigma,x,l,h) = em(data,subs,X,betas_start,sigma_start,likfun_task; maxiter = maxiter, emtol=emtol, parallel=parallel, full=full);

    # get ibic and iaic
    model_ibic = ibic(x,l,h,betas,sigma,size(data,1))
    model_iaic = iaic(x,l,h,betas,sigma)

    # save the results
    model_res = Dict();
    model_name = this_model["model_name"];
    model_res["model_name"] = model_name;
    model_res["param_names"] = this_model["param_names"];
    model_res["betas"] = betas;
    model_res["sigma"] = sigma;
    model_res["x"] = x;
    model_res["l"] = l;
    model_res["h"] = h;
    model_res["ibic"] = model_ibic;
    model_res["iaic"] = model_iaic;
    model_res["sub"] = unique(data[!,:sub])
    
    
    if run_loo
        liks = loocv(data,subs,x,X,betas,sigma,likfun_task;emtol=emtol, parallel=parallel, full=full)
        model_res["loo"] = liks;
    end
    
    
    #model_res["likfun"] = likfun_model;
    #return model_res
    if save_res
        @save("$to_save_folder/$model_name.jld2", model_res)
    end
    
    if return_res
        return model_res
    end
    
end



# maximum likelihood fit the function -- 
function fit_model_ml(this_model,data, to_save_folder; n_start = 3)
    
    # setup up input for EM
    param_names = this_model["param_names"];
    
    # input for em
    NP = length(param_names);
    NS = length(unique(data[!,:sub]));
    subs = 1:NS;
    X = ones(NS);
    
    # starting parameters for seach
    betas_start = 1. * ones(1,NP);
    sigma_start = 2. * ones(NP);
    
    # define the likelihood function
    likfun_model = this_model["likfun"];
    likfun_task = (x,data) -> likfun_model(x,data);

    # fit the model using em
    model_name = this_model["model_name"];
    IJulia.clear_output(true)
    println("calling ML fit: $model_name")
    
    l = zeros(NS);
    bic = zeros(NS);
    aic = zeros(NS);
    psuedo_Rsq = zeros(NS);
    x = zeros(NS, NP);
    for i = 1:NS
        print(i)
        sub_data = @where(data,:sub .== i);
        n_trials = size(sub_data,1);
        this_likfun = (x) -> likfun_task(x,@where(data,:sub .== i));
        
        l_curr = 1e9; x_curr = zeros(NP);
        for start_idx = 1:n_start
            startx = generate_start_vals(this_likfun, NP);
            (this_l, this_x) = optimizesubject(this_likfun,startx);
            if this_l < l_curr
                l_curr = this_l;
                x_curr = this_x;
            end
        end
        
        psuedo_Rsq[i] = l_curr / (log(.5)*n_trials);
        
        (l[i], x[i,:]) = (l_curr, x_curr);
        
        bic[i] = l_curr + (NP/2)*log(n_trials);
        aic[i] = l_curr + NP;
    end

    # save the results
    model_res_ml = Dict();
    model_res_ml["model_name"] = model_name;
    model_res_ml["param_names"] = this_model["param_names"];
    model_res_ml["x"] = x;
    model_res_ml["l"] = l; # anything else?
    model_res_ml["psuedo_R_sq"] = psuedo_Rsq;
    model_res_ml["bic"] = bic;
    model_res_ml["aic"] = aic;

    @save("$to_save_folder/$model_name.jld2", model_res_ml)
end