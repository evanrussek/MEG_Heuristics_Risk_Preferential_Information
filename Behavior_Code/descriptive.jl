using CSV
using DataFrames
using DataFramesMeta
using CategoricalArrays # for making data categorical
using Statistics # for median
using StatsBase # for mad
using Gadfly # for plotting
using Cairo # for plotting
using Fontconfig # for plotting
using MixedModels # for linear mixed effects models
using Loess # for Loess smoothing over response rates
using Interpolations # for flat smoothing over response rates / thresholds

function clean_data(data_raw; bad_subs = [])
    data = @linq data_raw |> 
        subset(:block_number .> 7, .!ismissing.(:phase), :phase .== "TEST", :accept .!= "NA") |>
        select(:s_num, :block_number, :safe_val_actual, :safe_val_base, :trigger_val_actual,
        :trigger_val_base, :other_noise, :o1_trigger, :gl_type, :accept, :rt, :phase, :p_trigger,
        :p_o1, :o1_val, :o2_val, :safe_val, :outcome_reached, :choice_number, :trial_number);
    data[!,:choice_number] = Int64.(data[!,:choice_number]);
    data[!,:task_block_number] = data[!,:block_number] .- 7;
    data[!,:gain_trial] = data[!,:safe_val] .> 0;
    data[!,:loss_trial] = data[!,:safe_val] .< 0;
    data[!,:sub] = data[!,:s_num];
    data.rt_sec = data.rt ./ 1000;
    data.accept = parse.(Float64,data.accept);
    data.accept = Int64.(data.accept);
    
    med_rts = by(data,[:sub, :safe_val_base, :trigger_val_base, :p_trigger], df -> DataFrame(med_rt = median(df.rt)));
        
    data = leftjoin(data, med_rts, on = [:sub, :safe_val_base, :trigger_val_base, :p_trigger]);
    data[!,:high_rt] = data[!,:med_rt] .<= data[!,:rt];
    
    for b in bad_subs
        data = @subset(data, :s_num .!= b);
    end
    data.sub = CategoricalArray(groupindices(groupby(data,:s_num)));

    return data
end

function get_data()
    data_raw = CSV.read("$project_folder/data/meg_behavior.csv", DataFrame);
    data = clean_data(data_raw);
end
    

function plot_subj_choice_data(data,s_num; include_rt = false)
    
    s_data = @subset(data, :s_num .== s_num);
    task_s_num = s_data.s_num[1];
    s_accept_data = by(s_data, [:safe_val_base, :trigger_val_base, :p_trigger], :accept => mean);
    s_accept_data.safe_val_base_cat = CategoricalArray(Int64.(s_accept_data.safe_val_base));
    
    s_accept_data.gain = s_accept_data.safe_val_base .> 0;
    
    gain_data = @subset(s_accept_data, :safe_val_base .> 0)
    
    # ... # ... #
    p_gain = plot(gain_data ,x = :p_trigger, y = :accept_mean, color = :safe_val_base_cat,
            xgroup = :trigger_val_base,
            Scale.color_discrete(levels = unique(sort(gain_data.safe_val_base_cat))),
            Scale.xgroup(levels = unique(sort(gain_data.trigger_val_base))),
            Geom.subplot_grid(Geom.point(), Geom.line()),
            Guide.ylabel("Prop Accept"),
            Guide.xlabel("Prob Win by Win - Loss Value"),
            Guide.ColorKey(title = "Safe Value"),
            Guide.title("Sub: $task_s_num Gain Trials"),
            Theme(line_width = 2pt));

    loss_data = @subset(s_accept_data, :safe_val_base .< 0)
    p_loss = plot(loss_data ,x = :p_trigger, y = :accept_mean, color = :safe_val_base_cat,
            xgroup = :trigger_val_base,
            Scale.color_discrete(levels = unique(sort(loss_data.safe_val_base_cat))),
            Scale.xgroup(levels = unique(sort(loss_data.trigger_val_base))),
            Geom.subplot_grid(Geom.point(), Geom.line()),
            Guide.ylabel("Prop Accept"),
            Guide.xlabel("Prob Loss by Win - Loss Value"),
            Guide.ColorKey(title = "Safe Value"),
            Guide.title("Sub: $task_s_num Loss Trials"),
            Theme(line_width = 2pt));
    
    p_rt = plot(s_data, x = :rt_sec, Geom.histogram(),
        Guide.xlabel("Response Time (Seconds)"), 
        Guide.ylabel("# Responses"),
        Guide.title("Sub: $task_s_num RT Dist"),
        Coord.cartesian(xmin = 0, xmax = 6));
    
    if include_rt
        return vstack([p_gain; p_loss; p_rt]);
    else
        return vstack([p_gain; p_loss]);
    end

end

function plot_subj_rt_by_cond(data, s_num)

    # plot the log_rt by condition...
    s_data = @subset(data, :sub .== s_num);
    s_data.log_rt = log.(s_data.rt);
    s_rt_data = by(s_data, [:safe_val_base, :trigger_val_base, :p_trigger], :log_rt => mean);
    s_rt_data.safe_val_base_cat = CategoricalArray(Int64.(s_rt_data.safe_val_base));

    gain_data = @subset(s_rt_data, :safe_val_base .> 0)
    p_gain = plot(gain_data ,x = :p_trigger, y = :log_rt_mean, color = :safe_val_base_cat,
                xgroup = :trigger_val_base,
                Scale.color_discrete(levels = unique(sort(gain_data.safe_val_base_cat))),
                Scale.xgroup(levels = unique(sort(gain_data.trigger_val_base))),
                Geom.subplot_grid(Geom.point(), Geom.line()),
                Guide.ylabel("Prop Accept"),
                Guide.xlabel("Prob Win by Win - Loss Value"),
                Guide.ColorKey(title = "Safe Value"),
                Guide.title("Sub: $s_num Gain Trials"),
                Theme(line_width = 2pt));

    loss_data = @subset(s_rt_data, :safe_val_base .< 0)
    p_loss = plot(loss_data ,x = :p_trigger, y = :log_rt_mean, color = :safe_val_base_cat,
            xgroup = :trigger_val_base,
            Scale.color_discrete(levels = unique(sort(loss_data.safe_val_base_cat))),
            Scale.xgroup(levels = unique(sort(loss_data.trigger_val_base))),
            Geom.subplot_grid(Geom.point(), Geom.line()),
            Guide.ylabel("Prop Accept"),
            Guide.xlabel("Prob Win by Win - Loss Value"),
            Guide.ColorKey(title = "Safe Value"),
            Guide.title("Sub: $s_num Loss Trials"),
            Theme(line_width = 2pt));

    this_plot = vstack([p_gain; p_loss]);
    return this_plot
end

function plot_group_choice_data(data; title_str = "")
    
    all_s_accept_data = by(data, [:sub,:safe_val_base, :trigger_val_base, :p_trigger], df -> DataFrame(accept = mean(df.accept)))
    agg_accept_data = by(all_s_accept_data, [:safe_val_base, :trigger_val_base, :p_trigger], df -> DataFrame(accept = mean(df.accept)));

    agg_accept_data.safe_val_base_cat = CategoricalArray(Int64.(agg_accept_data.safe_val_base));

    gain_data = @subset(agg_accept_data, :safe_val_base .> 0)
    gain_data.safe_val_base_cat = CategoricalArray(Int64.(gain_data.safe_val_base));

    # ... # ... #
    p_gain = plot(gain_data ,x = :p_trigger, y = :accept, group = :safe_val_base_cat,  color = :safe_val_base_cat,
            xgroup = :trigger_val_base,
            Scale.color_discrete(levels = unique(sort(gain_data.safe_val_base_cat))),
            Scale.xgroup(levels = unique(sort(gain_data.trigger_val_base))),
            Geom.subplot_grid(Geom.point(), Geom.line()),
            Guide.ylabel("Prop Accept"),
            Guide.xlabel("Prob Win by Win - Loss Value"),
            Guide.ColorKey(title = "Safe Value"),
            Guide.title("$title_str Gain Trials"),
            Theme(line_width = 2pt));

    loss_data = @subset(agg_accept_data, :safe_val_base .< 0);
    p_loss = plot(loss_data ,x = :p_trigger, y = :accept, color = :safe_val_base_cat,
            xgroup = :trigger_val_base,
            Scale.color_discrete(levels = unique(sort(loss_data.safe_val_base_cat))),
            Scale.xgroup(levels = unique(sort(loss_data.trigger_val_base))),
            Geom.subplot_grid(Geom.point(), Geom.line()),
            Guide.ylabel("Prop Accept"),
            Guide.xlabel("Prob Loss by Win - Loss Value"),
            Guide.ColorKey(title = "Safe Value"),
            Guide.title("$title_str Loss Trials"),
            Theme(line_width = 2pt));
    
    return vstack([p_gain; p_loss]);
end

function make_agg_rt_data(data; split_accept = false)
    data.log_rt = log.(data.rt);
    if split_accept
        all_s_rt_data = by(data,[:sub, :safe_val_base, :trigger_val_base, :p_trigger,:accept], df -> DataFrame(log_rt = mean(df.log_rt)))
        agg_rt_data = by(all_s_rt_data, [:safe_val_base, :trigger_val_base, :p_trigger,:accept], df -> DataFrame(log_rt = mean(df.log_rt), rt = median(df.rt)))
    else
        all_s_rt_data = by(data,[:sub, :safe_val_base, :trigger_val_base, :p_trigger], df -> DataFrame(log_rt = mean(df.log_rt)));
        agg_rt_data = by(all_s_rt_data, [:safe_val_base, :trigger_val_base, :p_trigger], df -> DataFrame(log_rt = mean(df.log_rt)))
    end;
    agg_rt_data.safe_val_base_cat = CategoricalArray(Int64.(agg_rt_data.safe_val_base));
    return agg_rt_data
end;

function plot_group_rt_data(data; title_str = "", plot_type = 2)

    agg_rt_data = make_agg_rt_data(data);
    gain_data = @subset(agg_rt_data, :safe_val_base .> 0)
    gain_data.safe_val_base_cat = CategoricalArray(Int64.(gain_data.safe_val_base));

    if plot_type == 1
    # ... # ... #
        p_gain = plot(gain_data ,x = :p_trigger, y = :log_rt, group = :safe_val_base_cat,  color = :safe_val_base_cat,
                xgroup = :trigger_val_base,
                Scale.color_discrete(levels = unique(sort(gain_data.safe_val_base_cat))),
                Scale.xgroup(levels = unique(sort(gain_data.trigger_val_base))),
                Geom.subplot_grid(Geom.point(), Geom.line()),
                Guide.ylabel("Log RT"),
                Guide.xlabel("Prob Trigger by Win - Loss Value"),
                Guide.ColorKey(title = "Safe Value"),
                Guide.title("$title_str Gain Trials"),
                Theme(line_width = 2pt));


        loss_data = @subset(agg_rt_data, :safe_val_base .< 0);
        p_loss = plot(loss_data ,x = :p_trigger, y = :log_rt, color = :safe_val_base_cat,
                xgroup = :trigger_val_base,
                Scale.color_discrete(levels = unique(sort(loss_data.safe_val_base_cat))),
                Scale.xgroup(levels = unique(sort(loss_data.trigger_val_base))),
                Geom.subplot_grid(Geom.point(), Geom.line()),
                Guide.ylabel("Log RT"),
                Guide.xlabel("Prob Trigger by Win - Loss Value"),
                Guide.ColorKey(title = "Safe Value"),
                Guide.title("$title_str Loss Trials"),
                Theme(line_width = 2pt));
    else
        
        gain_data.p_trigger_cat = CategoricalArray(gain_data.p_trigger)
        p_gain = plot(gain_data ,x = :safe_val_base, y = :log_rt, group = :p_trigger_cat,  color = :p_trigger_cat,
                xgroup = :trigger_val_base,
                Scale.color_discrete(levels = unique(sort(gain_data.p_trigger))),
                Scale.xgroup(levels = unique(sort(gain_data.trigger_val_base))),
                Geom.subplot_grid(Geom.point(), Geom.line()),
                Guide.ylabel("Log RT"),
                Guide.xlabel("Prob Trigger by Win - Loss Value"),
                Guide.ColorKey(title = "Safe Value"),
                Guide.title("$title_str Gain Trials"),
                Theme(line_width = 2pt));


        loss_data = @subset(agg_rt_data, :safe_val_base .< 0);
        p_loss = plot(loss_data ,x = :p_trigger, y = :log_rt, color = :safe_val_base_cat,
                xgroup = :trigger_val_base,
                Scale.color_discrete(levels = unique(sort(loss_data.safe_val_base_cat))),
                Scale.xgroup(levels = unique(sort(loss_data.trigger_val_base))),
                Geom.subplot_grid(Geom.point(), Geom.line()),
                Guide.ylabel("Log RT"),
                Guide.xlabel("Prob Trigger by Win - Loss Value"),
                Guide.ColorKey(title = "Safe Value"),
                Guide.title("$title_str Loss Trials"),
                Theme(line_width = 2pt));
    end

    p = vstack([p_gain; p_loss]); # split this on accept vs reject... 
    return p
end


function make_agg_accept_data(data)
    all_s_accept_data = by(data, [:sub,:safe_val_base, :trigger_val_base, :p_trigger], df -> DataFrame(accept = mean(df.accept)))
    agg_accept_data = by(all_s_accept_data, [:safe_val_base, :trigger_val_base, :p_trigger], df -> DataFrame(accept = mean(df.accept)));
    agg_accept_data.safe_val_base_cat = CategoricalArray(Int64.(agg_accept_data.safe_val_base));
    return agg_accept_data
end


# this focuses on the probability
function plot_choice_sim_data_comp(data, simdata; title_str = "")

    agg_accept_data = make_agg_accept_data(data);
    gain_data = @subset(agg_accept_data, :safe_val_base .> 0)
    gain_data.safe_val_base_cat = CategoricalArray(Int64.(gain_data.safe_val_base));
    loss_data = @subset(agg_accept_data, :safe_val_base .< 0)
    loss_data.safe_val_base_cat = CategoricalArray(Int64.(loss_data.safe_val_base));

    agg_accept_sim = make_agg_accept_data(simdata)
    gain_sim = @subset(agg_accept_sim, :safe_val_base .> 0)
    gain_sim.safe_val_base_cat = CategoricalArray(Int64.(gain_sim.safe_val_base));
    loss_sim = @subset(agg_accept_sim, :safe_val_base .< 0)
    loss_sim.safe_val_base_cat = CategoricalArray(Int64.(loss_sim.safe_val_base));


    p_gain = plot(gain_data ,x = :p_trigger, group = :safe_val_base_cat,
                xgroup = :trigger_val_base, ygroup = :safe_val_base_cat,
                Scale.xgroup(levels = unique(sort(gain_data.trigger_val_base))),
                Scale.ygroup(levels = unique(sort(gain_data.safe_val_base_cat))),
                Geom.subplot_grid(
                    layer(gain_data, y = :accept, Geom.point()),
                    layer(gain_sim, y = :accept, Geom.line),
            Guide.xticks(ticks = [0, .5, 1]),
            Guide.yticks(ticks = [0, .5, 1]),

            ),
                Guide.ylabel("Prop Accept by Safe Value"),
                Guide.xlabel("Prob Trig by Trig Value"),
                Guide.title("$title_str Gain Trials"), 
                Theme(line_width = 1.25pt, point_size = 1.8pt));

    p_loss = plot(loss_data ,x = :p_trigger, group = :safe_val_base_cat, 
                xgroup = :trigger_val_base, ygroup = :safe_val_base_cat,
                Scale.xgroup(levels = unique(sort(loss_data.trigger_val_base))),
                Scale.ygroup(levels = unique(sort(loss_data.safe_val_base))),

                Geom.subplot_grid(
                    layer(loss_data, y = :accept, Geom.point),
                    layer(loss_sim, y = :accept, Geom.line),
            Guide.xticks(ticks = [0, .5, 1]),
            Guide.yticks(ticks = [0, .5, 1])          
            ),
                Guide.ylabel("Prop Accept by Safe Value"),
                Guide.xlabel("Prob Trig by Trig Value"),
                Guide.title("$title_str Loss Trials"),
                Theme(line_width = 1.25pt, point_size = 1.8pt));
    return hstack([p_gain; p_loss])
end

# this focuses on the probability
function plot_choice_sim_data_comp2(data, ev_simdata, simdata2; title_str = "")
    
    simdata = ev_simdata

    agg_accept_data = make_agg_accept_data(data);
    gain_data = @subset(agg_accept_data, :safe_val_base .> 0)
    gain_data.safe_val_base_cat = CategoricalArray(Int64.(gain_data.safe_val_base));
    loss_data = @subset(agg_accept_data, :safe_val_base .< 0)
    loss_data.safe_val_base_cat = CategoricalArray(Int64.(loss_data.safe_val_base));

    agg_accept_sim = make_agg_accept_data(ev_simdata)
    gain_sim = @subset(agg_accept_sim, :safe_val_base .> 0)
    gain_sim.safe_val_base_cat = CategoricalArray(Int64.(gain_sim.safe_val_base));
    loss_sim = @subset(agg_accept_sim, :safe_val_base .< 0)
    loss_sim.safe_val_base_cat = CategoricalArray(Int64.(loss_sim.safe_val_base));

    agg_accept_sim2 = make_agg_accept_data(simdata2)
    gain_sim2 = @subset(agg_accept_sim2, :safe_val_base .> 0)
    gain_sim2.safe_val_base_cat = CategoricalArray(Int64.(gain_sim2.safe_val_base));
    loss_sim2 = @subset(agg_accept_sim2, :safe_val_base .< 0)
    loss_sim2.safe_val_base_cat = CategoricalArray(Int64.(loss_sim2.safe_val_base));
    
    
    p_gain = plot(gain_data ,x = :p_trigger, group = :safe_val_base_cat,
                xgroup = :trigger_val_base, ygroup = :safe_val_base_cat,
                Scale.xgroup(levels = unique(sort(gain_data.trigger_val_base))),
                Scale.ygroup(levels = unique(sort(gain_data.safe_val_base_cat))),
                Geom.subplot_grid(
                    layer(gain_data, y = :accept, Geom.point(), Theme(default_color=colorant"grey",  point_size = 1.8pt)),
                    layer(gain_sim, y = :accept, Geom.line, Theme(default_color=colorant"orange", line_width = 1pt)),
                    layer(gain_sim2, y = :accept, Geom.line, Theme(default_color=colorant"blue", line_width = 1pt)),
            Guide.xticks(ticks = [0, .5, 1]),
            Guide.yticks(ticks = [0, .5, 1]),
            ),
                Guide.ylabel("Prop. Accept by Safe Val."),
                Guide.xlabel("Prob. Trig. Outcome by Trig. Val."),
                Guide.title(""),
        Guide.manual_color_key("", ["Data", title_str, "Expected Value"], ["grey", "blue","orange" ])) 
    
    
    

    p_loss = plot(loss_data ,x = :p_trigger, group = :safe_val_base_cat, 
                xgroup = :trigger_val_base, ygroup = :safe_val_base_cat,
                Scale.xgroup(levels = unique(sort(loss_data.trigger_val_base))),
                Scale.ygroup(levels = unique(sort(loss_data.safe_val_base))),

                Geom.subplot_grid(
                    layer(loss_data, y = :accept, Geom.point(), Theme(default_color=colorant"grey",  point_size = 1.8pt)),
                    layer(loss_sim, y = :accept, Geom.line, Theme(default_color=colorant"orange", line_width = 1pt)),
                    layer(loss_sim2, y = :accept, Geom.line, Theme(default_color=colorant"blue", line_width = 1pt)),
            Guide.xticks(ticks = [0, .5, 1]),
            Guide.yticks(ticks = [0, .5, 1])          
            ),
                Guide.ylabel("Prop. Accept by Safe Val."),
                Guide.xlabel("Prob. Trig. Outcome by Trig. Val."),
                Guide.title(""),
            Guide.manual_color_key("", ["Data", title_str, "Expected Value"], ["grey", "blue","orange" ])) 

        return hstack([p_gain; p_loss])
end







# this focuses on the trig
function plot_choice_sim_data_comp_trig(data, simdata; title_str = "")

    agg_accept_data = make_agg_accept_data(data);
    gain_data = @subset(agg_accept_data, :safe_val_base .> 0)
    gain_data.safe_val_base_cat = CategoricalArray(Int64.(gain_data.safe_val_base));
    loss_data = @subset(agg_accept_data, :safe_val_base .< 0)
    loss_data.safe_val_base_cat = CategoricalArray(Int64.(loss_data.safe_val_base));

    agg_accept_sim = make_agg_accept_data(simdata)
    gain_sim = @subset(agg_accept_sim, :safe_val_base .> 0)
    gain_sim.safe_val_base_cat = CategoricalArray(Int64.(gain_sim.safe_val_base));
    loss_sim = @subset(agg_accept_sim, :safe_val_base .< 0)
    loss_sim.safe_val_base_cat = CategoricalArray(Int64.(loss_sim.safe_val_base));


    p_gain = plot(gain_data ,x = :trigger_val_base, group = :safe_val_base_cat,
                xgroup = :p_trigger, ygroup = :safe_val_base_cat,

                Scale.xgroup(levels = unique(sort(gain_data.p_trigger))),
                Scale.ygroup(levels = unique(sort(gain_data.safe_val_base_cat))),
                Geom.subplot_grid(
                    layer(gain_data, y = :accept, Geom.point),
                    layer(gain_sim, y = :accept, Geom.line),Guide.yticks(ticks = [0, .2, .4, .6, .8, 1])),
                Guide.ylabel("Prop. Accept"),
                Guide.xlabel("Trig Value by Prob Trig"),
                Guide.title("$title_str Gain Trials"),
                Theme(line_width = 2pt));

    p_loss = plot(loss_data ,x = :trigger_val_base,
                xgroup = :p_trigger, ygroup = :safe_val_base_cat,
                Scale.xgroup(levels = unique(sort(loss_data.p_trigger))),
                Scale.ygroup(levels = unique(sort(loss_data.safe_val_base))),
                Geom.subplot_grid(
                    layer(loss_data, y = :accept, Geom.point),
                    layer(loss_sim, y = :accept, Geom.line)),
                Guide.ylabel("Prop Accept"),
                Guide.xlabel("Trig Value by Prob Trig"),
                Guide.title("$title_str Loss Trials"),
                Theme(line_width = 2pt));
    return hstack([p_gain; p_loss])
end

# this focuses on the trig
function plot_choice_sim_data_comp_safe(data, simdata; title_str = "")

    agg_accept_data = make_agg_accept_data(data);
    gain_data = @subset(agg_accept_data, :safe_val_base .> 0)
    gain_data.safe_val_base_cat = CategoricalArray(Int64.(gain_data.safe_val_base));
    loss_data = @subset(agg_accept_data, :safe_val_base .< 0)
    loss_data.safe_val_base_cat = CategoricalArray(Int64.(loss_data.safe_val_base));

    agg_accept_sim = make_agg_accept_data(simdata)
    gain_sim = @subset(agg_accept_sim, :safe_val_base .> 0)
    gain_sim.safe_val_base_cat = CategoricalArray(Int64.(gain_sim.safe_val_base));
    loss_sim = @subset(agg_accept_sim, :safe_val_base .< 0)
    loss_sim.safe_val_base_cat = CategoricalArray(Int64.(loss_sim.safe_val_base));

    p_gain = plot(gain_data ,x = :safe_val_base,
                xgroup = :trigger_val_base, ygroup = :p_trigger,
                Scale.xgroup(levels = unique(sort(gain_data.trigger_val_base))),
                Scale.ygroup(levels = unique(sort(gain_data.p_trigger))),
                Geom.subplot_grid(
                    layer(gain_data, y = :accept, Geom.point),
                    layer(gain_sim, y = :accept, Geom.line)),
                Guide.ylabel("Prop Accept by Prob Trig"),
                Guide.xlabel("Safe by Trig Value"),
                Guide.title("$title_str Gain Trials"),
                Theme(line_width = 2pt));

    p_loss = plot(loss_data ,x = :safe_val_base, 
                xgroup = :trigger_val_base, ygroup = :p_trigger,
                Scale.xgroup(levels = unique(sort(loss_data.trigger_val_base))),
                Scale.ygroup(levels = unique(sort(loss_data.p_trigger))),

                Geom.subplot_grid(
                    layer(loss_data, y = :accept,Geom.point),
                    layer(loss_sim, y = :accept,Geom.line)),
                Guide.ylabel("Prop Accept by Prob Trig"),
                Guide.xlabel("Safe by Trig Value"),
                Guide.title("$title_str Loss Trials"),
                Theme(line_width = 2pt));
    return hstack([p_gain; p_loss])
end
