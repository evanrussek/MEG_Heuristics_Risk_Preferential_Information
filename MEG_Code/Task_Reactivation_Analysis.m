% Analyze Task Reactivations. Makes all Task Reactivaiton Figures.

% Enter MEG Decision Study Folder
study_folder = '/Users/erussek/Dropbox/MEG_Decision_Study_WC';

% run permutations and save figs?
run_permutations = false;
make_permutation_plots = false;
run_permutations_key = false;
make_permutation_plots_key = false;
NPerm = 2;
save_all_figs = false;
save_perm_dists = false;

% start figure counter
fig_idx = 0;

% File name for task reactivations
penalty_l1 = .002;
prop_null = 1;
RP_file_name = ['Test_Train_Out_vs_All_L1_', num2str(penalty_l1),'_PN_',num2str(prop_null),'.mat'];

% some info on train and test times
delay_vals_train = 0:10:500;
delay_vals_test = 0:10:500;
NTr = length(delay_vals_train);
NTe = length(delay_vals_test);

% remove 2 subjs who had too many repeating choices (see Methods).
subj_list = 1:21; bad_subjs = [6, 19];
for bs = bad_subjs
    subj_list = subj_list(subj_list ~= bs);
end
NS = length(subj_list);

%% load chioce trial type information tables
choice_info_tables = cell(NS,1);
for s_idx = 1:NS
    s_num = subj_list(s_idx); 
    time_file = [study_folder, '/All_Event_Info_Tables/Subj_', num2str(s_num), 'All_Event_Time_Table.mat'];
    temp = load(time_file, 'subj_table'); % time table 
    choice_info_tables{s_idx} =  temp.subj_table(contains(temp.subj_table.event, 'CHOICE'),:);
end

%% LOAD REACTIVATION DATA
group_struc_folder = fullfile(study_folder, 'Classifier_Activations', 'Task_Reactivations');
% group_struc_folder = fullfile(study_folder, 'Classifier_Activations_Test', 'Task_Reactivations');

subj_rp_data = cell(NS,1);
for s_idx = 1:NS
    % load this subject's reactivations
    s_num = subj_list(s_idx);
    subj_struc_folder = fullfile(group_struc_folder, ['subj_', num2str(s_num)]);
    temp = load(fullfile(subj_struc_folder,RP_file_name), 'predact_mtx');
    
    % convert to reactivation probability
    subj_rp_data{s_idx}  = 1 ./ (1 + exp(-temp.predact_mtx));
end

%% LOAD BEHAVIORAL COVARIATES
temp = load(fullfile(study_folder, 'Behavioral_Params.mat'));
beh_params = temp.Beh_Params;


%% Run first level to predict O1 vs O2 from between trial conditions

% structure to store betas
NBetas = 2; % one beta for prob effect and one beta for reward effect
first_level_betas = zeros(NS, NBetas, NTr, NTe);
min_rt = 500;
for s_idx = 1:NS
    
    % get RP O2 - O1 diff -- var to be predited
    rp_data = subj_rp_data{s_idx};
    rp_diff = squeeze(rp_data(:,:,:,2)) - squeeze(rp_data(:,:,:,1));

    % make design matrix
    subj_table = choice_info_tables{s_idx};
    keep_trials = find(subj_table.rt >= min_rt); % keep this?
    abs_rew_diff = abs(subj_table.o2_val) - abs(subj_table.o1_val);
    prob_diff = (5 - subj_table.choice_number)./5 - .5; % prob related to choice number
    X = [ones(length(prob_diff),1),prob_diff, abs_rew_diff];

    % run regression separately for each train time-point
    temp_betas = zeros(size(X,2), size(rp_diff,2), size(rp_diff,3));
    for tt_idx = 1:size(rp_diff,2) % run down test_idx... % can run through this more dynamically...
        temp_betas(:,tt_idx, :) = pinv(X(keep_trials,:))*squeeze(rp_diff(keep_trials,tt_idx,:));
    end
    first_level_betas(s_idx,:,:,:) = temp_betas(2:3,:,:); % don't need intercept
end
    

%% which subjs to use for each analysis -- note no BIS data for subj. 1.
keep_subj_neural = 1:NS;
keep_subj_neuralQ = 2:NS;
keep_subj_beh = 1:NS;
keep_subj_BIS = 2:NS;

%%  turn off regression warnings - necessary for permutation tests.
warning('off', 'all')

%% Predict prob beh. from prob. neural
behavioral_betas = beh_params.beta_prob(keep_subj_beh);
neural_betas = squeeze(first_level_betas(keep_subj_neural,1,:,:)); % this is the prob effect
neur_prob_beh_prob_t_map = run_btw_subj_regression(neural_betas, behavioral_betas);
smooth_neur_prob_beh_prob_t_map = smooth_map(1.5,neur_prob_beh_prob_t_map,0);
[max_neur_prob_beh_prob, train_time, test_time] = get_max_train_test(smooth_neur_prob_beh_prob_t_map, 0:10:500, 0:10:500);
fig_idx = fig_idx + 1; close(figure(fig_idx)); figure(fig_idx);
imagesc(delay_vals_train, delay_vals_test, smooth_neur_prob_beh_prob_t_map)
xlabel({'Test Timepoint: (ms)',  'Rel. Prob. Stim. Onset'})
ylabel({'Train Timepoint (ms)'; 'Rel. Out. Stim. Onset'})
title("Neural Prob ~ Beh. Prob Relationship")
axis square
colorbar

% run permutation test
if run_permutations_key
    display("Running Neural Prob Beh Prob Permutations");
    neur_prob_beh_prob_perm_dist = run_btw_subj_permutation(neural_betas(:,3:end,:),behavioral_betas, NPerm);
end

% plot permutation distribution and compute significance...
if make_permutation_plots_key
    fig_idx = fig_idx + 1; close(figure(fig_idx)); figure(fig_idx);
    hold on
    histogram(neur_prob_beh_prob_perm_dist)
    xline(max_neur_prob_beh_prob)
    p_val = 1 - sum(max_neur_prob_beh_prob > neur_prob_beh_prob_perm_dist)/length(neur_prob_beh_prob_perm_dist);
    title(['Neural Prob ~ Beh. Prob Relationship; t_{peak} = ',num2str(max_neur_prob_beh_prob), ' p = ', num2str(p_val)])
    xlabel('t-statistic')
    ylabel('Number of permutations')
end


%% make the raw plot
fig_idx = fig_idx + 1; close(figure(fig_idx)); figure(fig_idx);
make_raw_plot(neural_betas,behavioral_betas,delay_vals_train, delay_vals_test,train_time,test_time);

%%  predict rew beh. from prob. neural
behavioral_betas = beh_params.beta_rew(keep_subj_beh);
neural_betas = squeeze(first_level_betas(keep_subj_neural,1,:,:)); % this is the prob effect
neur_prob_beh_rew_t_map = run_btw_subj_regression(neural_betas, behavioral_betas);
smooth_neur_prob_beh_rew_t_map = smooth_map(1.5,neur_prob_beh_rew_t_map,0);
[max_neur_prob_beh_rew, train_time, test_time] = get_max_train_test(smooth_neur_prob_beh_rew_t_map, 0:10:500, 0:10:500);
fig_idx = fig_idx + 1; close(figure(fig_idx)); figure(fig_idx);
imagesc(delay_vals_train, delay_vals_test, smooth_neur_prob_beh_rew_t_map)
xlabel({'Test Timepoint: (ms)',  'Rel. Prob. Stim. Onset'})
ylabel({'Train Timepoint (ms)', 'Rel. Out. Stim. Onset'})
title("Neural Prob ~ Beh. Rew Relationship")
axis square
colorbar

% run permutation test
if  run_permutations
    display("Running Neural Prob Beh Rew Permutations");
    neur_prob_beh_rew_perm_dist = run_btw_subj_permutation(neural_betas(:,3:end,:),behavioral_betas, NPerm);
end

% plot permutation distribution and compute significance...
if make_permutation_plots
    fig_idx = fig_idx + 1; close(figure(fig_idx)); figure(fig_idx);
    hold on
    histogram(neur_prob_beh_rew_perm_dist)
    xline(max_neur_prob_beh_rew)
    p_val = 1 - sum(max_neur_prob_beh_rew > neur_prob_beh_rew_perm_dist)/length(neur_prob_beh_rew_perm_dist);
    title(['Neural Prob ~ Beh. Rew Relationship; t_{peak} = ',num2str(max_neur_prob_beh_rew), ' p = ', num2str(p_val)])
    xlabel('t-statistic')
    ylabel('Number of permutations')
end
%% make the raw plot
fig_idx = fig_idx + 1; close(figure(fig_idx)); figure(fig_idx);
make_raw_plot(neural_betas,behavioral_betas,delay_vals_train, delay_vals_test,train_time,test_time);

%% predict rew neural. from beh rew. neural
behavioral_betas = beh_params.beta_rew(keep_subj_beh);
neural_betas = squeeze(first_level_betas(keep_subj_neural,2,:,:)); % this is the rew effect
neur_rew_beh_rew_t_map = run_btw_subj_regression(neural_betas, behavioral_betas);
smooth_neur_rew_beh_rew_t_map = smooth_map(1.5,neur_rew_beh_rew_t_map,0);
[max_neur_rew_beh_rew, train_time, test_time] = get_max_train_test(smooth_neur_rew_beh_rew_t_map, 0:10:500, 0:10:500);

fig_idx = fig_idx + 1; close(figure(fig_idx)); figure(fig_idx);
imagesc(delay_vals_train, delay_vals_test, smooth_neur_rew_beh_rew_t_map)
xlabel({'Test Timepoint: (ms)',  'Rel. Prob. Stim. Onset'})
ylabel({'Train Timepoint (ms)'; 'Rel. Out. Stim. Onset'})
title("Neural Rew ~ Beh. Rew Relationship")
axis square
colorbar

% run permutation test
if run_permutations_key
    display("Running Neural Rew Beh Rew Permutations");
    neur_rew_beh_rew_perm_dist = run_btw_subj_permutation(neural_betas(:,3:end,1:end),behavioral_betas, NPerm);
end

% plot permutation distribution and compute significance...
if make_permutation_plots_key
    fig_idx = fig_idx + 1; close(figure(fig_idx)); figure(fig_idx);
    hold on
    histogram(neur_rew_beh_rew_perm_dist)
    xline(max_neur_rew_beh_rew)
    p_val = 1 - sum(max_neur_rew_beh_rew > neur_rew_beh_rew_perm_dist)/length(neur_rew_beh_rew_perm_dist);
    title(['Neural Rew ~ Beh. Rew Relationship; t_{peak} = ',num2str(max_neur_rew_beh_rew), ' p = ', num2str(p_val)])
    xlabel('t-statistic')
    ylabel('Number of permutations')
end

% make raw plot
fig_idx = fig_idx + 1; close(figure(fig_idx)); figure(fig_idx);
make_raw_plot(neural_betas,behavioral_betas,delay_vals_train, delay_vals_test,train_time,test_time);

%%  predict rew neural. from prob. behavior
behavioral_betas = beh_params.beta_prob(keep_subj_beh);
neural_betas = squeeze(first_level_betas(keep_subj_neural,2,:,:)); % this is the prob effect
neur_rew_beh_prob_t_map = run_btw_subj_regression(neural_betas, behavioral_betas);
smooth_neur_rew_beh_prob_t_map = smooth_map(1.5,neur_rew_beh_prob_t_map,0);
[max_neur_rew_beh_prob, train_time, test_time] = get_max_train_test(smooth_neur_rew_beh_prob_t_map, 0:10:500, 0:10:500);

fig_idx = fig_idx + 1; close(figure(fig_idx)); figure(fig_idx);
imagesc(delay_vals_train, delay_vals_test, smooth_neur_rew_beh_prob_t_map)
xlabel({'Test Timepoint: (ms)',  'Rel. Prob. Stim. Onset'})
ylabel({'Train Timepoint (ms)'; 'Rel. Out. Stim. Onset'})
title("Neural Rew ~ Beh. Prob. Relationship")
axis square; colorbar;

% run permutation test
if run_permutations
    display("Running Neural Rew Beh Rew Permutations");
    neur_rew_beh_prob_perm_dist = run_btw_subj_permutation(neural_betas(:,3:end,1:end),behavioral_betas, NPerm);
end

% plot permutation distribution and compute significance...
if make_permutation_plots
    fig_idx = fig_idx + 1; close(figure(fig_idx)); figure(fig_idx);
    hold on
    histogram(neur_rew_beh_prob_perm_dist)
    xline(max_neur_rew_beh_prob)
    p_val = 1 - sum(max_neur_rew_beh_prob > neur_rew_beh_prob_perm_dist)/length(neur_rew_beh_prob_perm_dist);
    title(['Neural Rew ~ Beh. Prob Relationship; t_{peak} = ', num2str(max_neur_rew_beh_prob), ' p = ', num2str(p_val)])
    xlabel('t-statistic')
    ylabel('Number of permutations')
end
% make raw plot
fig_idx = fig_idx + 1; close(figure(fig_idx)); figure(fig_idx);
make_raw_plot(neural_betas,behavioral_betas,delay_vals_train, delay_vals_test,train_time,test_time);
%% Safe Activation - Reward Relationship
first_level_RP_safe = zeros(NS,NTr,NTe);
for s_idx = 1:NS
    rp_data = subj_rp_data{s_idx};
    first_level_RP_safe(s_idx,:,:) = squeeze(mean(rp_data(:,:,:,3),1));
   % first_level_RP_safe(s_idx,:,:) = squeeze(mean(rp_data(:,:,:,3) - mean(rp_data(:,:,:,1:2),4) ,1));
end

behavioral_betas = beh_params.beta_rew(keep_subj_beh);
neural_betas = squeeze(first_level_RP_safe(keep_subj_neural,:,:)); % this is the prob effect
neur_safe_beh_rew_t_map = run_btw_subj_regression(neural_betas, behavioral_betas);
smooth_neur_safe_beh_rew_t_map = smooth_map(1.5,neur_safe_beh_rew_t_map,0);
[max_neur_safe_beh_rew, train_time, test_time] = get_max_train_test(smooth_neur_safe_beh_rew_t_map, 0:10:500, 0:10:500);

% plot this
fig_idx = fig_idx + 1; close(figure(fig_idx)); figure(fig_idx);
imagesc(delay_vals_train(3:end), delay_vals_test, smooth_neur_safe_beh_rew_t_map(3:end,1:end))
xlabel({'Test Timepoint: (ms)',  'Rel. Prob. Stim. Onset'})
ylabel({'Train Timepoint (ms)'; 'Rel. Out. Stim. Onset'})
title("Neural Safe ~ Beh. Rew Relationship")
axis square
colorbar

% run permutation test
if run_permutations_key
    display("Running Neural Safe Beh Rew Permutations");
    neur_safe_beh_rew_perm_dist = run_btw_subj_permutation(neural_betas(:,3:end,1:end),behavioral_betas, NPerm);
    xlabel('t-statistic')
    ylabel('Number of permutations')
end

% plot permutation distribution and compute significance...
if make_permutation_plots_key
    fig_idx = fig_idx + 1; close(figure(fig_idx)); figure(fig_idx);
    hold on
    histogram(neur_safe_beh_rew_perm_dist)
    xline(max_neur_safe_beh_rew)
    p_val = 1 - sum(max_neur_safe_beh_rew > neur_safe_beh_rew_perm_dist)/length(neur_safe_beh_rew_perm_dist);
    title(['Neural Safe ~ Beh. Rew Relationship; t_{peak} = ',num2str(max_neur_safe_beh_rew), ' p = ', num2str(p_val)])
    xlabel('t-statistic')
    ylabel('Number of permutations')
end
fig_idx = fig_idx + 1; close(figure(fig_idx)); figure(fig_idx);
make_raw_plot(neural_betas,behavioral_betas,delay_vals_train, delay_vals_test,train_time,test_time);

%% Safe - Prob relationship

behavioral_betas = beh_params.beta_prob(keep_subj_beh);
neural_betas = squeeze(first_level_RP_safe(keep_subj_neural,:,:)); % this is the prob effect
neur_safe_beh_prob_t_map = run_btw_subj_regression(neural_betas, behavioral_betas);
smooth_neur_safe_beh_prob_t_map = smooth_map(1.5,neur_safe_beh_prob_t_map,0);
[max_neur_safe_prob_rew, train_time, test_time] = get_max_train_test(smooth_neur_safe_beh_prob_t_map, 0:10:500, 0:10:500);

% plot this
fig_idx = fig_idx + 1; close(figure(fig_idx)); figure(fig_idx);
imagesc(delay_vals_test, delay_vals_train(3:end), smooth_neur_safe_beh_prob_t_map(3:end,1:end))
xlabel({'Test Timepoint: (ms)',  'Rel. Prob. Stim. Onset'})
ylabel({'Train Timepoint (ms)'; 'Rel. Out. Stim. Onset'})
title("Neural Safe ~ Beh. Prob. Relationship")
axis square
colorbar


% run permutation test
if run_permutations
    display("Running Neural Safe Beh Rew Permutations");
    neur_safe_beh_prob_perm_dist = run_btw_subj_permutation(neural_betas(:,3:end,1:end),behavioral_betas, NPerm);
    xlabel('t-statistic')
    ylabel('Number of permutations')
end

% plot permutation distribution and compute significance...
if make_permutation_plots
    fig_idx = fig_idx + 1; close(figure(fig_idx)); figure(fig_idx);
    hold on
    histogram(neur_safe_beh_prob_perm_dist)
    xline(max_neur_safe_prob_rew)
    p_val = 1 - sum(max_neur_safe_prob_rew > neur_safe_beh_prob_perm_dist)/length(neur_safe_beh_prob_perm_dist);
    title(['Neural Safe ~ Beh. Prob Relationship; t_{peak} = ',num2str(max_neur_safe_prob_rew), ' p = ', num2str(p_val)])
    xlabel('t-statistic')
    ylabel('Number of permutations')
end

%% BIS score vs prob
behavioral_betas = beh_params.BIS(keep_subj_BIS);
neural_betas = squeeze(first_level_betas(keep_subj_neuralQ,1,:,:)); % this is the prob effect
neur_prob_beh_BIS_t_map = run_btw_subj_regression(neural_betas, behavioral_betas);
smooth_neur_prob_beh_BIS_t_map = smooth_map(1.5,neur_prob_beh_BIS_t_map,0);
[min_neur_prob_beh_BIS, train_time, test_time] = get_min_train_test(smooth_neur_prob_beh_BIS_t_map, 0:10:500, 0:10:500);
fig_idx = fig_idx + 1; close(figure(fig_idx)); figure(fig_idx);
imagesc(delay_vals_train, delay_vals_test, smooth_neur_prob_beh_BIS_t_map)
xlabel({'Test Timepoint: (ms)',  'Rel. Prob. Stim. Onset'})
ylabel({'Train Timepoint (ms)'; 'Rel. Out. Stim. Onset'})
title("Neural Prob ~ BIS Score")
axis square
colorbar

% run permutation test
if run_permutations_key
    display("Running Neural Prob BIS Permutations");
    neur_prob_beh_BIS_perm_dist = run_btw_subj_permutation(neural_betas(:,3:end,:),behavioral_betas, NPerm, true);
end

% plot permutation distribution and compute significance...
if make_permutation_plots_key
    fig_idx = fig_idx + 1; close(figure(fig_idx)); figure(fig_idx);
    hold on
    histogram(neur_prob_beh_BIS_perm_dist)
    xline(min_neur_prob_beh_BIS)
    p_val = 1 - sum(min_neur_prob_beh_BIS < neur_prob_beh_BIS_perm_dist)/length(neur_prob_beh_BIS_perm_dist);
    title(['Neural Prob ~ BIS Relationship; t_{peak} = ',num2str(min_neur_prob_beh_BIS), ' p = ', num2str(p_val)])
    xlabel('t-statistic')
    ylabel('Number of permutations')
end

% make raw plot
fig_idx = fig_idx + 1; close(figure(fig_idx)); figure(fig_idx);
make_raw_plot(neural_betas,behavioral_betas,delay_vals_train, delay_vals_test,train_time,test_time);
%% -- save all the permutation distributions... 

if save_perm_dists
    save('permdists.mat', 'neur_prob_beh_prob_perm_dist', 'neur_prob_beh_rew_perm_dist', ...
        'neur_rew_beh_rew_perm_dist', 'neur_rew_beh_prob_perm_dist', 'neur_safe_beh_rew_perm_dist', ...
        'neur_safe_beh_prob_perm_dist', 'neur_prob_beh_BIS_perm_dist')
end

if save_all_figs
    FolderName = fullfile(study_folder, 'Reactivation_Figs');
    mkdir(FolderName);
    FigList = findobj(allchild(0), 'flat', 'Type', 'figure');
    for iFig = 1:length(FigList)
      FigHandle = FigList(iFig);
      FigName   = get(FigHandle, 'Name');
      savefig(FigHandle, fullfile(FolderName, ['fig_', num2str(iFig) '.fig']));
    end
end
    
%% Functions 

% run between subject regression
function t_stat_map = run_btw_subj_regression(neural_betas, behavioral_betas)
    warning('off', 'all')

    NTr = size(neural_betas,2);
    NTe = size(neural_betas,3);
    t_stat_map = zeros(NTr,NTe);
    for train_idx = 1:NTr
        for test_idx = 1:NTe
            [beta, stats1] = robustfit(zscore(behavioral_betas), zscore(squeeze(neural_betas(:,train_idx, test_idx))));
            t_stat_map(train_idx,test_idx) = stats1.t(2);
        end
    end
end

% run btw subj permutation
function permutation_vals = run_btw_subj_permutation(neural_betas, behavioral_betas, NPerm, use_min)

    if nargin < 4
        use_min = false;
    end
    
    permutation_vals = zeros(NPerm,1);
    NBetas = length(behavioral_betas);
    parfor perm_idx = 1:NPerm
        if rem(perm_idx,5) == 0
            display(['Permutation Number: ', num2str(perm_idx)])
        end
        behavioral_betas_perm = behavioral_betas(randperm(NBetas));
        t_stat_map_perm = run_btw_subj_regression(neural_betas, behavioral_betas_perm);
        smooth_t_stat_map_perm = smooth_map(1.5,t_stat_map_perm,0);
        if use_min
            permutation_vals(perm_idx) = min(min(smooth_t_stat_map_perm));
        else
            permutation_vals(perm_idx) = max(max(smooth_t_stat_map_perm));
        end
    end
    
end

% smooth t statistic map
function smoothed_map = smooth_map(smooth_mm, this_map, padding)
    if smooth_mm <= 0
        smoothed_map = this_map;
    else
        if nargin > 2
            smoothed_map = imgaussfilt(this_map,smooth_mm, 'Padding', padding);
        else
            smoothed_map = imgaussfilt(this_map,smooth_mm);
        end
    end
end

% get max train time, test time for a matrix
function [max_val, train_time, test_time] = get_max_train_test(this_mtx, delay_vals_train, delay_vals_test)
    [max_val, max_idx] = max(this_mtx(:));
    [train_idx, test_idx] = ind2sub(size(this_mtx), max_idx); train_time = delay_vals_train(train_idx); test_time = delay_vals_test(test_idx);
end


% get max train time, test time for a matrix
function [min_val, train_time, test_time] = get_min_train_test(this_mtx, delay_vals_train, delay_vals_test)
    [min_val, min_idx] = min(this_mtx(:));
    [train_idx, test_idx] = ind2sub(size(this_mtx), min_idx); train_time = delay_vals_train(train_idx); test_time = delay_vals_test(test_idx);
end

% make raw beta plot
function make_raw_plot(neural_betas,behavioral_betas,delay_vals_train, delay_vals_test,train_time,test_time)

    hold on
    
    peak_neural_betas = squeeze(neural_betas(:,delay_vals_train == train_time, delay_vals_test == test_time));
    [beta, stats1] = robustfit(behavioral_betas, peak_neural_betas);
    x = min(behavioral_betas):.1:max(behavioral_betas);
    plot(x, beta(1) + beta(2)*x, 'k-', 'linewidth', 1.5)
    plot(behavioral_betas, peak_neural_betas, 'ko')%,  'linewidth', 2)
    xlabel('Behavioral Beta')
    ylabel({'Neural Beta', ['(Train: ' , num2str(train_time), ' ms, Test: ', num2str(test_time), ' ms)']})
    axis square
end

