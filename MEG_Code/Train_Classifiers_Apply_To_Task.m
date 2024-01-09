% Train Classifier for Each Outcome on Multiple Time-points
% and Apply to Epoched Task Choice Data

% Enter path of MEG_Decision_Study folder
study_folder = '/Users/erussek/Dropbox/MEG_Decision_Study_WC';

% add path to util folder
util_folder = fullfile(study_folder, 'MEG_analysis_scripts', 'utils');
addpath(genpath(util_folder));

% make folder to save results
activation_folder = fullfile(study_folder, 'Classifier_Activations_Test');
mkdir(activation_folder);
group_struc_folder = fullfile(activation_folder, 'Task_Reactivations');
mkdir(group_struc_folder);

% parameters for the classifier
penalty_l1 = .002; % L1 lasso penalty
penalty_l2 = 0; % L2 lasso penalty
prop_null = 1; % proportion of null data to include
which_train_images = 1:3; % which images to build classifiers
n_tp_embed = 0; % how many time-points to include aroudn key-timepoint

% what time-points and subjects to include?
delay_vals_train = 0:10:500; % train time points
delay_vals_test = 0:10:500; % test time points
subj_list = 1:21; % subjects to include

NS = length(subj_list);
NTr = length(delay_vals_train);
NTe = length(delay_vals_test);

% file name for saving reactivations
file_name = ['Test_Train_Out_vs_All_L1_', num2str(penalty_l1),'_PN_',num2str(prop_null),'.mat'];

% Begin subject loop 
parfor s_idx = 1:length(subj_list)
    s_num = subj_list(s_idx);
    fprintf(['Starting subject ', num2str(s_num), '\n'])

    % specify event types and number of images
    n_images = 7;
    all_im_idx = 1:7;
    image_types = {'OUTCOME', 'OUTCOME', 'OUTCOME', 'CHOICE', 'CHOICE', 'CHOICE', 'CHOICE'};
    image_numbers = [1 2 3 1 2 3 4];
    state_names = {'O1', 'O2', 'OS', 'C1', 'C2', 'C3', 'C4'};

    % load time file and epoched train and task data
    time_file = fullfile(study_folder, 'All_Event_Info_Tables', ['Subj_',num2str(s_num),'All_Event_Time_Table.mat']);
    a = load(time_file, 'subj_table'); % time table / 
    ed = load(fullfile(study_folder, 'Epoched_Data', 'Epoched_Train_Data', ['Subj_', num2str(s_num), '_Epoched_Train_Data.mat']),  'run_ep_data', 'time_points_events_train');
    ted = load(fullfile(study_folder, 'Epoched_Data', 'Epoched_Task_Choice_Data', ['Subj_', num2str(s_num), '_Epoched_Task_Choice_Data.mat']),  'task_ep_data', 'time_points_events_task');    
    loc_table = a.subj_table(strcmp(a.subj_table.phase, 'LOC') & a.subj_table.block_number < 4,:);
    task_table = a.subj_table(strcmp(a.subj_table.phase, 'TASK') & contains(a.subj_table.event, 'CHOICE'),:);
    loc_data = cat(1, ed.run_ep_data{:});
    task_data = cat(1, ted.task_ep_data{:});
        
    % create structures to store results for this subject
    n_event_rows = size(task_table,1);
    event_pred_act_mtx = zeros(n_event_rows, NTr, NTe, length(which_train_images));

    % assemble all data for the classifier
    image_task_rows_loc = cell(n_images,1);
    image_task_rows_task = cell(n_images,1);
    for im_idx = all_im_idx
        image_task_rows_loc{im_idx} = find(loc_table.image_number == image_numbers(im_idx) & strcmp(loc_table.image_type, image_types{im_idx}));
        image_task_rows_task{im_idx} = find(task_table.image_number == image_numbers(im_idx) & strcmp(task_table.image_type, image_types{im_idx}));
    end

    image_loc_data = cell(n_images,1);
    image_loc_labels = cell(n_images,1);
    image_task_data = cell(n_images,1);
    image_task_labels = cell(n_images,1);
    for im_idx = all_im_idx
        image_loc_data{im_idx} = loc_data(image_task_rows_loc{im_idx},:,:);
        image_loc_labels{im_idx} = im_idx*ones(size(image_task_rows_loc{im_idx}));

        image_task_data{im_idx} = task_data(image_task_rows_task{im_idx},:,:);
        image_task_labels{im_idx} = im_idx*ones(size(image_task_rows_task{im_idx}));
    end

    train_data_all = cat(1,image_loc_data{:});
    train_labels_all = cat(1,image_loc_labels{:});
    
    % embed the data
    train_data_all_embed = zeros(size(train_data_all,1), (2*n_tp_embed + 1)*size(train_data_all,2), NTr);
    for train_tp_idx = 1:length(delay_vals_train)
        this_idx_ms = find(ed.time_points_events_train == delay_vals_train(train_tp_idx));

        this_embed_data = train_data_all(:,:, this_idx_ms - n_tp_embed : this_idx_ms + n_tp_embed);
        this_embed_data_stack = reshape(this_embed_data,size(this_embed_data,1),[]);
        train_data_all_embed(:,:,train_tp_idx) = this_embed_data_stack; % think this works
    end    
    
    % embed the null data
    null_data_all_embed = zeros(size(train_data_all,1), (2*n_tp_embed + 1)*size(train_data_all,2), 1);
    this_idx_ms = find(ed.time_points_events_train == -200);
    this_embed_data = train_data_all(:,:, this_idx_ms - n_tp_embed : this_idx_ms + n_tp_embed);
    this_embed_data_stack = reshape(this_embed_data,size(this_embed_data,1),[]);
    null_data_all_embed(:,:,1) = this_embed_data_stack; % think this works
    null_data_all_embed = null_data_all_embed(randperm(size(null_data_all_embed,1)),:);
    
    
    % start loop over train timepoints
    for train_tp_idx = 1:NTr
        
        display(['Subj: ', num2str(s_num), ' train_idx: ', num2str(train_tp_idx), ' of ', num2str(NTr)]);

        % create train examples and labels
        train_data = squeeze(train_data_all_embed(:,:,train_tp_idx));        
        n_null = round(size(train_data,1)*prop_null); % try add some null data
        
        null_data = null_data_all_embed(1:n_null, :, :); %zeros(n_null, size(train_data,2));
        tran_data_wnull = [train_data; null_data];   
        tran_data_scale_wnull = scaleFunc(tran_data_wnull); % scale the training data
        labels_wnull = [train_labels_all; (n_images+1)*ones(n_null,1)];

        % train and store classifier for each image
        all_im_betas = zeros(n_images, size(train_data,2));
        all_im_betas_z = zeros(n_images, size(train_data,2));
        all_im_int = zeros(n_images,1);
        for im_idx = which_train_images%1:3%n_images % only 7 betas are greater than 0...
            this_im_labels = labels_wnull == im_idx;
            l1p = penalty_l1; l2p = 0; % just the l1_p... 

            [beta, fitInfo] = lassoglm(tran_data_scale_wnull, this_im_labels, 'binomial', ...
                'Alpha', l1p / (2*l2p+l1p), 'Lambda', 2*l2p + l1p, 'Standardize', false);
            intercept = fitInfo.Intercept;
            all_im_betas(im_idx,:) = beta;
            all_im_betas_z(im_idx,:) = zscore(beta);
            all_im_int(im_idx) = intercept;
        end
        
        % apply the classifier to each event's epoch in the task
        event_data = task_data(:,:,find(delay_vals_test(1) == ted.time_points_events_task):find(delay_vals_test(end) == ted.time_points_events_task)); % trials x sensors x timepoints test

        if n_tp_embed > 0 % embed task data
            event_data_embed = zeros(size(event_data,1), (2*n_tp_embed + 1)*size(event_data,2), size(event_data,3));
            for test_idx = 1:size(event_data,3)
                this_embed_data = zeros(size(event_data_embed,1), size(event_data,2), (2*n_tp_embed + 1));
                this_idx = test_idx - n_tp_embed : test_idx + n_tp_embed;
                good_idx = find(this_idx >= 1 & this_idx <= size(event_data,3));
                this_idx = this_idx(good_idx);
                this_embed_data(:,:,good_idx) = event_data(:,:, this_idx);
                this_embed_data_stack = reshape(this_embed_data,size(this_embed_data,1),[]);
                event_data_embed(:,:,test_idx) = this_embed_data_stack;
            end
        else
            event_data_embed = event_data;
        end

        % reshape data so it can be multiplied by beta weights easily
        tdp = permute(event_data_embed,[1 3 2]);
        tdp_rs = reshape(tdp, size(tdp,1)*size(tdp,2), size(tdp,3)); % (n_epochs x n_timepoints) x n_sensors
        scaled_tdp_rs = scaleFunc(tdp_rs);

        % project data onto beta weights
        pred_act = scaled_tdp_rs*all_im_betas'; 

        % reshape
        pred_act_epoch = zeros(size(tdp,1), size(tdp,2), length(which_train_images));
        pred_inv_log_epoch = zeros(size(tdp,1), size(tdp,2), length(which_train_images));
        for im_idx = which_train_images
            % n_trials x n_timepoints x n_images
            pred_act_epoch(:,:,im_idx) = reshape(pred_act(:,im_idx), size(tdp,1), size(tdp,2)); % n_epoch x n_timepoints
            pred_inv_log_epoch(:,:,im_idx) = all_im_int(im_idx) + pred_act_epoch(:,:,im_idx);
        end

        % store classifier activations for event/time-point
        event_pred_act_mtx(:,train_tp_idx, :, :) = pred_inv_log_epoch;
                
    end % end train timepoint loop

    % Save Results
    subj_struc_folder = fullfile(group_struc_folder, ['Subj_', num2str(s_num)]);
    if ~exist(subj_struc_folder, 'dir'); mkdir(subj_struc_folder); end 
    
    predact_mtx = event_pred_act_mtx;
    parsave(fullfile(subj_struc_folder, file_name), delay_vals_train, delay_vals_test, predact_mtx, state_names);
    
end % end subject loop


% save function (needed for parfor)
function parsave(fname,  delay_vals_train, delay_vals_test, predact_mtx, state_names)
  save(fname, 'delay_vals_train', 'delay_vals_test', 'predact_mtx', 'state_names');
end



