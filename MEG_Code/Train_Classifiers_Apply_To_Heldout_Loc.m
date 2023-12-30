% Train Classifier for Each Outcome on Multiple Time-points
% and Apply to Epoched Heldout Localizer Data

% Enter path of MEG_Decision_Study folder
study_folder =  'D:\MEG_Decision_Study';

% add path to util folder
util_folder = fullfile(study_folder, 'MEG_analysis_scripts', 'utils');
addpath(genpath(util_folder));

% Folder to save reactivations
activation_folder = fullfile(study_folder, 'Classifier_Activations_Revised');
mkdir(activation_folder);
group_struc_folder = fullfile(activation_folder, 'Loc_Reactivations');
mkdir(group_struc_folder);

% Parameters for the classifier
p_l1_list = .002%.005:.001:.01;%.001:.001:.004; % L1 penalty
p_l2_list = 0;% L2 penalty
delay_vals_train = -350:10:850; % timepoints on which to train classifier (millisecond, relative to iamge onset)
delay_vals_test = -350:10:850; % tiempoints on which to test each classifier (millisecond, relative to image onset, on hold out data)
n_tp_embed = 0; % how many timepoints to use for feature data (0 means just the timepoint delay_vals_train is on, 1 means 10 ms on either side of this, 2 means 20 ms on either isde of this)
penalty_l2 = 0;

% Subjects to include
subj_list = 1:21;

% Number of holdout rounds
n_holdout = 5; 
n_examples = 60; % 
hold_out_idxs = reshape(1:n_examples, n_examples/n_holdout, n_holdout)';

prop_null = 1;
pn_idx = 1;



for pen_idx = 1:length(p_l1_list)% loop through penalty onset
    penalty_l1 = p_l1_list(pen_idx);
    file_name = ['Test_Train_Out_vs_All_L1_', num2str(penalty_l1),'_PN_',num2str(prop_null),'.mat'];

    parfor(s_idx = 1:length(subj_list),4) % loop through subjects (be careful w/ parfor = can use too much memory)
        s_num = subj_list(s_idx);
        fprintf(['Starting Subject ', num2str(s_num), '\n'])

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% specify event types and number of images
        n_images = 7;
        all_im_idx = 1:7;
        image_types = {'OUTCOME', 'OUTCOME', 'OUTCOME', 'CHOICE', 'CHOICE', 'CHOICE', 'CHOICE'};
        image_numbers = [1 2 3 1 2 3 4];
        state_names = {'O1', 'O2', 'OS', 'C1', 'C2', 'C3', 'C4'};

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% load time file and epoched train data and epoched task data%%%%
        time_file = fullfile(study_folder, 'All_Event_Info_Tables', ['Subj_',num2str(s_num),'All_Event_Time_Table.mat']);
        a = load(time_file, 'subj_table'); % load  subj_table which has information on what image is presented for each trial
        ed = load(fullfile(study_folder, 'Epoched_Data_Revised', 'Epoched_Train_Data', ['Subj_', num2str(s_num), '_Epoched_Train_Data.mat']),  'run_ep_data', 'time_points_events_train');

        loc_table = a.subj_table(strcmp(a.subj_table.phase, 'LOC') & a.subj_table.block_number < 4,:); % localizer data is first 3 blocks
        loc_data = cat(1, ed.run_ep_data{:}); % concatenate all hte localizer data
        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%% Assemble all data for the classifier %%%%%%%%%%%%%%%%%%%
        image_task_rows_loc = cell(n_images,1);
        image_task_rows_task = cell(n_images,1);

        %%%% SPLIT data up into different sets for cross validation
        image_task_rows_loc_train = cell(5,7); 
        image_task_rows_loc_test = cell(5,7);
        image_task_loc_test_last_label = cell(5,7);

        all_labels = zeros(size(loc_table.image_number));
        for im_idx = all_im_idx
             these_rows = find(loc_table.image_number == image_numbers(im_idx) & strcmp(loc_table.image_type, image_types{im_idx})); % 60 images per...
             all_labels(these_rows) = im_idx;
        end

        last_label = [0; all_labels(1:end-1)];

        for im_idx = all_im_idx
            % get rows in data of this image
            these_rows = find(loc_table.image_number == image_numbers(im_idx) & strcmp(loc_table.image_type, image_types{im_idx})); % 60 images per...
            all_labels(these_rows) = im_idx;
            % randomly permute these rows
            these_rows_perm = these_rows(randperm(length(these_rows)));
            for cv_idx = 1:5 % 5 fold cross validation - want even number of images in each row
                these_test_rows = 12*(cv_idx - 1) + 1: cv_idx*12;
                all_rows = 1:60;
                these_train_rows = all_rows(~ismember(1:60, these_test_rows));
                % select rows for this image for this train versus test data for
                % this cross validation index
                image_task_rows_loc_train{cv_idx ,im_idx} = these_rows_perm(these_train_rows );%these_rows(randperm(length(these_rows)));
                image_task_rows_loc_test{cv_idx, im_idx} = these_rows_perm(these_test_rows);
                image_task_loc_test_last_label{cv_idx, im_idx} = last_label(these_rows_perm(these_test_rows)); 
            end
        end
        %

        % embed the data - 
        % train data all emed is ntrials x n_features x n_train_timepoints -
        % n_features might have multiple timepoints of sensor data.
        train_data_all_embed = zeros(size(loc_data,1), (2*n_tp_embed + 1)*size(loc_data,2), length(delay_vals_test));
        for train_tp_idx = 1:length(delay_vals_test)
            this_time_point = delay_vals_test(train_tp_idx);
            this_idx_ms = find(ed.time_points_events_train == this_time_point);
            this_embed_data = loc_data(:,:, this_idx_ms - n_tp_embed : this_idx_ms + n_tp_embed);
            this_embed_data_stack = reshape(this_embed_data,size(this_embed_data,1),[]);
            train_data_all_embed(:,:,train_tp_idx) = this_embed_data_stack; % think this works
        end
        % this is now in the order of trials as they occur

        % what to store results with? 
        n_class = 7;
        class_applied_loc_data = zeros(size(loc_data,1), size(loc_data,3), n_class);

        % now start the cv idx
        pred_inv_log_epoch_all = zeros(420, length(delay_vals_train), length(delay_vals_test), 7);



        for train_tp_idx = 1:length(delay_vals_train)
            display(['Subj: ',num2str(s_num), 'train_idx: ', num2str(train_tp_idx), 'pen_idx: ', num2str(pen_idx)])

            pred_act_epoch = [];
            pred_reg_epoch = [];
            pred_inv_log_epoch = [];
            pred_labels = [];
            last_labels = [];
            
            for cv_idx = 1:5 % 5 fold cross validation... 
               % display(['starting cv idx: ', num2str(cv_idx)])
                image_loc_data = cell(n_images,1);
                image_loc_labels = cell(n_images,1);
                image_null_data = cell(n_images,1);
                for im_idx = all_im_idx
                    image_loc_data{im_idx} = train_data_all_embed(image_task_rows_loc_train{cv_idx,im_idx},:,:);
                    image_loc_labels{im_idx} = im_idx*ones(size(image_task_rows_loc_train{cv_idx,im_idx}));
                    image_null_data{im_idx} = loc_data(image_task_rows_loc_train{cv_idx,im_idx},:, ed.time_points_events_train == -200);
                end
                train_data_all = cat(1,image_loc_data{:});
                train_labels_all = cat(1,image_loc_labels{:});

                % balance the null data
                null_data_all = [];
                for ndidx = 1:7
                    null_data_all = [null_data_all; image_null_data{ndidx}(1:(48*prop_null(pn_idx)),:)];
                end
                % 60 examples per image... so let's do 6 way cross val

                % start loop over train time points
                this_time_point = delay_vals_train(train_tp_idx);
                this_train_idx = find(delay_vals_test == this_time_point); % 170 -- 23

                this_prop_null = prop_null(pn_idx);
                train_data = squeeze(train_data_all(:,:,this_train_idx));
                tran_data_wnull = [train_data; null_data_all];

                tran_data_scale_wnull = scaleFunc(tran_data_wnull); % scale the training data

                labels_wnull = [train_labels_all; (n_images+1)*ones(size(null_data_all,1),1)];

                all_im_betas = zeros(n_images, size(train_data,2));
                all_im_betas_z = zeros(n_images, size(train_data,2));
                all_im_int = zeros(n_images,1);

                % train outcome classifiers
                for im_idx = 1:3
                    this_im_labels = labels_wnull == im_idx;
                    l1p = penalty_l1; l2p = 0; % just the l1_p... 

                    [beta, fitInfo] = lassoglm(tran_data_scale_wnull, this_im_labels, 'binomial', ...
                        'Alpha', l1p / (2*l2p+l1p), 'Lambda', 2*l2p + l1p, 'Standardize', false);

                    intercept = fitInfo.Intercept;
                    all_im_betas(im_idx,:) = beta;
                    all_im_int(im_idx) = intercept;
                end

                % apply to heldout data
                for class_idx = 1:7
                    this_im_test_data = train_data_all_embed(image_task_rows_loc_test{cv_idx,class_idx},:,:,:);
                    tdp = permute(this_im_test_data,[1 3 2]);
                    tdp_rs = reshape(tdp, size(tdp,1)*size(tdp,2), size(tdp,3)); % (n_epochs x n_timepoints) x n_sensors
                    scaled_tdp_rs = scaleFunc(tdp_rs);

                    % compute multiple measures of projecting data onto beta weights
                    pred_act = scaled_tdp_rs*all_im_betas'; % check that scale func is being applied correctly    
                    pred_reg = scaled_tdp_rs*pinv([ones(size(all_im_betas,2),1) all_im_betas'])';
                    pred_reg_epoch_im = zeros(size(tdp,1), size(tdp,2), n_images);
                    pred_act_epoch_im = zeros(size(tdp,1), size(tdp,2), n_images);
                    pred_inv_log_epoch_im = zeros(size(tdp,1), size(tdp,2), n_images);

                    for im_idx = 1:n_images
                        % n_trials x n_timepoints x n_images
                        pred_act_epoch_im(:,:,im_idx) = reshape(pred_act(:,im_idx), size(tdp,1), size(tdp,2)); % n_epoch x n_timepoints
                        pred_inv_log_epoch_im(:,:,im_idx) = all_im_int(im_idx) + pred_act_epoch_im(:,:,im_idx);
                    end

                    pred_inv_log_epoch = [pred_inv_log_epoch; pred_inv_log_epoch_im];
                    pred_labels = [pred_labels; class_idx*ones(size(pred_reg_epoch_im,1),1)];
                    last_labels = [last_labels; image_task_loc_test_last_label{cv_idx, class_idx}];
                end
            end % end CV loop
            pred_inv_log_epoch_all(:,train_tp_idx,:,:) = pred_inv_log_epoch;
        end % end train time-point loop
        % make a folder to save the results..


        
        subj_struc_folder = fullfile(group_struc_folder, ['Subj_', num2str(s_num)]);
        if ~exist(subj_struc_folder, 'dir')
            mkdir(subj_struc_folder);
        end 

        parsave(fullfile(subj_struc_folder, file_name), delay_vals_test, delay_vals_train, pred_inv_log_epoch_all, pred_labels);
    end
end

%%%%%%%%% save function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function parsave(fname, delay_vals_test, delay_vals_train, pred_inv_log_mtx, pred_labels)
  save(fname,  'delay_vals_train', 'delay_vals_test','pred_inv_log_mtx', 'pred_labels')
end

