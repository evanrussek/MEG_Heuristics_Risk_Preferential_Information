% Epoch each subject's preprocessed task data around each presentation of
% probability stimulus.

% Enter path to SPM 12 folder
SPM_folder = 'D:\spm12';
% Enter Path to the MEG_Decision_Study folder
study_folder = 'D:\MEG_Decision_Study';

%% Set folders to save things
mkdir(fullfile(study_folder, 'Epoched_Data'));
to_save_folder = fullfile(study_folder, 'Epoched_Data', 'Epoched_Task_Choice_Data');
mkdir(to_save_folder)

%% add path to SPM and utils
addpath(genpath(SPM_folder))
util_folder = fullfile(study_folder, 'MEG_analysis_scripts', 'utils');
addpath(genpath(util_folder))

%% which subjects to include
subj_list = 1:21;
NS = length(subj_list);

% what time-points around each prob. stim. onset should be epoched (in ms)
task_time_points_events = 0:10:500;
time_points_events = task_time_points_events;

% load the folder for this subject...
for s_idx = 1:NS
    s_num = subj_list(s_idx);
    display(['Epoching Task Choice Data. Subj: ', num2str(s_num)])

    % load subj's event timing information
    time_file = fullfile(study_folder, 'All_Event_Info_Tables', ['Subj_',num2str(s_num),'All_Event_Time_Table.mat']);
    temp = load(time_file, 'subj_table');
    subj_table = temp.subj_table;
    
    % load list of good channels (made in epoch train data).
    load(fullfile(study_folder, 'Epoched_Data', 'Good_Channels', ['Subj_', num2str(s_num), '_Good_Channels.mat']), 'GoodChannel');
    
    % cell to store epoched task data
    task_ep_data = cell(8,1);
    
    block_nums = 8:15;
    task_runs = 6:13;
    if s_num == 1 % subject 1 runs coded differently
        task_runs = 8:15;
    end

    % loop through each task run
    for r_idx = 1:length(block_nums)
        
        % load data from this run and isolate good channels
        run_num = task_runs(r_idx);
        inSt = num2str(run_num, '%02d');
        pp_data_folder = fullfile(study_folder,'Preprocessed_Data'); 
        pp_run_folder = fullfile(pp_data_folder, ['Subj_', num2str(s_num)], ['run_' num2str(run_num,'%02d')]);
        pp_run_mat = fullfile(pp_run_folder, 'Adffspmeeg.mat');
        DLCI=spm_eeg_load(pp_run_mat);
        chan_MEG = indchantype(DLCI,'meeg'); % no allconds
        good_channel_all_runs = find(sum(GoodChannel') == size(GoodChannel,2));
        good_channels = intersect(good_channel_all_runs,chan_MEG);
        Cleandata = DLCI(good_channels,:,:); % 276 x 40174    
        good_meg = Cleandata;

        % get onsets of each probability stimulus event (coded 'Choice')
        this_run_table = subj_table(subj_table.scanner_run_number == run_num & contains(subj_table.event, 'CHOICE'),:);
        onset_idxs = this_run_table.onset_idx_ds;

        % structure to store run epoched data
        n_trials = size(onset_idxs,1);
        n_sensors = size(good_meg,1);
        n_tp = length(time_points_events);
        ep_data = zeros(n_trials, n_sensors,n_tp);

        % idices around each event to store (each indice is 10 ms).
        rel_idxs = time_points_events/10;

        % epoch data from each trial individually
        for i = 1:n_trials
            this_onset = onset_idxs(i);
            these_idxs = this_onset + rel_idxs;
            good_idxs = (these_idxs > 0 & these_idxs < size(good_meg,2));
            this_meg = zeros(size(good_meg,1),length(these_idxs));
            this_meg(:,good_idxs) = good_meg(:,these_idxs(good_idxs));
            ep_data(i,:,:) = this_meg;
        end

        % remove bad trials
        ep_data_new = zeros(size(ep_data));
        for i = 1:n_trials
            filt_mtx = artToZero(squeeze(ep_data(i,:,:))'); % row is time...
            ep_data_new(i,:,:) = filt_mtx';
        end
        
        % store run in cell
        task_ep_data{r_idx} = ep_data_new;
    end

    % save subject epoched data and also reference time-points.
    time_points_events_task = time_points_events;
    task_data = cat(1, task_ep_data{:});
    save(fullfile(to_save_folder,['Subj_', num2str(s_num), '_Epoched_Task_Choice_Data.mat']), 'task_ep_data', 'time_points_events_task');
end

