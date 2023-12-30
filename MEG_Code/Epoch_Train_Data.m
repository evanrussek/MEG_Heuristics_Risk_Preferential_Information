% Epoch each subject's pre-processed localizer data. Also identify and save
% good channels for each subject.

% Enter path to SPM 12 folder
SPM_folder = 'D:\spm12';
% Enter Path to the MEG_Decision_Study folder
study_folder = 'D:\MEG_Decision_Study';

%% Set folders to save things
% Folder where epoched trained data should be saved
mkdir(fullfile(study_folder, 'Epoched_Data_Revised'));
to_save_folder = fullfile(study_folder, 'Epoched_Data', 'Epoched_Train_Data');
mkdir(to_save_folder);
% Where should list of good channels be saved
to_save_good_channel_folder = fullfile(study_folder, 'Epoched_Data', 'Good_Channels');
mkdir(to_save_good_channel_folder);

%% add path to SPM and utils
addpath(genpath(SPM_folder))
util_folder = fullfile(study_folder, 'MEG_analysis_scripts', 'utils');
addpath(genpath(util_folder))

%% which subjects to apply to
subj_list = 1:21;
NS = length(subj_list);

% what time-points around each event should be epoched (in ms)
time_points_events = -250:10:600;

for s_idx = 1:NS
    s_num = subj_list(s_idx);
    
    display(['Epoching Localizer Data. Subj: ', num2str(s_num)])
    time_file = fullfile(study_folder, 'All_Event_Info_Tables', ['Subj_',num2str(s_num),'All_Event_Time_Table.mat']);
    temp = load(time_file, 'subj_table');
    subj_table = temp.subj_table;
    
    % subj 1 did 5 rounds of localizer - others did 3.
    if (s_num == 1); n_runs = 15; else; n_runs = 13; end

    % matrix to store which of 276 channels are good
    GoodChannel = nan(276,n_runs);
    
    % figure out good channels separately for each run
    for run_num = 1:n_runs
        
        % load Preprocessed data for this run.
        inSt = num2str(run_num, '%02d');
        pp_data_folder = fullfile(study_folder,'Preprocessed_Data');
        pp_run_folder = fullfile(pp_data_folder, ['Subj_', num2str(s_num)],...
        ['run_' num2str(run_num,'%02d')]);
        pp_run_mat = fullfile(pp_run_folder, 'Adffspmeeg.mat');
        run_meg = spm_eeg_load(pp_run_mat);
        
        % record which channels are good for this run
        ch_lbls = chanlabels(run_meg);
        chan_good_idxs = indchantype(run_meg,'meeg','GOOD');
        chan_ALL  = indchantype(run_meg,'meeg');
        [~,chan_bad_idxs]=setdiff(chan_ALL,chan_good_idxs);
        index = ones(1,276); % is this correct?
        index(chan_bad_idxs) = 0;
        GoodChannel(:,run_num)=index;
    end
    
    % save good channel for this subject
    subj_good_channel_file = fullfile(to_save_good_channel_folder, ['Subj_', num2str(s_num), '_Good_Channels.mat']);
    save(subj_good_channel_file, 'GoodChannel');

    % go through 3 localizer runs and epoch data for each trial
    run_ep_data = cell(3,1);
    for run_num = 1:3
        
        % load run data and get good channels
        inSt = num2str(run_num, '%02d');
        pp_data_folder = fullfile(study_folder,'Preprocessed_Data');
        pp_run_folder = fullfile(pp_data_folder, ['Subj_', num2str(s_num)],...
            ['run_' num2str(run_num,'%02d')]);
        pp_run_mat = fullfile(pp_run_folder, 'Adffspmeeg.mat');

        DLCI=spm_eeg_load(pp_run_mat);
        chan_MEG = indchantype(DLCI,'meeg'); 
        good_channel_all_runs = find(sum(GoodChannel') == size(GoodChannel,2));
        good_channels = intersect(good_channel_all_runs,chan_MEG);
        Cleandata = DLCI(good_channels,:,:);
        good_labels = ch_lbls(good_channels);
        good_meg = Cleandata;
        
        % get onset of each event for this run
        this_run_table = subj_table(subj_table.scanner_run_number == run_num,:);
        onset_idxs = this_run_table.onset_idx_ds(~isnan(this_run_table.onset_idx_ds));
        
        % check that we have the right number of events
        if length(onset_idxs) ~= 140
            error('wrong number of events')
        end

        % matrix to save data -> ntrial x nsensor x ntime-point
        n_trials = size(onset_idxs,1);
        n_sensors = size(good_meg,1);
        n_tp = length(time_points_events);
        ep_data = zeros(n_trials, n_sensors,n_tp);

        % each idx is 10 ms of time - thus, we'll include these indices
        % around each event (at 0)
        rel_idxs = time_points_events/10;

        % do the epoching
        for i = 1:n_trials
            this_onset = onset_idxs(i);
            these_idxs = this_onset + rel_idxs;
            good_idxs = (these_idxs > 0 & these_idxs < size(good_meg,2));
            this_meg = zeros(size(good_meg,1),length(these_idxs));
            this_meg(:,good_idxs) = good_meg(:,these_idxs(good_idxs));
            if sum(good_idxs) > 0
                ep_data(i,:,:) = this_meg;%good_meg;%good_meg(:,these_idxs);
            end
        end
        
        %  kill bad trials using util function 'artToZero'            
        ep_data_new = zeros(size(ep_data));
        for i = 1:n_trials
            filt_mtx = artToZero(squeeze(ep_data(i,:,:))');
            ep_data_new(i,:,:) = filt_mtx';
        end

        run_ep_data{run_num} = ep_data_new;
    end

    time_points_events_train = time_points_events;
    save(fullfile(to_save_folder, ['Subj_', num2str(s_num), '_Epoched_Train_Data.mat']), 'run_ep_data', 'time_points_events', 'good_labels', 'time_points_events_train');
end
