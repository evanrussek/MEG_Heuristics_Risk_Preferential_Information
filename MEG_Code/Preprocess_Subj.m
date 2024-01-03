function preprocess_subj(s_num, study_folder)

subj_struct_fold = [study_folder,'/last_events/Subj_', num2str(s_num)];
load(fullfile(subj_struct_fold,'last_event_times.mat'))

is.highpass=0.5;

const_pad = 3000; % this is 3 seconds that we add to the end of the block...;
spL = fullfile(study_folder,['Preprocessed_Data/Subj_', num2str(s_num),'/run_']);
spN = fullfile(study_folder,['/MEG_data/Subj_', num2str(s_num)]);

if s_num == 1
    run_vec = 1:15;
else
    run_vec = 1:13;
end

% just start with the first block
for run_num_idx = 1:length(run_vec)
    
    run_num = run_vec(run_num_idx);
    inSt = num2str(run_num, '%02d');

    % make a folder for the new data
    localPath = [spL, inSt '.ds'];
    mkdir(localPath);

    % raw data folder
    ds_str = dir([spN, '/*', inSt, '.ds']);
    ds_folder = [spN, '/', ds_str.name];


    %% Import - CTF resting state using OPT
    S = struct;
    S.fif_file = ds_folder;
    S.spm_file = fullfile(localPath,'spmeeg.mat');
    S.other_channels = {'UADC001','UADC003','UADC005'}; %% check whether these are correct

    D = osl_convert_script(S);
    D = D.chantype(find(strcmp(D.chanlabels,'UADC001')),'EOG1');
    D = D.chantype(find(strcmp(D.chanlabels,'UADC005')),'EOG2');
    D = D.chantype(find(strcmp(D.chanlabels,'UADC003')),'EOG3'); 

    D.save()   

    S = struct;
    S.D = fullfile(localPath,'spmeeg.mat');
    S.prefix='';            
    event = ft_read_event(ds_folder);
    sample = last_event_idx(run_num) + const_pad; 
    S.timewin = [0,round(sample(end)/1200*1000)];

    D=spm_eeg_crop_YL(S);
    D.save()

    %% Phase 1 - Filter
    opt=[];
    opt.maxfilter.do=0;

    spm_files{1}=fullfile(localPath,'spmeeg.mat'); 
    structural_files{1}=[]; % leave empty if no .nii structural file available

    opt.spm_files=spm_files;
    opt.datatype='ctf';

    % HIGHPASS
    if is.highpass ==0
        opt.highpass.do=0;
        opt.dirname=fullfile(localPath,['highpass_',num2str(is.highpass)]);
    else
        opt.highpass.cutoff = is.highpass; % create different folder for different frenquency band (0.5; 1; 20)
        opt.dirname=fullfile(localPath,['highpass_',num2str(opt.highpass.cutoff)]);
        opt.highpass.do=1;
    end

    % Notch filter settings
    opt.mains.do=1;

    % DOWNSAMPLING
    opt.downsample.do=0;

    % IDENTIFYING BAD SEGMENTS 
    opt.bad_segments.do=0;

    % Set to 0 for now
    opt.africa.do=0;
    opt.epoch.do=0;
    opt.outliers.do=0;
    opt.coreg.do=0;

    %%%%%%%%%%%%%%%%%%%%%
    opt = osl_run_opt(opt);


    %% Phase 2 - downsample + bad segment            
    opt2=[];
    opt2.maxfilter.do=0;

    opt2.spm_files = opt.results.spm_files;
    opt2.datatype='ctf';

    % optional inputs
    opt2.dirname=opt.dirname; % directory opt settings and results will be stored    
    opt2.convert.spm_files_basenames = opt.results.spm_files_basenames;

    % DOWNSAMPLING
    opt2.downsample.do=1;
    opt2.downsample.freq=100;

    % IDENTIFYING BAD SEGMENTS 
    opt2.bad_segments.do=1;

    % Set to 0 for now
    opt2.africa.do=0;
    opt2.epoch.do=0;
    opt2.outliers.do=0;
    opt2.coreg.do=0;
    opt2.highpass.do=0;
    opt2.mains.do=0;
    %%%%%%%%%%%%%%%%%%%%%
    opt2 = osl_run_opt(opt2); 

    %% phase 3 - ICA 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    opt3 = [];
    % required inputs
    opt3.spm_files = opt2.results.spm_files;
    opt3.datatype='ctf';

    % optional inputs
    opt3.dirname=opt2.dirname; % directory opt settings and results will be stored

    % africa settings
    opt3.africa.do=1;
    opt3.africa.ident.artefact_chans    = {'EOG1','EOG2','EOG3'};
    opt3.africa.precompute_topos = false;
    opt3.africa.ident.mains_kurt_thresh = 0.5;
    opt3.africa.ident.do_kurt = true; 
    opt3.africa.ident.do_cardiac = true;
    opt3.africa.ident.do_plots = true;
    opt3.africa.ident.do_mains = false;

    % turn the remaining options off
    opt3.maxfilter.do=0;
    opt3.convert.spm_files_basenames = opt2.results.spm_files_basenames;
    opt3.downsample.do=0;
    opt3.highpass.do=0;
    opt3.bad_segments.do=0;
    opt3.epoch.do=0;
    opt3.outliers.do=0;
    opt3.coreg.do=0;

    opt3 = osl_run_opt(opt3);

    %% Phase 4 - Epoch + outliter + coreg
    opt4 = [];
    % required inputs
    opt4.spm_files = opt3.results.spm_files;
    opt4.datatype='ctf';

    % optional inputs
    opt4.dirname=opt3.dirname; %

    opt4.epoch.do=0;
    opt4.outliers.do=0;
    opt4.bad_segments.do=0; 

    %% coreg for subsequent source analysis - Already Done!
    opt4.coreg.do=1;
    opt4.coreg.use_rhino=0; 
    opt4.coreg.useheadshape=0;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%           
    % turn the remaining options off
    opt4.maxfilter.do=0;
    opt4.convert.spm_files_basenames = opt3.results.spm_files_basenames;
    opt4.downsample.do=0;
    opt4.africa.todo.ica=0;
    opt4.africa.todo.ident=0;
    opt4.africa.todo.remove=0;

    opt4=osl_run_opt(opt4);

    %% Display Results
    opt4 = osl_load_opt(opt4.dirname);
    close all;
    fclose('all'); 
end
