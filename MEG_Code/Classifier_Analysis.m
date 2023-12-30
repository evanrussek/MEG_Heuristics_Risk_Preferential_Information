% Analyze classifier activaitons on heldout localizer data - this is used
% to pick a l1 penalty and make some classifier performance plots.

% Enter path of MEG_Decision_Study folder
study_folder =  'D:\MEG_Decision_Study';
study_folder = '/Users/erussek/Dropbox/MEG_Decision_Study_WC';

% Path to localizer activations
group_struc_folder = fullfile(study_folder, 'Classifier_Activations', 'Loc_Reactivations');

% Parameters for the classifier
p_l1_list = .001:.001:.01;
delay_vals_train = 0:10:500; 
delay_vals_test = 0:10:500;
prop_null = 1;
all_im_idx = 1:7;

% Subjects to include
subj_list = 1:21;

% Number of subjects, train , test timepoints, penalties
NS = length(subj_list);
NTr = length(delay_vals_train);
NTe = length(delay_vals_test);
NPen = length(p_l1_list);

%% Load saved classifiers
group_prob = zeros(NPen, NS, NTr, NTe);
group_class = zeros(NPen, NS, NTr, NTe);
group_outcome = zeros(NPen, NS, NTr, NTe);
group_prob_outcome = zeros(NPen, NS, NTr, NTe);
group_prob_other_mtx = zeros(NPen, NS, NTr, NTe);
group_s1_prob_s1_incorr = zeros(NPen, NS, NTr, NTe);
group_s1_prob_s1_corr = zeros(NPen, NS, NTr, NTe);

for pen_idx = 1:NPen
    penalty_l1 = p_l1_list(pen_idx);
    file_name = ['Test_Train_Out_vs_All_L1_', num2str(penalty_l1),'_PN_',num2str(prop_null),'.mat'];
    
    for s_idx = 1:NS
        s_num = subj_list(s_idx);
        subj_struc_folder = fullfile(group_struc_folder, ['subj_', num2str(s_num)]);
        s_res = load(fullfile(subj_struc_folder,file_name));
        
        pred_inv_log_mtx = s_res.pred_inv_log_mtx;
        pred_prob_mtx = 1 ./ (1 + exp(-pred_inv_log_mtx));
        [max_val, pred_class_mtx] = max(pred_inv_log_mtx, [], 4); % predicted class out of 7
        [max_val, pred_class_outcome_mtx] =  max(pred_inv_log_mtx(:,:,:,1:3), [], 4); % predicted class out of 3 outcomes...           
   
        correct_inv_log = zeros(size(pred_inv_log_mtx,1), size(pred_inv_log_mtx,2), size(pred_inv_log_mtx,3));
        correct_prob = zeros(size(pred_inv_log_mtx,1), size(pred_inv_log_mtx,2), size(pred_inv_log_mtx,3));
        other_prob = zeros(size(pred_inv_log_mtx,1), size(pred_inv_log_mtx,2), size(pred_inv_log_mtx,3));
        correct_class = zeros(size(pred_inv_log_mtx,1), size(pred_inv_log_mtx,2), size(pred_inv_log_mtx,3));
        correct_outcome = zeros(size(pred_inv_log_mtx,1), size(pred_inv_log_mtx,2), size(pred_inv_log_mtx,3));
 
        for t_idx = 1:length(s_res.pred_labels)
            correct_inv_log(t_idx,:, :) = pred_inv_log_mtx(t_idx, :, :,  s_res.pred_labels(t_idx));
            correct_prob(t_idx,:, :) = pred_prob_mtx(t_idx, :, :,  s_res.pred_labels(t_idx));
            other_im_both = all_im_idx(s_res.pred_labels(t_idx) ~= all_im_idx);
            first_other_im = other_im_both(2);
            other_prob(t_idx,:,:) = mean(pred_prob_mtx(t_idx,:,:,other_im_both),4);
          %  other_prob(t_idx,:,:) = mean(pred_prob_mtx(t_idx,:,:,all_im_idx),4);
            correct_class(t_idx,:, :) = pred_class_mtx(t_idx, :, :) == s_res.pred_labels(t_idx);
            correct_outcome(t_idx,:, :) = pred_class_outcome_mtx(t_idx, :, :) == s_res.pred_labels(t_idx);
        end
        
        mn_corr_inv_log_mtx = squeeze(mean(correct_inv_log));
        mn_corr_prob_mtx = squeeze(mean(correct_prob));
        mn_corr_class_mtx = squeeze(mean(correct_class));
        outcome_trials = find(s_res.pred_labels < 4); 
        mn_corr_outcome_class_mtx = squeeze(mean(correct_outcome(outcome_trials,:,:)));
        mn_prob_outcome_mtx = squeeze(mean(correct_prob(outcome_trials,:,:)));
        mn_prob_other_mtx = squeeze(mean(other_prob(outcome_trials,:,:)));
        
     %   group_inv_log(pen_idx,s_idx,:,:) = mn_corr_inv_log_mtx;
        group_prob(pen_idx,s_idx, :, :) = mn_corr_prob_mtx;
        group_class(pen_idx, s_idx,:,:) = mn_corr_class_mtx;
        group_outcome(pen_idx,s_idx,:,:) = mn_corr_outcome_class_mtx;
        group_prob_outcome(pen_idx,s_idx,:,:) = mn_prob_outcome_mtx;
        group_prob_other_mtx(pen_idx,s_idx,:,:) = mn_prob_other_mtx;
        
    end
end

%% Select a penalty -- best one is .002.

mn_class = squeeze(mean(group_outcome, 2)); % NPen x NTr x NTe
sem_class = squeeze(std(group_outcome,1, 2))/sqrt(21); % NPen x NTr x NTe

subj_mn_class = zeros(NPen, NS, NTr);
for p_idx = 1:NPen
    for s_idx = 1:NS
        subj_mn_class(p_idx,s_idx,:) = diag(squeeze(group_outcome(p_idx,s_idx,:,:)));
    end
end

subj_mn_class_s = mean(subj_mn_class,3);
mn_class = mean(subj_mn_class_s,2);
sem_class = std(subj_mn_class_s,[],2)/sqrt(21);

max_val = max(max(mn_class,[],3),[],2);

diag_class_vals = zeros(NPen, NTr);
diag_class_sem = zeros(NPen, NTr);
for i = 1:NPen
    diag_class_vals(i,:) = diag(squeeze(mn_class(i,:,:)));
    diag_class_sem(i,:) = diag(squeeze(sem_class(i,:,:)));
end

close(figure(1))
figure(1)
hold on
plot(p_l1_list,mn_class, '-o')
%errorbar(p_l1_list, mn_class, sem_class)
xlabel('L1 Regularization Parameter')
ylabel('Cross validation accuracy (mean of diagonal)')
axis square


figure(2)
%% Plot Diagonal of Accuracy Matrix
group_mtx = squeeze(group_outcome(2,:,:,:));
group_diag_corr = zeros(size(group_mtx,1), size(group_mtx,2));
for s_idx = 1:NS
    group_diag_corr(s_idx,:) = diag(squeeze(group_mtx(s_idx,:,:)));
end
close(figure(2))
figure(2)
hold on
shadedErrorBar(delay_vals_test, mean(group_diag_corr), std(group_diag_corr)./sqrt(NS)) %,'lineprops', {'color', 'gray'})
yline(.3594, 'k--'); % permutation thresh. calculated in other script
axis square
axis([10, 500, .3, .65])
ylabel('Cross-Validation Accuracy')
xlabel('Timepoint (ms) Rel. Out. Stim. Onset')

shadedErrorBar(delay_vals_test, mean(this_subj_mn_class), std(this_subj_mn_class)./sqrt(NS)) %,'lineprops', {'color', 'gray'})

%% Plot Diagonal of Accuracy Matrix
group_mtx = squeeze(group_outcome(2,:,:,:));
group_diag_corr = zeros(size(group_mtx,1), size(group_mtx,2));
for s_idx = 1:NS
    group_diag_corr(s_idx,:) = diag(squeeze(group_mtx(s_idx,:,:)));
end
close(figure(2))
figure(2)
hold on
shadedErrorBar(delay_vals_test, mean(group_diag_corr), std(group_diag_corr)./sqrt(NS),'lineprops', {'color', 'red'})
%yline(.3594, 'k--'); % permutation thresh. calculated in other script
axis square
axis([10, 500, .3, .65])
ylabel('Cross-Validation Accuracy')
xlabel('Timepoint (ms) Rel. Out. Stim. Onset')

shadedErrorBar(delay_vals_test, mean(this_subj_mn_class), std(this_subj_mn_class)./sqrt(NS),'lineprops', {'color', 'blue'})
legend({'Without Trial Removal','With Trial Removal'})

