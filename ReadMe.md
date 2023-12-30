
Code is divided into MEG analysis and Behavioral Analysis.


#### Behavioral Analysis

This was done using Julia version 1.5.2

Model fitting was done with an em procedure. EM code (contained in folder em) was written by Nathaniel Daw. Functions that interface with this code to fit models are in the file model_fitting_functions.jl.

Likelihood functions for each model are defined in likfuns_sample (for the sample models) and lik_funs AH_Pros (for additive heuristic and prospect theory). These models are fit in the functions fit_AH_and_Pros.jl and fit_sample_models.jl.

The response time data from the priming/recognition task is analyzed in the notebook Priming_Task_Analysis.ipynb

#### MEG analysis

These scripts go from preprocessed data to key reactivation figures.

Scripts were run in Matlab, version 2022b. Note that both epoching scripts require SPM12.

In order, they should be run:

Epoch_Train_Data.m: Epochs preprocessed localizer data from -250 ms to 650 ms around each image onset, places in Epoched_Data/Epoched_Train_Data.
Also identifies good MEG channels and saves this in Epoched_Data/Good_channels

Epoch_Task_Choice_Data.m: Epochs preprocessed task data from 0 ms to 500 ms following each probability stimulus onset, places in Epoched_Data/Epoched_Task_Choice_Data.

Train_Classifiers_Apply_To_Task: Trains classifier for each outcome for each traintimepoint (0 - 500 ms) and aplies each classifier to epoched task choice data.
Reactivations from task are saved in Classifier_Activations/Task_Reactivations.

Task_Reactivation_Analysis.m: Analyzes classifier reactivaitons, generates all reactivation figures in manuscript. 

Note that each of these can be run without pre-running the others, because intermediate steps have been saved.

To examine classifer performance and choose an L1 penalty the following scripts should be run:

Train_Classifiers_Apply_To_Heldout_Loc.m: Trains classifier for each outcome for multiple timepoints and applied to heldout localizer data. 
Activaoitns stores in Classifier_Activations/Loc_Reactivations.

Classifier_Analysis.m: Plots classifier accuracy for diff. L1 penalties and diff. timepoints.

Preprocessing was run on FIL clusters but for reference, the subject preprocessing script is also included: Preprocess_Subj.m. 
This script uses OSL and needs to be run on a Mac or Linux.

The script folder should be in a folder 'MEG_Decision_Study', which should also include the following:

All_Event_Info_Tables: Includes the time-point and information around each event from each run, with a separate file for each participant.

Preprocessed_Data: Includes for each subject, for each run, 'Adffspmeeg' spm file, which is preprocessed data (see preprocessing script for steps involved).

Epoched_Data (created in epoching scripts, but also saved):

Includes Localizer data, Epoched_Train_Data: Data epoched (-250 ms to 650 ms) around each stimulus onset in localizer task. 
and Task Data: Epoched_Task_Choice_Data: Data epoched 0 ms to 500 ms around each probability stimulus onset in decisoin making task.
Good_channels: list of good channels for each participant (created in epoch train data script).

Classifier_Activations: Created in trainclassifier scripts, but also saved.
Activations of trained classifiers on held-out localizer data (Loc_Reactivations) for a variety of settings
and also on Task_Choice_Epoched_Data, at best performing L1.

Behavioral_Params: Each participants behavioral parameters from Additive Heuristic Model, and also BIS score.


