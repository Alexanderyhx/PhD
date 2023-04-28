% Load Dataset
%setenv('datafolder', '/Users/yaale244/Desktop/Alexpersonalneucube/basesplit/test') %set environment if testing in matlab
d=getenv('datafolder')
% Extract the last directory name from the path in d
[~, folder_name, ~] = fileparts(d);

ratio=1;
flag=true; %empty flag=[] means recall, true means classification, false means regression

%% Load feature names
    
Names=textread(fullfile(d, 'feature_names_eeg.txt'), '%s');

%% iterate through CS and Ctrl
%Cubeparameters
unsup_params.potential_leak_rate=0.0020;
unsup_params.STDP_rate=0.01;
unsup_params.threshold_of_firing=0.5;
unsup_params.training_round=1;
unsup_params.refactory_time=6;
unsup_params.LDC_probability=0;

%% choose top 5 features

idxtop5 = [14;15;16;23;24] %5 features used for stimulation also used for prediction

n = 2;

for i = 1:n
    if i == 1
        % Load the dataset using cs_custom_load_dataset
        dataset = cs_top_custom_load_dataset(d,ratio,flag,Names,idxtop5);
        cubegroup='cs'
    else %Load the dataset using ctrl_load_dataset
        dataset = ctrl_top_custom_load_dataset(d,ratio,flag,Names,idxtop5);
        cubegroup='ctrl'
    end

    %% Input parameters for recall and training
    dataset.training_data=dataset.data;
    dataset.validation_data=dataset.data;
    dataset=StepForward_encoding(dataset,true); %use true for training + classifier, false for recall

   %% load initialized cube
    load('get5cube22022023.mat'); %loads locked cube with standardized connections and weights
    neucube=get5cube(neucube,idxtop5,folder_name)
    %% Add unsupervised parameters

    neucube_weight_befor_training=neucube.neucube_weight;

    %% unsupervised training
    neucube=Neucube_unsupervised(dataset, neucube);
    neucube.step=4;
    neucube.classifier.C=1;
    cubegroup_name=strcat(folder_name, '_', cubegroup, 'stim5');
    cubegroup_filename=strcat(cubegroup_name, '.mat');
    if strcmp(cubegroup, 'cs')
        cs5 = neucube; % Save neucube data to variable CS if cubegroup is 'CS'
        save(cubegroup_filename,'cs5'); %saves to particular filepath
    elseif strcmp(cubegroup, 'ctrl')
        ctrl5 = neucube; % Save neucube data to variable control if cubegroup is 'control'
        save(cubegroup_filename, 'ctrl5'); %saves to particular filepath
    end
end

sub5weight=cs5.neucube_weight-ctrl5.neucube_weight;
sub5_name=strcat(folder_name, '_substim5');
sub5_filename=strcat(sub5_name, '.mat');
sub5=neucube;
save(sub5_filename, 'sub5');
neucube.neucube_weight=sub5weight;


%% Load all samples and train on 5subcube
[dataset,num_sample]=top_custom_load_dataset(d,ratio,flag,Names,idxtop5);

dataset.training_data=dataset.data;

dataset.validation_data=dataset.data;

%% Stepforward encoding
dataset=StepForward_encoding(dataset,true); %use true for training + classifier, false for recall

%% supervised training
neucube=Neucube_supervised(dataset, neucube);

%% Verify Classifier
sample_amount=dataset.sample_amount_for_validation;
target_value=dataset.target_value_for_validation;
dataset.spike_state_for_validation=dataset.spike_state_for_training;
[neucube,output_neurals_test_weight]=Neucube_verification(dataset, neucube);%update cube with new data and create output neuron weights using deSNN.m

csvwrite(strcat(d,"stim5",".csv"),output_neurals_test_weight)
% Output neuron proportion and FIN matrix
allfeatures=cluster(sub5,folder_name,'stim5');

