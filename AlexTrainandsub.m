%% Load Dataset
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

n = 2;

for i = 1:n
    if i == 1
        % Load the dataset using cs_custom_load_dataset
        dataset = cs_custom_load_dataset(d,ratio,flag,Names);
        cubegroup='cs'
    else %Load the dataset using ctrl_load_dataset
        dataset = ctrl_custom_load_dataset(d,ratio,flag,Names);
        cubegroup='ctrl'
    end

    %% Input parameters for recall and training
    dataset.training_data=dataset.data;
    dataset.validation_data=dataset.data;
    dataset=StepForward_encoding(dataset,true); %use true for training + classifier, false for recall

   %% load initialized cube

    load("initialiseallfeaturescubeasof20042022.mat");
    %% Add unsupervised parameters
    
    neucube.STDP_rate=unsup_params.STDP_rate;
    neucube.threshold_of_firing=unsup_params.threshold_of_firing;
    neucube.potential_leak_rate=unsup_params.potential_leak_rate;
    neucube.refactory_time=unsup_params.refactory_time;
    neucube.LDC_probability=unsup_params.LDC_probability;
    neucube.LDC_initial_weight=0.05;
    neucube.training_round=unsup_params.training_round;
    neucube_weight_befor_training=neucube.neucube_weight;

    %% unsupervised training
    neucube=Neucube_unsupervised(dataset, neucube);
    neucube.step=4;
    neucube.classifier.C=1;
    cubegroup_name=strcat(folder_name, '_', cubegroup);
    cubegroup_filename=strcat(cubegroup_name, '.mat');
    if strcmp(cubegroup, 'cs')
        cs = neucube; % Save neucube data to variable CS if cubegroup is 'CS'
        save(cubegroup_filename,'cs'); %saves to particular filepath
    elseif strcmp(cubegroup, 'ctrl')
        ctrl = neucube; % Save neucube data to variable control if cubegroup is 'control'
        save(cubegroup_filename, 'ctrl'); %saves to particular filepath
    end
end

subweight=cs.neucube_weight-ctrl.neucube_weight;
sub_name=strcat(folder_name, '_sub');
sub_filename=strcat(sub_name, '.mat');
sub=neucube;
save(sub_filename, 'sub');
neucube.neucube_weight=subweight;


%% Load all samples and train on subcube
[dataset,num_sample]=custom_load_dataset(d,ratio,flag,Names);


%% Input parameters for recall and training:

dataset.training_data=dataset.data;

dataset.validation_data=dataset.data;

%% Stepforward encoding
dataset=StepForward_encoding(dataset,true); %use true for training + classifier, false for recall
%% create supervised parameters

classifier_flag=1;
mod=0.005;
drift=0.8;
K=3;
sigma=1;
C = 1;

% neucube.classifier=reset_classifier(sup_params); % empty classifier
neucube.classifier_flag=classifier_flag;
neucube.classifier.mod=mod;
neucube.classifier.drift=drift;
neucube.classifier.K=K;
neucube.classifier.sigma=sigma;

%% supervised training
neucube=Neucube_supervised(dataset, neucube);

neucube.step=5;

%% Verify Classifier
sample_amount=dataset.sample_amount_for_validation;
target_value=dataset.target_value_for_validation;
dataset.spike_state_for_validation=dataset.spike_state_for_training;
[neucube,output_neurals_test_weight]=Neucube_verification(dataset, neucube);%update cube with new data and create output neuron weights using deSNN.m

csvwrite(strcat(d,".csv"),output_neurals_test_weight)
%% Output neuron proportion and FIN matrix
allfeatures=cluster(sub,folder_name,'32');

%% choose top 5 features

idxtop5 = allfeatures.idxtop5

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
    load('get5cube22022023.mat') %loads locked cube with standardized connections and weights
    neucube=get5cube(neucube,idxtop5,folder_name)
    %% Add unsupervised parameters

    neucube_weight_befor_training=neucube.neucube_weight;

    %% unsupervised training
    neucube=Neucube_unsupervised(dataset, neucube);
    neucube.step=4;
    neucube.classifier.C=1;
    cubegroup_name=strcat(folder_name, '_', cubegroup);
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
sub5_name=strcat(folder_name, '_sub5');
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

csvwrite(strcat(d,"top5",".csv"),output_neurals_test_weight)
%% Output neuron proportion and FIN matrix
allfeatures=cluster(sub5,folder_name,'5');
