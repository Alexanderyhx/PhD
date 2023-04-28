%---Load dataset---
d='/Users/yaale244/Desktop/Alexpersonalneucube/test';
file_name=d;
flag=[]; %empty flag=[] means recall, true means classification, false means regression
[dataset,num_sample]=custom_load_dataset(d,file_name,flag);

%INPUT PARAMETERS for recall or training:
dataset.training_set_ratio=1;
dataset.training_data=dataset.data;
dataset.validation_data=dataset.data;
dataset.target_value_for_validation=dataset.target_value_for_training;
dataset.sample_amount_for_training=num_sample;
dataset.sample_amount_for_validation=num_sample;
%--------------------Load feature names---------------------%

Names=textread(fullfile(d, 'feature_names_eeg.txt'), '%s');


%-----Use Stepforward encoding function---------%
dataset.encoding.method=4; %for stepforward
if dataset.encoding.method==4
    dataset=StepForward_encoding(dataset,false); %flag here is different from above, it means training, empty means validation
end

%---LOAD NEUCUBE NOW IF USING RECALL--%

load("30-32 trained sub cube 20042022.mat") %for cybersickness algorithm, change as needed.

%----Verify Classifier----%
sample_amount=dataset.sample_amount_for_validation;
target_value=dataset.target_value_for_validation; % this is for classification and regression not recall? not sure
[neucube]=Neucube_verification(dataset, neucube);%update cube with new data and create output neuron weights using deSNN.m

% dataset.predict_value_for_validation=predict_value_for_validation;

%csvwrite(strcat('/Users/yaale244/Desktop/Alexpersonalneucube/',outlayerweights),neucube.classifier.output_neurals_test_weight)




