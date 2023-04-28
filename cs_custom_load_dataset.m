function [dataset,num_sample]=cs_custom_load_dataset(d,ratio,flag,Names)



%reads files in a tokenized order and prepares some variables for dataset

files = dir(fullfile(d, '*sam*.csv'));%%choose all the files starting with sam.

files = files(1:32);
count=1;
token=[];

for i=1:size(files,1)    %for all the files tokenize the digits
    exp=regexp(files(i).name,'sam(\d+)','tokens');   %%search for numbers after 'sam' in each file
    if(~isempty(exp)) %%checks if the sample file contains digits after 'sam'
        newfiles(count)=files(i);
        token(count)=str2num(cell2mat(exp{1}));
        index(count)=i;
        count=count+1;
    end
end

if(size(token,1)~=0)
    [B,I]=sort(token); %%sort the tokens from the 'sam' files
    for i=1:size(I,2) %%read the sample files in increasing order of the digits
        data=csvread(fullfile(d,newfiles(I(i)).name));
        if(i==1)
            num_time=size(data,1);
            num_feature=size(data,2);      
            all_data(:,:,i)=data;
        end
        if(i>1)
            if((size(data,1)==num_time)&&(size(data,2)==num_feature))
                all_data(:,:,i)=data;
            end
        end
    end
end
eeg_data=all_data;
num_sample=size(all_data,3);
% files = dir(fullfile(d, 'tar*.csv'));%%choose all the files starting with tar
files = dir(fullfile(d, 'tar_test1.csv'));%%choose all the files starting with tar
class_label=csvread(fullfile(d,files(1).name));
class_label=class_label';

%creates empty dataset
dataset=create_empty_dataset();
if ~exist('Names','var') || ~iscell(Names)
    Names={};
    for k=1:size(eeg_data,2)
        Names{k}=sprintf('feature %d',k);
    end
end

if ~isempty(flag)
    cls=unique(class_label);
    if flag==false;
        dataset.number_of_class=1;
        dataset.type=2;
    else
        dataset.number_of_class=length(cls);
        dataset.type=1;
        target=zeros(size(class_label));
        for k=1:length(cls)
            L=class_label==cls(k);
            target(L)=k;
        end
        class_label=target;
    end
else
    dataset.type=3;
    class_label=[];
end

%fills dataset with variables
dataset.training_set_ratio=ratio;
dataset.file_name=d;
dataset.data=eeg_data; 
dataset.target_value=class_label(:); 
dataset.length_per_sample=size(eeg_data,1); 
dataset.feature_number=size(eeg_data,2);
dataset.total_sample_number=size(eeg_data,3);  
dataset.feature_name=Names;


dataset.training_data=[];
dataset.target_value_for_training=[];
dataset.spike_state_for_training=[];
dataset.training_time_length=size(eeg_data,1); 
dataset.sample_amount_for_training=0;
dataset.training_sample_id=[];

dataset.validation_data=[];
dataset.target_value_for_validation=[];
dataset.spike_state_for_validation=[];
dataset.validation_time_length=size(eeg_data,1);
dataset.sample_amount_for_validation=0;


training_sample_id=[];
validation_sample_id=[];
dataset.training_sample_id=training_sample_id;
dataset.validation_sample_id=validation_sample_id;

total_sample_number = dataset.total_sample_number;
sample_id=1:total_sample_number;

class_label_for_training=[];
class_label_for_validation=[];


training_percentage=dataset.training_set_ratio;

for c=1:dataset.number_of_class
    label=cls(c);
    L=class_label==label;
    sample_id_of_this_class=sample_id(L);
    number_of_samples_this_class=sum(L);
    number_of_training_sample = total_sample_number; %used to generate connection weights for all data
    number_of_validation_sample= total_sample_number; %used to generate connection weights for all data
%     number_of_training_sample=floor(number_of_samples_this_class*training_percentage)
%     number_of_validation_sample=number_of_samples_this_class-number_of_training_sample
    
    if training_percentage==1
        training_idx=1:number_of_samples_this_class;
        validation_idx=1:number_of_samples_this_class; %also added this for validation idx
    elseif training_percentage==0
        validation_idx=1:number_of_samples_this_class;
    else
        idx=randperm(number_of_samples_this_class);
        training_idx=idx(1:number_of_training_sample);
        validation_idx=idx(number_of_training_sample+1:end);
    end
    
    if number_of_training_sample>0
        class_label_for_training = ones(1,number_of_training_sample)*c;
        training_sample_id=cat(2,training_sample_id,sample_id_of_this_class(training_idx));
        
    end
    
    if number_of_validation_sample>0
        class_label_for_validation = ones(1,number_of_validation_sample)*c;
        validation_sample_id=cat(2,validation_sample_id,sample_id_of_this_class(validation_idx));
    end
    sample_amount_for_training=number_of_training_sample;
    sample_amount_for_validation=number_of_validation_sample;
    
end


dataset.sample_amount_for_training=sample_amount_for_training;
dataset.sample_amount_for_validation=sample_amount_for_validation;


dataset.target_value_for_training=class_label_for_training;
dataset.target_value_for_validation=class_label_for_validation;
