%---Load dataset---%
d='/Users/yaale244/Desktop/Alex19022023/CSBCIPredictionAlgorithm/tcDCS post stim/2s 30-31/placebo';
file_name=d;
flag=[]; %empty flag=[] means recall, true means classification, false means regression
[sam_files,count,token]=custom_filebyfileload(d,file_name,flag);

for i=1:size(sam_files,1)    %for all the files tokenize the digits
    exp=regexp(sam_files(i).name,'sam(\d+)','tokens');   %%search for numbers after 'sam' in each file
    if(~isempty(exp)) %%checks if the sample file contains digits after 'sam'
        newfiles(count)=sam_files(i);
        token(count)=str2num(cell2mat(exp{1}));
        index(count)=i;
        count=count+1;
    end
end

if(size(token,1)~=0)
    [B,I]=sort(token); %%sort the tokens from the 'sam' files
end
%--------------------Load feature names---------------------%
    
    Names=textread(fullfile(d, 'feature_names_eeg.txt'), '%s');



%-------READ SAMPLES ONE BY ONE----%
for i=1:size(I,2) %%read the sample files in increasing order of the digitså
        
    %       creates empty dataset
    dataset=create_empty_dataset();
 
    data=csvread(fullfile(d,newfiles(I(i)).name));
    size(data)
    num_time=size(data,1);
    num_feature=size(data,2);      
%     data(:,:,1)=data; %this line appends data from the next iteration, comment this to remove
   
    eeg_data=data;
    num_sample=size(data,3);
    files = dir(fullfile(d, 'tar*.csv'));%%choose all the files starting with tar
    class_label=csvread(fullfile(d,files(1).name));
    class_label=class_label';
    

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
    
%         fills dataset with variables
    dataset.file_name=file_name;
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


    %INPUT PARAMETERS for recall or training:
    dataset.training_set_ratio=1;
    dataset.training_data=dataset.data;
    dataset.validation_data=dataset.data;
    dataset.target_value_for_validation=dataset.target_value_for_training;
    dataset.sample_amount_for_training=num_sample;
    dataset.sample_amount_for_validation=num_sample;
    
    
    
    %-----Use Stepforward encoding function---------%
    dataset.encoding.method=4; %for stepforward
    if dataset.encoding.method==4
        dataset=StepForward_encoding(dataset,false); %flag here is different from above, it means training, empty means validation
    end
    
    %---LOAD NEUCUBE NOW IF USING RECALL--%
    
    load("30-32 trained sub cube 20042022.mat") %for cybersickness algorithm, change as needed.
    
    %----Verify Classifier----%
    sample_amount=dataset.sample_amount_for_validation;
    target_value=dataset.target_value_for_validation; 
    [neucube,output_neurals_test_weight]=Neucube_verification(dataset, neucube);%update cube with new data and create output neuron weights using deSNN.m
    writematrix(output_neurals_test_weight,sprintf('placebooutlayerweightsam%d.csv',i))
    clearvars -except B count d exp file_name flag i I index newfiles sam_file token % remove clearvars to see the last added variables
    clearvars -global -except B count d exp file_name flag i I index newfiles sam_file token %remove clearvars to see the last added variables
end    
    
%     %-----Update x and y in neucube.classifier folder-----%
%     maxy=40;
%     if sample_amount>maxy
%         layer=ceil(sample_amount/maxy);
%         n=ceil(sample_amount/layer);
%         xx=[];
%         yy=[];
%         for k=1:layer-1;
%             xx=[xx zeros(1, n)+k];
%             yy=[yy (1:n)];
%             sample_amount=sample_amount-n;
%         end
%         xx=[xx zeros(1, sample_amount)+layer];
%         yy=[yy (1:sample_amount)];
%     else
%         layer=1;
%         xx=ones(1, sample_amount);
%         yy=1:sample_amount;
%     end
%     
%     if isempty(target_value)
%         return
%     end
%     
%     cls=unique(target_value);
%     neucube.classifier.x=x;
%     neucube.classifier.y=y;

%   
% 
%     
% 
