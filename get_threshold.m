function variable_threshold=get_threshold(dataset,alpha)
data=dataset.data; %use all data
%data=dataset.training_data; % just use training data
spk=diff(data,1,1);
nan_idx = isnan(spk); %find NaN values in spk, this needed so the code doesn't bug out below when calculating the mean and std.

% Replace NaN values with 0
spk(nan_idx) = 0;
variable_threshold=zeros(1,size(data,2));
for k=1:size(spk,3)
    spk1=spk(:,:,k);
    variable_threshold=variable_threshold+mean(abs(spk1),1)+std(abs(spk1),0,1)*alpha;
end
variable_threshold=variable_threshold'/size(spk,3);
%variable_threshold=alpha %keep the threshold the same as inputted value
%and not averaged across all samples
