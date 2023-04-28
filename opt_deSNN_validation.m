function [firing_order, output_neurals_test_weight]=opt_deSNN_validation(dataset, neucube)
    
classifier=neucube.classifier;
if isempty(classifier)
    error('No classifier!');
end
neucube_output_for_validation=neucube.neucube_output;
length_per_sample=dataset.validation_time_length;
input_dimension = neucube.number_of_neucube_neural;
amount_to_validate = dataset.sample_amount_for_validation; %
drift = classifier.drift;
mod_for_deSNN =  classifier.mod;
driftup= classifier.driftup;
driftdown= classifier.driftdown;
K=classifier.K;
sigma=classifier.sigma;
C=classifier.C;
output_neurals_train_weight=classifier.output_neurals_train_weight;
training_target_value=classifier.training_target_value;


output_neurals_test_PSP = zeros(1,amount_to_validate);
output_neurals_test_weight = zeros(amount_to_validate,input_dimension);
output_neurals_test_order = ones(amount_to_validate,input_dimension) * -100; 

order_sn = 0;

for x = 1:amount_to_validate

    order_sn = 0;
    order=[];
    for m = 1:length_per_sample
        for n=1:input_dimension
            if neucube_output_for_validation(((x-1)*length_per_sample+m),n) == 1
                if output_neurals_test_order(x,n) < 0   %表示这个突触还没有spike到来过
                    output_neurals_test_order(x,n) = order_sn;
                    order_sn = order_sn + 1;
                    output_neurals_test_weight(x,n) = mod_for_deSNN^order_sn;
                    order(end+1)=n;
                else
                    output_neurals_test_weight(x,n) = output_neurals_test_weight(x,n) + driftup;
                end
                output_neurals_test_PSP(x) = output_neurals_test_PSP(x) + output_neurals_test_weight(x,n);
            else
                output_neurals_test_weight(x,n) =output_neurals_test_weight(x,n) - driftdown;
            end
        end
    end
    
   
    firing_order{x}=order;
end



% if dataset.type==2 %regression
%      output_tartget_value=wknnr(output_neurals_train_weight',training_target_value,output_neurals_test_weight',K, sigma);
% else
%      output_tartget_value=wknn(output_neurals_train_weight',training_target_value,output_neurals_test_weight',K);
% end

