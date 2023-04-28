function neucube=deSNN_training(dataset, neucube)        

classifier=neucube.classifier;

input_dimension = neucube.number_of_neucube_neural;
length_per_sample=dataset.training_time_length;
amount_to_train = dataset.sample_amount_for_training; % 
class_label_for_training=dataset.target_value_for_training;
neucube_output_for_training=neucube.neucube_output;
mod_for_deSNN=classifier.mod;
drift=classifier.drift;
C=classifier.C;

output_neurals_PSP = zeros(1,amount_to_train);
output_neurals_weight = zeros(amount_to_train,input_dimension);
output_neurals_class_sn = ones(1,amount_to_train) * -1;
output_neurals_order = ones(amount_to_train,input_dimension) * -100; 

for x = 1:amount_to_train
    
    order_sn = 0;
    order=[];
    
    for m = 1:length_per_sample
        for n=1:input_dimension
            if neucube_output_for_training(((x-1)*length_per_sample+m),n) == 1
                if output_neurals_order(x,n) < 0   %indicates that a spike has not arrived at this synpase 
                    output_neurals_order(x,n) = order_sn;
                    order_sn = order_sn + 1;
                    output_neurals_weight(x,n) = mod_for_deSNN^order_sn;
                    order(end+1)=n;
                else
                    output_neurals_weight(x,n) = output_neurals_weight(x,n) + drift;
                end
                output_neurals_PSP(x) = output_neurals_PSP(x) + output_neurals_weight(x,n);
            else
                output_neurals_weight(x,n) = output_neurals_weight(x,n) - drift;
            end
        end
    end
    output_neurals_class_sn(x) = class_label_for_training(x);   %assign the class number to which the output neuron belongs 
    firing_order{x}=order;
    
end

classifier.output_neurals_train_weight=output_neurals_weight;
classifier.training_target_value=output_neurals_class_sn;
classifier.firing_order=firing_order;
classifier.output_neurals_PSP=output_neurals_PSP;

neucube.classifier=classifier;