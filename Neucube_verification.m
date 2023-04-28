function [neucube,output_neurals_test_weight]=Neucube_verification(dataset, neucube)
stage=2; %to use validation data
STDP=0;
neucube=Neucube_updating(dataset,neucube,stage,STDP,1);
flag=neucube.classifier_flag;

%Use deSNN_validation function: update neucube with firing order+outlayer weights
[firing_order, output_neurals_test_weight]=deSNN_validation(dataset, neucube);
neucube.classifier.firing_order=firing_order;
neucube.classifier.output_neurals_test_weight=output_neurals_test_weight;
