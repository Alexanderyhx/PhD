%% ONLY RUN THIS ONCE AND USE THIS CUBE FOR ALL 5 CUBES.

load('subverifiedenv.mat') %run any 32 feature cube, then save environment and use this code to initialize a 5 feature cube

SWR=neucube.small_world_radius*10
neucube.number_of_input=5 % specify number of features to train on
number_of_input=neucube.number_of_input; % then update the number of inputs in neucube
% neucube.input_mapping(1:2) = {neucube.input_mapping{1}(idxtop5,:), neucube.input_mapping{2}(idxtop5,:)}; % changes input mapping to top 5 features, commented out, please keep 32 features 10-10 config here, and then use function get5cube to change to whatever the idxtop5 is

neuinput=neucube.input_mapping{1}; %this is the input variable for features and their coordinates
neuron_location=cat(1,neucube.neuron_location(1:end-32,:),neuinput); %these are 1471+5 talairach neuron coordinates with 
neucube.neuron_location=neuron_location
neucube.number_of_neucube_neural=size(neuron_location,1)
number_of_neucube_neural=neucube.number_of_neucube_neural %1471reservoir+32input neurons count

% %% change from a 32 feature cube to 5 feature
% neucube.indices_of_input_neuron=neucube.indices_of_input_neuron(idxtop5,:) %indexes top5 feature location in the 1471 talairach coordinate system
indices_of_input_neuron=neucube.indices_of_input_neuron;
% 
% %find the index of each input neuron
% for k=1:size(neuinput,1)
%     coord=neuinput(k,:);
% 
%     L=ismember(neuron_location,coord,'rows');
% 
%     idx=find(L);
%     idx=sort(idx);
%     indices_of_input_neuron(k)=idx(1);
% end
% indices_of_input_neuron(indices_of_input_neuron==-1)=[];
% 
% 
% %find neurons in the middle
% L=ismember(neuinput,neuron_location,'rows');
% L=ismember(neuron_location,neuinput,'rows');
% neumid=neuron_location(~L,:);
% neucube.neumid=neumid;



%% establish connections
neudistance=L2_distance(neuron_location', neuron_location');  %distance between neurons
L=neudistance==0; %element L is true if neudistance is not 0
neudistance_inv=1./neudistance; % calculates inverse of each non-zero element. This is a modifier of weights based on euclidean distance, on the principle that the weight between two neurons is inversely proportional to their distance, so neurons closer are given more weight than those far away.
neudistance_inv(L)=0; %set to 0 to avoid runtime errors



%20 percent of the weight is positive number, and 80 percent is negative
neucube_weight = sign(rand(number_of_neucube_neural)-0.2).*rand(number_of_neucube_neural).*neudistance_inv;%makes weights, including for input neurons
neucube_connection = ones(number_of_neucube_neural); %starts a fully conneceted matrix with all ones, which will eventually be rewired.
choice=randi(2,number_of_neucube_neural)-1; %creates a matrix with the same rows and columns as the number of neurons
distancethreshold = SWR; %creates a small world radius, it limits creation of connections to its radius.
aaa=number_of_neucube_neural-number_of_input+1; %index value of the first input neuron
LL=false(number_of_neucube_neural,1); %assumes all neurons are non-input
LL(indices_of_input_neuron)=true; %assigns 'true' value at indices corrresponding to input neurons.

%in this code below, there are input neurons at the site of the reservoir.
%The reservoir neurons have connections with other reservoir neurons but
%not with the input neuron and not either for the reservoir neuron at the same input location. So no weights can be made between them. And anyway there is no distance between them and their weights would be modified to 0. 

for i =1:number_of_neucube_neural %presynaptic neuron
    for j = 1:number_of_neucube_neural %postsynaptic neuron
        
        if neudistance(i,j)>distancethreshold || j>= aaa || LL(j)  %distance must be less than threshold, for the postsynaptic neuron the index must be less than aaa and if LL is an indices of input neuron, then connection is 0
            neucube_connection(i,j)=0;
        elseif neucube_connection(i,j)==1 && neucube_connection(j,i)==1 %breaks symmetry in network wiring to remove redundancies
                if choice(i,j) == 1
                    neucube_connection(i,j) = 0;
                else
                    neucube_connection(j,i)=0;            
                end
        end
         neucube_weight(i,j)=neucube_connection(i,j)*neucube_weight(i,j); %makes weight = 1*the weight or 0* weight
         if i>=aaa %if index is more than 1471, i.e not a reservoir neuron
                 neucube_weight(i,j) = 0; %makes the appended input connections, which are in the columns after the reservoir neurons, zero (connections represent outgoing and ingoing). But this does not apply to the input neuron indices within the reservoir
         elseif LL(i) 
                neucube_weight(i,j) = 2*abs(neucube_weight(i,j)); %creates weights and connection
         end
        
    end

end

% %find the adjacent neightbors of each input neuron, kept for LDC learning
% input_neighbors={};
% for inputid=1:number_of_input
%     root_idx=indices_of_input_neuron(inputid);
%     [id_queue, rank_queue, pairs]=get_descendants(root_idx, neucube_connection, 1);
%     input_neighbors{inputid}.id_queue=id_queue;
%     input_neighbors{inputid}.rank_queue=rank_queue;
% end

%finalize initialization
neucube.neucube_weight=neucube_weight;
neucube.neucube_connection=neucube_connection;
% neucube.input_neighbors=input_neighbors;

%erase previous cube data
neucube.neucube_output=[]
neucube.spike_transmission_amount=[]
neucube.step = []
neucube.type=[]
neucube.classifier_flag=[]
neucube.classifier.output_neurals_train_weight = []
neucube.classifier.training_target_value =[]
neucube.classifier.firing_order =[]
neucube.classifier.output_neurals_PSP =[]
neucube.classifier.output_neurals_test_weight = []

save('locked5cube22022023.mat','neucube')
