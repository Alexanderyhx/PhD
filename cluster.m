function allfeatures=cluster(neucube,folder_name,numfeature)

%clustering
number_of_input=neucube.number_of_input;
number_of_neucube_neural=neucube.number_of_neucube_neural;
neucube_weight=neucube.neucube_weight;
spike_transmission_amount=neucube.spike_transmission_amount;
indices_of_input_neuron=neucube.indices_of_input_neuron;
neuron_location=neucube.neuron_location;
neuinput=neucube.input_mapping{1};
neumid=neucube.neumid;

N=number_of_neucube_neural-number_of_input;

display_with=1;
if display_with==1 %graph edge weighted by network weight
    W=abs(neucube_weight(1:N,1:N));
elseif display_with==2%graph edge weighted by spikes communication
    W=spike_transmission_amount(1:N,1:N);
end


W=(W+W'); %doubles the connection weights of W whilst also keeping neuron pairings the same connection weight
d=sum(W);
LL=d==0;

Dinv=diag(1./(sqrt(sum(W))+eps)); %explained by alex, creates a scaling multipler for each neuron based on its total connections with every other neuron, large total number = smaller scaling multiplier vice versa: diagonal matrix representing the sum of all connections for each neuron on a scale of 0 to 1, no negatives since W is absolute values, inflates small sum, deflates big sum
S=Dinv*W*Dinv; %explained by alex, this creates a value that will be minused off at the end, the larger the value the lower the final value, which means higher total connections with every other neuron still wins out. multiples the doubled connection weight of each neuron pair by each neuron's scaling multiplier twice. this essentially lowers the abs value since decimals are being multiplied
Y=zeros(size(W,1),number_of_input); %explained by alex, creates a matrix of zeroes row size same as W, and columns same as number of inputs 
for c=1:number_of_input
    Y(indices_of_input_neuron(c),c)=1;  %explained by alex, puts a value of 1 at each input neuron at its appropriate index
end

F=(eye(size(W))-0.99*S)\Y; %explained by alex, in simple english, what matrix multiplied by this diagonal matrix, gives a matrix size of Y (of values 0?) solves for matrix F, creates a matrix of 1's on the main diagonal the size of W, the values are altered to 1-0.99*matrix S

[~,label]=max(F,[],2); %explained by alex, the column corresponding to the max value in each row(neuron) belongs to that inputneuron, returns the maximum number of elements in each row of matrix F, tilde ~ is put because we dont care about the max value in the row.

feature_mapping=neucube.input_mapping;

indices_of_input_neuron=neucube.indices_of_input_neuron;
neuron_location=neucube.neuron_location;

neuinput=neucube.input_mapping{1};

% W=ClusterAnalysisW;
% label=ClusterAnalysislabel;

Names=feature_mapping{2};

% inputnueron=feature_mapping{1};

% total cut between clusters
Wcut=zeros(number_of_input);
E=ones(size(W));
for i=1:number_of_input
    for j=1:number_of_input
        %                 Li=double(label==i);
        %                 Lj=double(label==j);
        idxi=indices_of_input_neuron(i);
        idxj=indices_of_input_neuron(j);
        Li=label==label(idxi); % same cluster with this input neuron
        Lj=label==label(idxj); % same cluster with this input neuron
        Wcut(i,j)=Li'*W*Lj;
    end
end
Wcut(logical(eye(size(Wcut))))=0;
Wcut=round(Wcut/max(Wcut(:))*min(7,number_of_input));


wcut_name=strcat('wcut',numfeature,folder_name);
writematrix(Wcut,wcut_name) %gives CSV file for FIN for network analysis

% neuron proportion
vertex_num=zeros(number_of_input,1);
for c=1:number_of_input
    vertex_num(c)=sum(label==c);
end
total_count = sum(vertex_num);
for c = 1:length(vertex_num)
    vertex_num(c) = vertex_num(c) / total_count * 100;
end

if(size(Names,1)==1)
    Names=Names';
end


[sorted_vertex, idx] = sort(vertex_num, 'descend'); %sort vertex_num neuron proportion
sorted_Names = Names(idx);
idxtop5 = idx(1:5);
top_5_num=sorted_vertex(1:5); % takes top 5 vertex_num neuron proportion
top_5_name= sorted_Names(1:5); % takes top 5 Features

allfeatures.top_5_num = top_5_num;
allfeatures.top_5_name = top_5_name;
allfeatures.sorted_Names = Names(idx);
allfeatures.Wcut = Wcut;
allfeatures.idxtop5 = idxtop5;


percentValues = num2str(vertex_num);
% customStrings = strcat('(',percentValues);
% customStrings = strcat(customStrings,')');
customStrings = strcat(Names,',',percentValues);
np=customStrings;
np_name= strcat('np',numfeature,folder_name);
writecell(np,np_name); %gives CSV file for neuron proportions]