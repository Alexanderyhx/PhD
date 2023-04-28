

number_of_input=neucube.number_of_input;
number_of_neucube_neural=neucube.number_of_neucube_neural;
neucube_weight=neucube.neucube_weight;
spike_transmission_amount=neucube.spike_transmission_amount;
indices_of_input_neuron=neucube.indices_of_input_neuron;
neuron_location=neucube.neuron_location;
neuinput=neucube.input_mapping{1};
neumid=neucube.neumid;


input_neuron=str2num(get(handles.input_neuron_cluster_edit,'string'));


Names=neucube.input_mapping{2};
if is_all_neuron==1
    input_neuron=1:number_of_input;
end




%'Performing clustering...'

N=number_of_neucube_neural-number_of_input;
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

W=ClusterAnalysisW;
label=ClusterAnalysislabel;

Names=feature_mapping{2};
inputnueron=feature_mapping{1};

value = 1 %1 total cut between clusters, 2 average, 3 vertex number, 4 plotclusterW
switch value
    case 1 % total cut between clusters
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




        writematrix(Wcut,folder_name) %gives CSV file for FIN for network analysis


    case 2 % average cut between clusters
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
                Wcut(i,j)=Li'*W*Lj/sum(Li'*E*Lj);
            end
        end
        Wcut(logical(eye(size(Wcut))))=0;
        Wcut=Wcut/max(Wcut(:))*number_of_input;
        
    case 3 % vertext number
        vertex_num=zeros(number_of_input,1);
        for c=1:number_of_input
            vertex_num(c)=sum(label==c);
        end
        
        customStrings = strcat('(',percentValues);
        customStrings = strcat(customStrings,')');
        if(size(Names,1)==1)
            Names=Names';
        end
        customStrings = strcat(Names,customStrings);


        writecell(customStrings,folder_name); %gives CSV file for neuron proportions

    case 4 %plot clustersW
     
        d=sum(W);
        LL=d==0;
        input_neuron=1:number_of_input;
        N=number_of_neucube_neural-number_of_input;
end


number_of_input=neucube.number_of_input;
number_of_neucube_neural=neucube.number_of_neucube_neural;
feature_mapping=neucube.input_mapping;

spike_transmission_amount=neucube.spike_transmission_amount;
indices_of_input_neuron=neucube.indices_of_input_neuron;
neuron_location=neucube.neuron_location;
neuinput=neucube.input_mapping{1};

Names=feature_mapping{2};

%%information trace 
is_all_neuron=1
if is_all_neuron==1
    input_neuron=1:number_of_input;
end

display_with=1
if display_with==1 % max spike gradient
    N=number_of_neucube_neural-number_of_input;
    
    pathes=cell(N,1);
    rootID=zeros(N,1);
    W=spike_transmission_amount(1:N,1:N);
    
    %find path and root neuron of each one
    for k=1:N
        
        receiverID=k;

        
        path=[];
        loop=0;
        while ~any(receiverID==indices_of_input_neuron) && loop<=N %walk towards its max sender
            received_spikes=W(:,receiverID);
            M=max(received_spikes);
            if M==0
                break;
            end
            path(end+1)=receiverID;
            receiverID=find(received_spikes==M);
            receiverID=receiverID(1);
            loop=loop+1;
        end

        if length(path)>=1 && loop<=N
            path(end+1)=receiverID;
            pathes{k}=path;
            rootID(k)=receiverID;
        end
    end
    
    %plot the path
  
    for kk=input_neuron
        inputID=indices_of_input_neuron(kk);
      
        LL=rootID==inputID;
        for k=1:N
            if LL(k)==false
                continue;
            end
            path=pathes{k};
            width=3;
            for i=length(path):-1:2
                senderID=path(i); receiverID=path(i-1);
                sender=neuron_location(senderID,:);
                receiver=neuron_location(receiverID,:);
                if width<1
                    width=1;
                end
                width=width-1;
            end
            
        end
    end
   

else  % information amount or spreading level
    
    N=number_of_neucube_neural-number_of_input;

    W=spike_transmission_amount(1:N,1:N);
    for kk=input_neuron
        idx=indices_of_input_neuron(kk);
        
        n=0;
        id_queue=[];
        rank_queue=[];
        percent_queue=[];
        pairs=[]; % record index pair of a parent node and its kid node
        
        id_queue(end+1)=idx;    %start in root node, the input neuron
        rank_queue(end+1)=1;    %root rank 1
        percent_queue(end+1)=1; %has 100% information
        
        
        while n<length(id_queue)
            n=n+1; %sender's location in queue
            senderID=id_queue(n);
            
            receivers=find(W(senderID,:)>0); % all receivers
            
            for k=1:length(receivers)
                receiverID=receivers(k);
                ratio=W(senderID,receiverID)/sum(W(:,receiverID)); % percentage of the spikes received from this sender over the spikes received from all other senders
                %             ratio=spike_transmission_amount(receiverID,senderID)/sum(spike_transmission_amount(:,senderID)); % percentage of the spikes received from this sender over the spikes received from all other senders
                if ~any(receiverID==id_queue)% this receiver not in the queue
                    id_queue(end+1)=receivers(k);
                    rank_queue(end+1)=rank_queue(n)+1; % sender's level +1
                    percent_queue(end+1)=percent_queue(n)*ratio; %percent of spike information received from this input neuron
                    pairs(:,end+1)=[senderID;receiverID];
                    %                 if (display_with==1 && percent_queue(end)>display_threshold) || (display_with==2 && rank_queue(end)<=display_threshold)% info amount
                    %                     sender=neuron_location(senderID,:);
                    %                     receiver=neuron_location(receiverID,:);
                    %                     plot3([receiver(1);sender(1)],[receiver(2);sender(2)],[receiver(3);sender(3)],'color',classcolor(kk,:),'linewidth',2,'markersize',15); % show edge
                    %                     plot3([receiver(1);sender(1)],[receiver(2);sender(2)],[receiver(3);sender(3)],'*','color',classcolor(kk,:),'markersize',15); % show neuron
                    %                     plot3([receiver(1);sender(1)],[receiver(2);sender(2)],[receiver(3);sender(3)],'.','color',classcolor(kk,:),'markersize',15); % show neuron
                    %                 end
                elseif rank_queue(receiverID==id_queue)==max(rank_queue) % no recycle considered, i.e. no percentage is added to its parents or older...
                    ii=receiverID==id_queue;
                    percent_queue(ii)=percent_queue(ii)+percent_queue(n)*ratio;
                end
                
            end
        end
        
        
        linewidth=max(rank_queue)+1-rank_queue;
        
        minwidth=max(linewidth);
        for k=1:length(pairs)
            if (display_with==3 && percent_queue(k)>display_threshold) || (display_with==2 && rank_queue(k)<=display_threshold)% info amount
                if linewidth(k)<minwidth;
                    minwidth=linewidth(k);
                end
            end
        end
        linewidth=(linewidth-minwidth+1);
        for k=1:length(pairs)
            senderID=pairs(1,k);
            receiverID=pairs(2,k);
            if (display_with==3 && percent_queue(k)>display_threshold) || (display_with==2 && rank_queue(k)<=display_threshold)% info amount
                sender=neuron_location(senderID,:);
               receiver=neuron_location(receiverID,:);
            end
        end
        
    end
  
end
