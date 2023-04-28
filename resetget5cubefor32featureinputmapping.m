load("get5cube22022023.mat")
get5=neucube
load("32and5subverifiedenv.mat")
get5.input_mapping=cs.input_mapping
get5.indices_of_input_neuron=cs.indices_of_input_neuron

neucube = get5

save("get5cube22022023.mat", 'neucube')