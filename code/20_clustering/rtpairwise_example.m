load '../../data/original_files/spikes113.mat'

% randomly divide time bins into train/test sets, but keep all repeats of
% the same bin in the same set.
[ncells, nrepeats, nframes] = size(spikes113);

% create a pairwise maximum entropy model
model = maxent.createModel(ncells,'rtpairwise', nrepeats, nframes);

% train the model to a threshold of 15 standard deviations from the error of computing the marginals.
% because the distribution is larger (113 dimensions) we cannot explicitly iterate over all 2^113 states
% in memory and will use markov chain monte carlo (MCMC) methods to obtain an approximation
model = maxent.trainModel(model,reshape(spikes113, ncells, []),'threshold',15, 'max_nsamples', nan);