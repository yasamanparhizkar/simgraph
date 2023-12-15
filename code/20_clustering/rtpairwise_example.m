load '../../data/original_files/spikes113.mat'

% randomly divide time bins into train/test sets, but keep all repeats of
% the same bin in the same set.
[ncells, nrepeats, nframes] = size(spikes113);

% create a pairwise maximum entropy model
model = maxent.createModel(ncells,'rtpairwise', nrepeats, nframes);

% train the model to a threshold of 15 standard deviations from the error of computing the marginals.
% because the distribution is larger (113 dimensions) we cannot explicitly iterate over all 2^113 states
% in memory and will use markov chain monte carlo (MCMC) methods to obtain an approximation
model = maxent.trainModel(model,reshape(spikes113, ncells, []),'threshold',15, 'savefile', 'training_reentry.mat','save_delay', 300, 'max_nsamples', nan);

% save the optimized model
save('opt_model.mat', 'model');

% % normalized the model
% if strcmpi(model.type, 'rtpairwise') 
%             
%     nframes = model.nframes;
%     for t=1:nframes
%         % create the equivalent pairwise model
%         model_prim = maxent.createModel(ncells, 'pairwise');
%         model_prim.factors(1:ncells) = model.factors(t*ncells-ncells+1:t*ncells);
%         model_prim.factors(ncells+1:end) = model.factors(nframes*ncells+1:end);
%         model_prim.step_scaling(1:ncells) = model.step_scaling(t*ncells-ncells+1:t*ncells);
%         model_prim.step_scaling(ncells+1:end) = model.step_scaling(nframes*ncells+1:end);
%         % normalize
%         model_prim = maxent.wangLandau(model_prim,'binsize',0.1,'depth',15);
%         model.factors(t*ncells-ncells+1:t*ncells) = model_prim.factors(1:ncells);
%         model.factors(nframes*ncells+1:end) = model_prim.factors(ncells+1:end);
%     end
% end