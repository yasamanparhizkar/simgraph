load example50

% randomly divide into train/test sets
[ncells,nsamples] = size(spikes50);
idx_train = randperm(nsamples,ceil(nsamples/2));
idx_test = setdiff(1:nsamples,idx_train);
samples_train = spikes50(:,idx_train);
samples_test = spikes50(:,idx_test);

% create a pairwise maximum entropy model
model = maxent.createModel(ncells,'pairwise');

% train the model to a threshold of 1.5 standard deviations from the error of computing the marginals.
% because the distribution is larger (50 dimensions) we cannot explicitly iterate over all 5^20 states
% in memory and will use markov chain monte carlo (MCMC) methods to obtain an approximation
model = maxent.trainModel(model,samples_train,'threshold',1.5);


% get the marginals (firing rates and correlations) of the test data and see how they compare to the model predictions.
% here the model marginals could not be computed exactly so they will be estimated using monte-carlo. We specify the
% number of samples we use so that their estimation will have the same amoutn noise as the empirical marginal values
marginals_data = maxent.getEmpiricalMarginals(samples_test,model);
marginals_model = maxent.getMarginals(model,'nsamples',size(samples_test,2));

% plot them on a log scale
figure
loglog(marginals_data,marginals_model,'b*');
hold on;
minval = min([marginals_data(marginals_data>0)]);
plot([minval 1],[minval 1],'-r'); % identity line
xlabel('empirical marginal');
ylabel('predicted marginal');
title(sprintf('marginals in %d cells',ncells));

% the model that the MCMC solver returns is not normalized. If we want to compare the predicted and actual probabilities
% of individual firing patterns, we will need to first normalize the model. We will use the wang-landau algorithm for
% this. We chose parameters which are less strict than the default settings so that we will have a faster runtime.
disp('Normalizing model...');
model = maxent.wangLandau(model,'binsize',0.1,'depth',15);

% the normalization factor was added to the model structure. Now that we have a normalized model, we'll use it to
% predict the frequency of activity patterns. We will start by observing all the patterns that repeated at least twice
% (because a pattern that repeated at least once may grossly overrepresent its probability and is not meaningful in this
% sort of analysis)
limited_empirical_distribution = maxent.getEmpiricalModel(samples_test,'min_count',2);


% get the model predictions for these patterns
model_logprobs = maxent.getLogProbability(model,limited_empirical_distribution.words);

% nplot on a log scale
figure
plot(limited_empirical_distribution.logprobs,model_logprobs,'bo');
hold on;
minval = min(limited_empirical_distribution.logprobs);
plot([minval 0],[minval 0],'-r');  % identity line
xlabel('empirical pattern log frequency');
ylabel('predicted pattern log frequency');
title(sprintf('activity pattern frequency in %d cells',ncells));



% Wang-landau also approximated the model entropy, let's compare it to the entropy of the empirical dataset.
% for this we want to look at the entire set, not just the set limited repeating patterns
empirical_distribution = maxent.getEmpiricalModel(samples_test);

% it will not be surprising to see that the empirical entropy is much lower than the model, this is because the
% distribution is very undersampled
fprintf('Model entropy: %.03f bits, empirical entropy (test set): %.03f bits\n',model.entropy,empirical_distribution.entropy);

% generate samples from the distribution and compute their entropy. This should give a result which is must closer to
% the entropy of the empirical distribution...
samples_simulated = maxent.generateSamples(model,numel(idx_test));
simulated_empirical_distribution = maxent.getEmpiricalModel(samples_simulated);
fprintf('Entropy of simulated data: %.03f bits\n',simulated_empirical_distribution.entropy);