function [L,label] = dataLoader_normalized_minmax(data,n_sample,b_ind,rsrng)
%DATALOADER Summary of this function goes here
%   Detailed explanation goes here
%   function [val] = cal_function(f1,f2)
%     val = exp(-sum((f1-f2).*(f1-f2)));
%   end
%data = csvread(path);
rng(rsrng); % random seed 0
reorderx=randperm(n_sample);
reorderx=reorderx(1:n_sample);
% K=round(num/n_sample); % test 4 samples at a time

% data_idx=(indices==1);
selected_data=data(reorderx,:);

feature=selected_data(:,1:end-1);

feature(isnan(feature))=0;

n_feature=size(feature,2);

label=selected_data(:,end);
label(label~=1)=-1;

feature_train = feature(b_ind,:);
[feature_train] = scale_01(feature_train);
feature_train_NaN=isnan(feature_train);
column_to_remove=[];
for ii=1:n_feature
    if length(find(feature_train_NaN(:,ii)))>0
    column_to_remove=[column_to_remove;ii];
    end
end

% mean_TRAIN_set_0 = mean(feature_train);
% std_TRAIN_set_0 = std(feature_train);
% 
% mean_TRAIN_set = repmat(mean_TRAIN_set_0,size(feature_train,1),1);
% std_TRAIN_set = repmat(std_TRAIN_set_0,size(feature_train,1),1);
% 
% feature_train = (feature_train - mean_TRAIN_set)./std_TRAIN_set;

% if length(find(isnan(feature_train)))>0
%     error('features have NaN(s)');
% end

% feature_train_l2=sqrt(sum(feature_train.^2,2));
% for i=1:size(feature_train,1)
%     feature_train(i,:)=feature_train(i,:)/feature_train_l2(i);
% end

feature_test = feature(length(b_ind)+1:n_sample,:);
[feature_test] = scale_01(feature_test);
feature_test_NaN=isnan(feature_test);
for ii=1:n_feature
    if length(find(feature_test_NaN(:,ii)))>0
    column_to_remove=[column_to_remove;ii];
    end
end

feature_train(:,column_to_remove)=[];
feature_test(:,column_to_remove)=[];

n_feature=size(feature_train,2);

% mean_TEST_set = repmat(mean_TRAIN_set_0,size(feature_test,1),1);
% std_TEST_set = repmat(std_TRAIN_set_0,size(feature_test,1),1);

% feature_test = (feature_test - mean_TEST_set)./std_TEST_set;

% feature_test_l2=sqrt(sum(feature_test.^2,2));
% for i=1:size(feature_test,1)
%     feature_test(i,:)=feature_test(i,:)/feature_test_l2(i);
% end

feature_REFORM = feature;
feature_REFORM(:,column_to_remove) = [];

feature_REFORM(b_ind,:) = feature_train;
feature_REFORM(length(b_ind)+1:n_sample,:) = feature_test;

[c,y] = get_graph_Laplacian_variables_ready(feature_REFORM,label,n_sample,n_feature);


% identity M
M=eye(n_feature);

% learned M by Cholesky Decomposition
% initial_label_index=zeros(n_sample,1);
% initial_label_index(b_ind)=1;
% initial_label_index=logical(initial_label_index);
% class_train_test=label;
% [U] = ...
%     Cholesky_M( feature_REFORM, ...
%     initial_label_index, ...
%     class_train_test);
% M=U*U';

[L] = graph_Laplacian( n_sample, c, M );
   
% A = zeros(num,num);
% for i=1:num
%   for j=1:num
%       if i~=j
%     A(i,j) = cal_function(data(i,1:dim-1),data(j,1:dim-1));
%       end
%   end
% end
% label = data(:,end);
% label(label==2)=-1;
end
