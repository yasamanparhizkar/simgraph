clear;
clc;
close all;
addpath('results\');
n_dataset=17;
n_results=1;
n_method=1;
font_size=12;
results_mean=zeros(n_dataset+1,n_results);

for dataset_i=1:n_dataset
    [dataset_str] = get_dataset_name(dataset_i);
    result_str=['results_' dataset_str '_std_scaling_aaai23_I_eg.mat'];
    load(result_str);
    results_mean(dataset_i,:)=mean(results_eg);   
end
results_mean(dataset_i+1,:)=mean(results_mean(1:dataset_i,:));


% results_mean=results_mean(:,[1:16 19:20]); % remove GLR


results_eigen_gap=results_mean;
results_time=results_mean(:,2:2:end);

datasize_order=zeros(n_dataset,1);
for dataset_i=1:17
    [dataset_str,read_data] = get_data_quiet(dataset_i);
    label=read_data(:,end);
    if dataset_i~=17 && dataset_i~=5 && dataset_i~=7
    K=5; % 5-fold
    elseif dataset_i==17
    K=1; 
    elseif dataset_i==5
    K=4;
    elseif dataset_i==7
    K=2;
    end
    rng(0);
    indices = crossvalind('Kfold',label,K); % K-fold cross-validation
    read_data_i=read_data(indices==1,:);
    datasize_order(dataset_i)=size(read_data_i,1);
end

[datasize_order_value,datasize_order_idx]=sort(datasize_order);
datasize_order_idx=[datasize_order_idx; n_dataset+1];

results_eigen_gap=results_eigen_gap(datasize_order_idx,:);

names = {'australian'; 'breast-cancer'; 'diabetes';...
    'fourclass'; 'german'; 'haberman';...
    'heart'; 'ILPD'; 'liver-disorders';...
    'monk1'; 'pima'; 'planning';...
    'voting'; 'WDBC'; 'sonar';...
    'madelon'; 'colon-cancer'; '\color{black}\bfavg.'};

names=names(datasize_order_idx);

ncolors = distinguishable_colors(n_method);
figure();hold on;

plot(results_eigen_gap,...
            'LineStyle','none',...
            'LineWidth',1,...
            'Marker','+',...
            'color',ncolors(1,:));
%             'color',ncolors(1,:),'DisplayName',num2str(method_name(1)));


ylabel('eigen-gap', 'FontSize', font_size);
set(gca,'fontname','times', 'FontSize', font_size) 
xlim([1 n_dataset+1]);
set(gca,'xtick',(1:n_dataset+1),'xticklabel',names);xtickangle(90);
ylim([min(vec(results_eigen_gap)) max(vec(results_eigen_gap))]);
grid on;
% legend;
title('supp Fig.3');