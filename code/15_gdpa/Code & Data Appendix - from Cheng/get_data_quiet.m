function [dataset_str,read_data] = get_data_quiet(dataset_i)
%GET_DATA 此处显示有关此函数的摘要
%   此处显示详细说明
addpath('datasets\');
% dataset_i = eval(input('please enter number 1-17 (# of the above datasets) to run: ', 's'));

if dataset_i==1
    read_data = importdata('australian.csv');
    dataset_str = 'australian';
elseif dataset_i==2
    read_data = importdata('breast-cancer.csv');
    dataset_str = 'breast-cancer';
elseif dataset_i==3
    read_data = importdata('diabetes.csv');
    dataset_str = 'diabetes';
elseif dataset_i==4
    read_data = importdata('fourclass.csv');
    dataset_str = 'fourclass';
elseif dataset_i==5
    read_data = importdata('german.csv');
    dataset_str = 'german';
elseif dataset_i==6
    read_data = importdata('haberman.csv');
    dataset_str = 'haberman';
elseif dataset_i==7
    read_data = importdata('heart.dat');
    dataset_str = 'heart';
elseif dataset_i==8
    read_data = importdata('Indian Liver Patient Dataset (ILPD).csv');
    dataset_str = 'ILPD';
elseif dataset_i==9
    read_data = importdata('liver-disorders.csv');
    dataset_str = 'liver-disorders';
elseif dataset_i==10
    read_data = importdata('monk1.csv');
    dataset_str = 'monk1';
elseif dataset_i==11
    read_data = importdata('pima.csv');
    dataset_str = 'pima';
elseif dataset_i==12
    read_data = importdata('planning.csv');
    dataset_str = 'planning';
elseif dataset_i==13
    read_data = importdata('voting.csv');
    dataset_str = 'voting';
elseif dataset_i==14
    read_data = importdata('WDBC.csv');
    dataset_str = 'WDBC';
elseif dataset_i==15
    read_data = importdata('sonar.csv');
    dataset_str = 'sonar';
elseif dataset_i==16
    read_data = importdata('madelon.csv');
    dataset_str = 'madelon';
elseif dataset_i==17
    read_data = importdata('colon-cancer.csv');
    dataset_str = 'colon-cancer';
end
end

