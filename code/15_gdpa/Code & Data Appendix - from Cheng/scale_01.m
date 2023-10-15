function [featureout] = scale_01(featurein)
%SCALE_01 Summary of this function goes here
%   Detailed explanation goes here
featureout=featurein;
[n,m]=size(featurein);
for i=1:m
    featureout(:,i)=0+(featureout(:,i)-min(featureout(:,i)))/(max(featureout(:,i))-min(featureout(:,i)));
end
end

