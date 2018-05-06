% Inspect train & test images
% Cathey Wang
% 04/25/2018

trainDataDir = '../data/train/';
testDataDir = '../data/test/';

ls = dir(trainDataDir);
ls = ls(3:end);
FF = length(ls);

trainFiles = cell(FF,1);
for i = 1:FF
    fn = ls(i).name;
    trainFiles{i} = [trainDataDir, fn];
end

dims = zeros(FF, 3);
for i = 1:FF
    img = imread(trainFiles{i});
    [y,x,c] = size(img);
    dims(i, :) = [y,x,c];
end

[mindims, minidx] = min(dims, [], 1);
[maxdims, maxidx] = max(dims, [], 1);

labelFile = '../data/train.csv';
fid = fopen(labelFile);
C = textscan(fid, '%s%s', 'delimiter', ',');
fclose(fid);

fileNames = C{1};
whaleIDs = C{2};
fileNames = fileNames(2:end);
whaleIDs = whaleIDs(2:end);

[uniqueIDs,i_whale,i_unique] = unique(whaleIDs);
uniqueIDs_counts = accumarray(i_unique,1);

