%% test file of canny algorithm.
close all;
clear all;
clc;
%%
img = imread('./picture/lena.jpg');
% img = imnoise(img,'salt & pepper',0.01);
%%
[~,~,dim] = size(img);
if dim > 1
    imgCanny1 = edge(rgb2gray(img),'canny'); % system function.
else
    imgCanny1 = edge(img,'canny'); % system function.
end
imgCanny2 = edge_canny(img,[5,5],1.4,0.9,0.4); % my function.
%%
figure;
subplot(2,2,1);
imshow(img);
title('original');
%%
subplot(2,2,3);
imshow(imgCanny1);
title('system function');
%%
subplot(2,2,4);
imshow(imgCanny2);
title('my function');