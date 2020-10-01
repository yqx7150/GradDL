%% The Code is created based on the methods described in the following papers: 
%   [1] Qiegen Liu, Shanshan Wang, Leslie Ying, Xi Peng, Yanjie Zhu, Dong Liang. Adaptive dictionary learning in sparse gradient domain for image recovery, 
%       IEEE Transactions on Image Processing, 2013, 22(12): 4652-4663.
%   Author: Qiegen Liu, Shanshan Wang, Leslie Ying, Xi Peng, Yanjie Zhu, Dong Liang
%   Date  : 10/25/2013
%   Version : 1.0 
%   The code and the algorithm are for non-comercial use only.
%   Copyright 2013, Department of Electronic Information Engineering, Nanchang University.
%   The current version is not optimized.

% All rights reserved.
% This work should only be used for nonprofit purposes.
%
% Please cite the paper when you use th code

clear all; close all; clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
getd = @(p)path(path,p);% Add some directories to the path
getd('ksvdbox13\');getd('ompbox10\');

%#######%%%%% testing image %%%%
M0 = imread('ortho1.jpg');  %n =  250;   M0 = phantom(n);
M0 = im2double(M0);
if (length(size(M0))>2);  M0 = rgb2gray(M0);   end
if (max(M0(:))<2);   M0 = M0*255;    end
figure(456); imshow(M0,[]);
[min(M0(:)),max(M0(:))]
%#######%%%%% generate the sampling trajectory %%%%%%%%%  
n =size(M0,1);
angles = 75; %
fprintf(1,'Generating beam mask with %d angles\n',angles)
Q1 = symmMap(n,angles);
fprintf(1,'Done\n');
k = sum(sum(Q1));
fprintf(1, 'n=%d, k=%d, Unsamped=%f\n', n, k,1-k/n/n);
figure(455); imshow(Q1,[]);  %mask at the central

%%%%%%%%%%%%% GradDL reconstruction %%%%%%%%
I1 = M0;
Q1 = fftshift(Q1);
sigma = 0;sigmai = 0;   %
DLMRIparams.num = 25;  %
DLMRIparams.n = 36;
DLMRIparams.K2 = 1*DLMRIparams.n;
DLMRIparams.N = 200*DLMRIparams.K2;
DLMRIparams.T0 =  floor(0.15*DLMRIparams.n);
DLMRIparams.Lambda = 140;
DLMRIparams.KSVDopt = 1;
%DLMRIparams.th = 0;
DLMRIparams.numiterateKSVD = 10;    %
DLMRIparams.r = 1;
DLMRIparams.mu2 = 3;
[Iout1,param1]=TVDLMRISBRealValuedInputs_fast(I1,Q1,sigma,sigmai,DLMRIparams);
[param1.InputPSNR ,param1.PSNR0,param1.PSNR(end)]
param1.Im = Iout1;


