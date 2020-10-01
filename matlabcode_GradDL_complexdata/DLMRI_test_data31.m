clear all; 
close all;
addpath('./Complexdata_ulities/');
addpath('./quality_assess/');
addpath('./test_data_31/');
%%
sigma = 0;   sigmai = 0;   %
DLMRIparams.num = 25;  %10;
DLMRIparams.n = 36;
DLMRIparams.K2 = DLMRIparams.n;
DLMRIparams.N = 200*DLMRIparams.K2;
DLMRIparams.Lambda = 140;
DLMRIparams.KSVDopt = 1;
DLMRIparams.numiterateKSVD = 10;    %
DLMRIparams.r = 1;

mask_names={'mask_radial70'};
i=1;
for maskname = mask_names
    load(maskname{1,1}) ; mask = eval(maskname{1,1});
    disp(maskname{1,1})
    n = size(mask,2);
%     mask = fftshift(mask);
    fprintf(1, 'n=%d, k=%d, Unsamped=%f\n', n, sum(sum(mask)),1-sum(sum(mask))/n/n);
    for ii = 27:27  %31
        load(['test_data_31/test_data_',num2str(ii,'%02d') ,'.mat']);
        image = Img./max(abs(Img(:)));
        tic;
        [reconImg,param1]=DLMRIComplexValuedInputs(image*255,mask,sigma,sigmai,DLMRIparams);
        toc
        [psnr4, ssim4, ~] = MSIQA(abs(reconImg)*255, abs(image)*255);
        hfen = norm(imfilter(abs(reconImg),fspecial('log',15,1.5)) ...
            - imfilter(abs(image),fspecial('log',15,1.5)),'fro');
        result(i).name = cat(2,'test_data_',num2str(ii,'%02d') ,'_',maskname{1,1});
        result(i).map_deblur = reconImg*255;
        result(i).psnr = psnr4;
        result(i).ssim = ssim4;
        result(i).hfen = hfen;
        i = i+1;
        save(['result/result_',maskname{1,1} ,'_test_data_',num2str(ii,'%02d') ],'result');
    end
end
% tt2 = abs(abs(image)-abs(gt));
% h = max(tt2(:));
% figure(i);imshow(tt2,[]);i=i+1;
% caxis([0,h]);colormap(jet);colorbar;axis off; axis equal;