function [Iout1,param1] = GradDLComplexValuedInputs(I1,Q1,sigma,sigmai,DLMRIparams)

%Function for reconstructing MR image from undersampled noisy k-space data. Given: Complex MR image to use for simulation.
% Inputs -
%       1. I1 : Input Complex MR Image - for instance obtained by inverse FFT reconstruction from raw fully sampled k-space data.
%       2. Q1 : Sampling Mask (for 2D DFT space) with zeros at non-sampled locations and ones at sampled locations
%       3. sigma : Simulated noise level (standard deviation of simulated k-space noise, if any). To be set to 0 if no new noise is to be added.
%       4. sigmai : Noise level (standard deviation of complex noise) in the DFT space of the peak-normalized input image.
%                   To be set to 0 if input image data is noiseless.
%       5. DLMRIparams: Structure that contains the parameters of the DLMRI algorithm. The various fields are as follows - 
%                   - num: Number of iterations of the DLMRI algorithm (required input. Example: 20-100)
%                   - n: Patch size (i.e., Total # of pixels in square patch)
%                   - K2: Number of dictionary atoms
%                   - N: Number of signals used for training
%                   - T0: Sparsity settings of patch
%                   - Lambda: Sets the weight \nu in the algorithm
%                   - thr: Thresholds used in sparse representation of patches during/after training. Must be a vector of same size 
%                          as the number of DLMRI iterations. Each element in the vector is the threshold for the corresponding iteration number.
%                   - numiterateKSVD: Number of iterations within the K-SVD algorithm.
%                   - r: Overlap Stride
%
%    Note that all the above parameters except 'num' get set to default values if not included in the input. 
%    K-SVD algorithm is initialized as mentioned in the paper.
%
% Outputs -
%       1. Iout1 - Image reconstructed with DLMRI algorithm from undersampled data.
%       3. param1 - Structure containing various parameter values/performance metrics from simulation for DLMRI.
%                 - InputPSNR : PSNR of fully sampled noisy reconstruction
%                 - PSNR0 : PSNR of normalized zero-filled reconstruction
%                 - PSNR : PSNR of the reconstruction at each iteration of the DLMRI algorithm
%                 - HFEN : HFEN of the reconstruction at each iteration of the DLMRI algorithm
%                 - itererror : norm of the difference between the reconstructions at successive iterations
%                 - Dictionary : Final DLMRI dictionary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%DLMRI Parameters initialization

sigma2=sqrt((sigmai^2)+(sigma^2));  %Effective k-space noise level 

num=DLMRIparams.num;  %DLMRI algorithm iteration count

%Lambda parameter
if (~isfield(DLMRIparams,'Lambda'))
    Lambda=140; %default
else
    Lambda=DLMRIparams.Lambda;
end

La2=(Lambda)/(sigma2); % \nu weighting of paper

%Overlap Stride
if (~isfield(DLMRIparams,'r'))
    r=1;    %default
else
    r=DLMRIparams.r;
end

%patch size
if (~isfield(DLMRIparams,'n'))
    n=36;    %default
else
    n=DLMRIparams.n;
end

%number of dictionary atoms
if (~isfield(DLMRIparams,'K2'))
    K2=1*n;    %default
else
    K2=DLMRIparams.K2;
end

%number of training signals
if (~isfield(DLMRIparams,'N'))
    N=200*K2;    %default
else
    N=DLMRIparams.N;
end

%sparsity levels
if (~isfield(DLMRIparams,'T0'))
    T0=round((0.2)*n);    %default
else
    T0=DLMRIparams.T0;
end

%error threshold for patches - allows squared error of (threshi^2)*n per patch.
if (~isfield(DLMRIparams,'thr'))
    sig=0.02; C2=1.15;  %Used for default error thresholds
    th=C2*sig; % used for computing default error thresholds
    tupr=[2 2 2 2 1.4*ones(1,num-4)]; %used for default error thresholds
    threshi=(th)*tupr;  %default error threshold vector
else
    threshi=DLMRIparams.thr;
end

%number of K-SVD iterations
if (~isfield(DLMRIparams,'numiterateKSVD'))
    numiterKSVD = 20;    %default
else
    numiterKSVD=DLMRIparams.numiterateKSVD;
end

%other parameters of K-SVD algorithm
param.errorFlag=0;
param.L=T0;
param.K=K2;
param.numIteration=numiterKSVD;
param.preserveDCAtom=0;
param.InitializationMethod='GivenMatrix';param.displayProgress=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%MAIN CODE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

I1=double(I1(:,:,1));
I1=I1/(max(max((abs(I1))))); %Normalize input image
[aa,bb]=size(I1);            %Compute size of image

DZ=((sigma/sqrt(2))*(randn(aa,bb)+(0+1i)*randn(aa,bb)));  %simulating noise
I5=fft2(I1);          %FFT of input image
I5=I5+DZ;             %add measurement noise in k-space

%Compute Input PSNR after adding noise
IG=abs(ifft2(I5));
InputPSNR=20*log10((sqrt(aa*bb))*(max(max(abs(IG))))/norm(double(abs(IG))-double(abs(I1)),'fro'));
param1.InputPSNR=InputPSNR;

index=find(Q1==1); %Index the sampled locations in sampling mask

I2=(double(I5)).*(Q1);  %Apply mask in DFT domain
I11=ifft2(I2);          % Inverse FFT - gives zero-filled result
I11p=I11;
PSNR0=20*log10(sqrt(aa*bb)/norm(double(abs(I11p))-double(abs(I1)),'fro')); %PSNR of zero-filled reconstruction
param1.PSNR0=PSNR0;
figure(133);imagesc(abs(I11));colormap(gray);axis off; axis equal;
%initializing simulation metrics
PSNR3=zeros(num,1);itererror=zeros(num,1);highfritererror=zeros(num,1);PSNR1=zeros(num,1);

%DLMRI iterations
for kp=1:num
    
    I11=(I11);
    I11=I11/(max(max((abs(I11)))));
    Iiter=I11;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    im_size = size(I11);
    [k2,k1] = meshgrid(0:im_size(2)-1,0:im_size(1)-1);
    fdx = 1 - exp(-2*pi*1j*k1/im_size(1));
    fdy = 1 - exp(-2*pi*1j*k2/im_size(2));
    Img2(:,:,1) = ifft2(fft2(real(I11)).*fdx);
    Img2(:,:,2) = ifft2(fft2(imag(I11)).*fdx);
    Img2(:,:,3) = ifft2(fft2(real(I11)).*fdy);
    Img2(:,:,4) = ifft2(fft2(imag(I11)).*fdy);
    I11dx = Img2(:,:,1) + 1i.*Img2(:,:,2);
    I11dy = Img2(:,:,3) + 1i.*Img2(:,:,4);
    %I11dxdy = [I11dx,I11dy];
    
    %Creating image patches    
    [blocks,idx] = my_im2col(I11dx,[sqrt(n),sqrt(n)],r); br=mean(blocks); %image patches
    TE=blocks-(ones(n,1)*br); %subtract means of patches
    [rows,cols] = ind2sub(size(I11dx)-sqrt(n)+1,idx);    
    N2=size(blocks,2);  %total number of overlapping image patches
    de=randperm(N2);    
    %Check if specified number of training signals is less or greater than the available number of patches.
    if(N2>N)   N4=N;  else  N4=N2; end    
    YH=TE(:,de(1:N4)); %Training data - using random subset of all patches    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
    %dictionary initialization : PCA + random training patches
    [UH,SH,VH]=svd(YH*YH');
    D0=zeros(n,K2);
    [hh,jj]=size(UH);
    if(K2>=n)
        D0(:,1:jj)=UH;
        p1=randperm(N4);
        for py=jj+1:K2
            D0(:,py)=YH(:,p1(py-jj));
        end
    else
        D0(:,1:K2)=UH(:,1:K2);
    end
    param.initialDictionary=D0; %initial dictionary for K-SVD algorithm
    YHU=YH;    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %K-SVD algorithm    %[D,output] = KSVDC(YHU,param); %K-SVD with sparsity threshold only
    [D,output] = KSVDC2(YHU,param,threshi(kp)); %K-SVD with both sparsity and error threshold (Best)    
    DLerror=(norm(YHU -(D*output.CoefMatrix),'fro'))^2;  %dictionary fitting error    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Computing sparse representations of all patches and summing up the patch approximations
    Weight= zeros(aa,bb);
    IMout = zeros(aa,bb); bbb=sqrt(n);
    for jj = 1:10000:size(blocks,2)
        jumpSize = min(jj+10000-1,size(blocks,2)); ZZ=TE(:,jj:jumpSize);
        Coefs = OMPerrn(D,ZZ,threshi(kp),2*param.L);  %sparse coding of patches
        ZZ= D*Coefs + ones(size(blocks,1),1) * br(jj:jumpSize); %sparse approximations of patches        
        %summing up patch approximations
        for i  = jj:jumpSize
            col = cols(i); row = rows(i);
            block =reshape(ZZ(:,i-jj+1),[bbb,bbb]);
            IMout(row:row+bbb-1,col:col+bbb-1)=IMout(row:row+bbb-1,col:col+bbb-1)+block;
            Weight(row:row+bbb-1,col:col+bbb-1)=Weight(row:row+bbb-1,col:col+bbb-1)+ones(bbb);
        end;
    end    
    I3ndx=IMout./Weight; %patch-averaged result
    
    %Creating image patches    
    [blocks,idx] = my_im2col(I11dy,[sqrt(n),sqrt(n)],r); br=mean(blocks); %image patches
    TE=blocks-(ones(n,1)*br); %subtract means of patches
    [rows,cols] = ind2sub(size(I11dy)-sqrt(n)+1,idx);    
    N2=size(blocks,2);  %total number of overlapping image patches
    de=randperm(N2);    
    %Check if specified number of training signals is less or greater than the available number of patches.
    if(N2>N)   N4=N;  else  N4=N2; end    
    YH=TE(:,de(1:N4)); %Training data - using random subset of all patches    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
    %dictionary initialization : PCA + random training patches
    [UH,SH,VH]=svd(YH*YH');
    D0=zeros(n,K2);
    [hh,jj]=size(UH);
    if(K2>=n)
        D0(:,1:jj)=UH;
        p1=randperm(N4);
        for py=jj+1:K2
            D0(:,py)=YH(:,p1(py-jj));
        end
    else
        D0(:,1:K2)=UH(:,1:K2);
    end
    param.initialDictionary=D0; %initial dictionary for K-SVD algorithm
    YHU=YH;    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %K-SVD algorithm    %[D,output] = KSVDC(YHU,param); %K-SVD with sparsity threshold only
    [D,output] = KSVDC2(YHU,param,threshi(kp)); %K-SVD with both sparsity and error threshold (Best)    
    DLerror=(norm(YHU -(D*output.CoefMatrix),'fro'))^2;  %dictionary fitting error    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Computing sparse representations of all patches and summing up the patch approximations
    Weight= zeros(aa,bb);
    IMout = zeros(aa,bb); bbb=sqrt(n);
    for jj = 1:10000:size(blocks,2)
        jumpSize = min(jj+10000-1,size(blocks,2)); ZZ=TE(:,jj:jumpSize);
        Coefs = OMPerrn(D,ZZ,threshi(kp),2*param.L);  %sparse coding of patches
        ZZ= D*Coefs + ones(size(blocks,1),1) * br(jj:jumpSize); %sparse approximations of patches        
        %summing up patch approximations
        for i  = jj:jumpSize
            col = cols(i); row = rows(i);
            block =reshape(ZZ(:,i-jj+1),[bbb,bbb]);
            IMout(row:row+bbb-1,col:col+bbb-1)=IMout(row:row+bbb-1,col:col+bbb-1)+block;
            Weight(row:row+bbb-1,col:col+bbb-1)=Weight(row:row+bbb-1,col:col+bbb-1)+ones(bbb);
        end;
    end    
    I3ndy=IMout./Weight; %patch-averaged result
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     I2=fft2(I3n);   %Move from image domain to k-space    
%     if(sigma2<1e-12)%K-space update formula
%         I2(index)=I5(index);
%     else
%         I2(index)= (1/(1+(La2)))*(I2(index) + (La2)*I5(index));
%     end    
%     I11=ifft2(I2);   %Use Inverse FFT to get back to image domain
    I11 = L2_image_from_edges_rect(I2, I3ndx, I3ndy, 0);
    
    figure(133);imagesc(abs(I11));colormap(gray);axis off; axis equal;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Compute various performance metrics
    %PSNR3(kp)=20*log10(sqrt(aa*bb)*255/norm(double(abs(I3n))-double(abs(I1)),'fro')) -(20*log10(255)); %PSNR of patch-averaged result
    itererror(kp)= norm(abs(Iiter) - abs(I11),'fro');
    highfritererror(kp)=norm(imfilter(abs(I11),fspecial('log',15,1.5)) - imfilter(abs(I1),fspecial('log',15,1.5)),'fro');
    PSNR1(kp)=20*log10(sqrt(aa*bb)*255/norm(double(abs(I11))-double(abs(I1)),'fro')) -(20*log10(255))
    
end

Iout1=abs(I11);  %output Image magnitude
param1.PSNR=PSNR1;
param1.HFEN=highfritererror;
param1.itererror=itererror;
param1.Dictionary=D;
