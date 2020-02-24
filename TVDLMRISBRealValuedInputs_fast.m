function [Iout1,param1] = TVDLMRISBRealValuedInputs_fast(I1,Q1,sigma,sigmai,DLMRIparams)

%Function for reconstructing real-valued image from simulated noisy undersampled k-space data.
% Inputs -
%       1. I1 : Input MR Image (real valued)
%       2. Q1 : Sampling Mask (for 2D DFT data) with zeros at non-sampled locations and ones at sampled locations
%       3. sigma : simulated noise level (standard deviation of simulated noise - added during simulation)
%       4. sigmai : Noise level (standard deviation of complex noise) in the DFT space of the peak-normalized input image.
%                   To be set to 0 if input image data is noiseless.
%       5. DLMRIparams: Structure that contains the parameters of the DLMRI algorithm. The various fields are as follows - 
%                   - num: Number of iterations of the DLMRI algorithm (required input. Example: about 15)
%                   - n: Patch size (i.e., Total # of pixels in square patch)
%                   - K2: Number of dictionary atoms
%                   - N: Number of signals used for training
%                   - T0: Sparsity settings of patch
%                   - Lambda: Sets the weight \nu in the algorithm
%                   - KSVDopt: If set to 1, K-SVD learning is done with fixed sparsity. For any other setting, K-SVD learning is done
%                              employing both sparsity level and an error threshold.
%                   - th: Threshold used in sparse representation of patches after training. To use threshold during training, set KSVDopt.
%                   - numiterateKSVD: Number of iterations within the K-SVD algorithm.
%                   - r: Overlap Stride
%
%    Note that all the above parameters except num get set to default values if not included in the input. 
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
    T0=round((0.15)*n);    %default
else
    T0=DLMRIparams.T0;
end
%error threshold for patches - allows error of (th^2)*n per patch.
if (~isfield(DLMRIparams,'th'))
    sig=0.02;  %Used for default error threshold
    C2=1.15;   %used for computing default error threshold
    th=C2*sig; % default error threshold
else
    th=DLMRIparams.th;
end
%Type of K-SVD learning
if (~isfield(DLMRIparams,'KSVDopt'))
    op=1; %default
else
    op=DLMRIparams.KSVDopt;
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
I1=I1/(max(max(I1))); %Normalize input image
[aa,bb]=size(I1);     %Compute size of image

DZ=((sigma2/sqrt(2))*(randn(aa,bb)+(0+1i)*randn(aa,bb)));  %simulating noise
I5=fft2(I1);          %FFT of input image
I5=I5+DZ;             %add measurement noise in k-space (simulated noise)

%Compute Input PSNR after adding noise
IG=abs(ifft2(I5));
InputPSNR=20*log10((sqrt(aa*bb))*(max(max(abs(IG))))/norm(double(abs(IG))-double(I1),'fro'));
param1.InputPSNR=InputPSNR;

index=find(Q1==1); %Index the sampled locations in sampling mask

I2=(double(I5)).*(Q1);  %Apply mask in DFT domain
I11=ifft2(I2);          % Inverse FFT - gives zero-filled result

%initializing simulation metrics
PSNR3=zeros(num,1);itererror=zeros(num,1);highfritererror=zeros(num,1);PSNR1=zeros(num,1);
figure(133);imagesc(abs(I11));colormap(gray);axis off; axis equal;
I11=I11/(max(max(abs(I11))));  %image normalization
figure(133);imagesc(fftshift(abs(I11)));colormap(gray);axis off; axis equal;

PSNR0=20*log10(sqrt(aa*bb)/norm(double(abs(I11))-double(I1),'fro')); %PSNR of normalized zero-filled reconstruction
param1.PSNR0=PSNR0;
% finite diff
sizeF = size(I11);
C.eigsDtD = abs(psf2otf([1,-1],sizeF)).^2 + abs(psf2otf([1;-1],sizeF)).^2;
TVD = @(U) ForwardD(U);
TVDt = @(X,Y) Dive(X,Y);
mu2 = DLMRIparams.mu2;
bregc1 = zeros(sizeF);
bregc2 = bregc1;

%DLMRI iterations
for kp=1:num    
    I11=abs(I11); % I11=I11/(max(max(I11)));
    Iiter=I11;    
    % finite diff
    [D1X,D2X] = TVD(I11);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Creating image patches    
    [blocks,idx] = my_im2col(D1X,[sqrt(n),sqrt(n)],r); br=mean(blocks); %image patches
    br2 = (ones(n,1))*br;
    TE=blocks-br2;           %subtract means of patches
    [rows,cols] = ind2sub(size(D1X)-sqrt(n)+1,idx);    
    N2=size(blocks,2); %total number of overlapping image patches
    de=randperm(N2);    
    %Check if specified number of training signals is less or greater than the available number of patches.
    if(N2>N)
        N4=N;
    else
        N4=N2;
    end    
    YH=TE(:,de(1:N4));   %Training data - using random selection/subset of patches    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %dictionary initialization : PCA + random training patches
    [UU,SS,VV]=svd(YH*YH');
    D0=zeros(n,K2);
    [hh,jj]=size(UU);
    D0(:,1:jj)=UU;
    p1=randperm(N4);
    for py=jj+1:K2
        D0(:,py)=YH(:,p1(py-jj));
    end
    param.initialDictionary=D0;   %initial dictionary for K-SVD algorithm    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %K-SVD algorithm - two versions below    
    if(op==1)
    %[D,output] = KSVD(YH,param);    %K-SVD algorithm with sparsity only
    %%%%%%%%%%%%%%%%%
    msgdelta = 5;verbose = 't';
    param.codemode = 'sparsity';  %'error';    
    param.Tdata = param.L;
    param.Edata = sqrt(n)*th;   % target error for omp
    param.data = YH;
    param.initdict=param.initialDictionary;
    param.dictsize = param.K;
    param.iternum = param.numIteration;
    [D,output.CoefMatrix] = ksvd(param,verbose,msgdelta);
    %%%%%%%%%%%%%%%%%
    else
    [D,output] = KSVDC2(YH,param,th);  %K-SVD algorithm with both sparsity and error threshold (FASTER!)
    end    
    DLerror=(norm(YH -(D*output.CoefMatrix),'fro'))^2;  %dictionary fitting error
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    %Computing sparse representations of all patches and summing up the patch approximations
    Weight= zeros(aa,bb);
    IMout = zeros(aa,bb); bbb=sqrt(n);
    for jj = 1:10000:size(blocks,2)
        jumpSize = min(jj+10000-1,size(blocks,2)); ZZ=TE(:,jj:jumpSize);
        %Coefs = OMPerrn(D,ZZ,th,2*param.L);   %sparse coding of patches
        %%%%%%%%%%%%%%%%%
        G = []; MEM_LOW = 1;MEM_NORMAL = 2;MEM_HIGH = 3;
        params.memusage = 'high';memusage = MEM_HIGH;
        if (memusage >= MEM_NORMAL)
            G = D'*D;
        end
        if (isempty(G))
            atomnorms = sum(D.*D);
        else
            atomnorms = diag(G);
        end
        if (any(abs(atomnorms-1) > 1e-2))
            error('Dictionary columns must be normalized to unit length');
        end
        Coefs = omp2(D'*ZZ,sum(ZZ.*ZZ),G,sqrt(n)*th,'maxatoms',2*param.L,'checkdict','off');
        %%%%%%%%%%%%%%%%%
        ZZ= D*Coefs + (ones(size(blocks,1),1) * br(jj:jumpSize)); %sparse approximations of patches        
        %summing up patch approximations
        for i  = jj:jumpSize
            col = cols(i); row = rows(i);
            block =reshape(ZZ(:,i-jj+1),[bbb,bbb]);
            IMout(row:row+bbb-1,col:col+bbb-1)=IMout(row:row+bbb-1,col:col+bbb-1)+block;
            Weight(row:row+bbb-1,col:col+bbb-1)=Weight(row:row+bbb-1,col:col+bbb-1)+ones(bbb);
        end;
    end    
    I3n1 = IMout./Weight;
    I3n1 = (I3n1+mu2*(D1X + bregc1))/(1+mu2);  %patch-averaged result
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Creating image patches    
    [blocks,idx] = my_im2col(D2X,[sqrt(n),sqrt(n)],r); br=mean(blocks); %image patches
    br2 = (ones(n,1))*br;
    TE=blocks-br2;           %subtract means of patches
    [rows,cols] = ind2sub(size(D2X)-sqrt(n)+1,idx);    
    N2=size(blocks,2); %total number of overlapping image patches
    de=randperm(N2);    
    %Check if specified number of training signals is less or greater than the available number of patches.
    if(N2>N)
        N4=N;
    else
        N4=N2;
    end    
    YH=TE(:,de(1:N4));   %Training data - using random selection/subset of patches    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %dictionary initialization : PCA + random training patches
    [UU,SS,VV]=svd(YH*YH');
    D0=zeros(n,K2);
    [hh,jj]=size(UU);
    D0(:,1:jj)=UU;
    p1=randperm(N4);
    for py=jj+1:K2
        D0(:,py)=YH(:,p1(py-jj));
    end
    param.initialDictionary=D0;   %initial dictionary for K-SVD algorithm    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %K-SVD algorithm - two versions below    
    if(op==1)
        %[D,output] = KSVD(YH,param);    %K-SVD algorithm with sparsity only
        %%%%%%%%%%%%%%%%%
        msgdelta = 5;verbose = 't';
        param.codemode = 'sparsity';  %'error';
        param.Tdata = param.L;
        param.Edata = sqrt(n)*th;   % target error for omp
        param.data = YH;
        param.initdict=param.initialDictionary;
        param.dictsize = param.K;
        param.iternum = param.numIteration;
        [D,output.CoefMatrix] = ksvd(param,verbose,msgdelta);
        %%%%%%%%%%%%%%%%%
    else
    [D,output] = KSVDC2(YH,param,th);  %K-SVD algorithm with both sparsity and error threshold (FASTER!)
    end    
    DLerror=(norm(YH -(D*output.CoefMatrix),'fro'))^2;  %dictionary fitting error
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    %Computing sparse representations of all patches and summing up the patch approximations
    Weight= zeros(aa,bb);
    IMout = zeros(aa,bb); bbb=sqrt(n);
    for jj = 1:10000:size(blocks,2)
        jumpSize = min(jj+10000-1,size(blocks,2)); ZZ=TE(:,jj:jumpSize);
        %Coefs = OMPerrn(D,ZZ,th,2*param.L);   %sparse coding of patches
        %%%%%%%%%%%%%%%%%
        G = []; MEM_LOW = 1;MEM_NORMAL = 2;MEM_HIGH = 3;
        params.memusage = 'high';memusage = MEM_HIGH;
        if (memusage >= MEM_NORMAL)
            G = D'*D;
        end
        if (isempty(G))
            atomnorms = sum(D.*D);
        else
            atomnorms = diag(G);
        end
        if (any(abs(atomnorms-1) > 1e-2))
            error('Dictionary columns must be normalized to unit length');
        end
        Coefs = omp2(D'*ZZ,sum(ZZ.*ZZ),G,sqrt(n)*th,'maxatoms',2*param.L,'checkdict','off');
        %%%%%%%%%%%%%%%%%
        ZZ= D*Coefs + (ones(size(blocks,1),1) * br(jj:jumpSize)); %sparse approximations of patches        
        %summing up patch approximations
        for i  = jj:jumpSize
            col = cols(i); row = rows(i);
            block =reshape(ZZ(:,i-jj+1),[bbb,bbb]);
            IMout(row:row+bbb-1,col:col+bbb-1)=IMout(row:row+bbb-1,col:col+bbb-1)+block;
            Weight(row:row+bbb-1,col:col+bbb-1)=Weight(row:row+bbb-1,col:col+bbb-1)+ones(bbb);
        end;
    end    
    I3n2 = IMout./Weight;
    I3n2 = (I3n2+mu2*(D2X + bregc2))/(1+mu2);  %patch-averaged result
    I3n = ifft2(fft2(TVDt(I3n1-bregc1,I3n2-bregc2))./(C.eigsDtD+eps));
    inn = abs(I3n)>1;I3n(inn)=1; 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    I2=fft2(I3n);  %Move from image domain to k-space    
    %K-space update formula
    if(sigma2<1e-10)
        I2(index)=I5(index);
    else
        I2(index)= (1./(mu2*C.eigsDtD(index)+La2)).*(mu2*C.eigsDtD(index).*I2(index) + La2*I5(index));
    end 
    I11=ifft2(I2);   %Use Inverse FFT to get back to image domain
    inn2= abs(I11)>1;I11(inn2)=1;
    figure(133);imagesc(abs(I11));colormap(gray);axis off; axis equal;
    % ==================%    Update % ==================
    bregc1 = bregc1 + (D1X - I3n1);
    bregc2 = bregc2 + (D2X - I3n2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    %Compute various performance metrics
    PSNR3(kp)=20*log10(sqrt(aa*bb)*255/norm(double(abs(I3n))-double(I1),'fro')) -(20*log10(255)); %PSNR of patch-averaged result
    itererror(kp)= norm(abs(Iiter) - abs(I11),'fro');
    highfritererror(kp)=norm(imfilter(abs(I11),fspecial('log',15,1.5)) - imfilter(I1,fspecial('log',15,1.5)),'fro');
    PSNR1(kp)=20*log10(sqrt(aa*bb)*255/norm(double(abs(I11))-double(I1),'fro')) -(20*log10(255))   
end

Iout1=abs(I11);
param1.PSNR=PSNR1;
param1.HFEN=highfritererror;
param1.itererror=itererror;
param1.Dictionary=D;
