%==========================================================================
% This script compares multi-spectral imagery (MSI) noise removal methods
% 
% Four quality assessment (QA) indices -- PSNR, SSIM, FSIM, ERGAS,SAM
% -- are calculated for each methods after denoising.
%
% 
%==========================================================================

clear all;close all;
addpath(genpath('Utilize\'));
import py.model.*
addpath(genpath('lib'));
dataname = 'test_center.mat';%  Please make sure this MSI is of size height x width x nbands and in range [0, 1].
                       %  can also use 'testMSI_1' and 'testMSI_1', as other examples.
% dataRoad = ['C:\\Users\\机械革命\\Desktop\\高光谱\\代码\\' dataname];
dataRoad = ['C:\\Users\\mazi\\Desktop\\matlab code\\' dataname];
saveroad = ['result/result_for_' dataname];
% if isempty(gcp)
%     parpool(3,'IdleTimeout', inf); % If your computer's memory is less than 8G, do not use more than 4 workers.
% end 
%% Set enable bits
sigma_ratio =25/255.0;  % higher sigma_ratio <--> heavier Gaussian noise
memorySaving = 1;  
% Setting 'memorySaving = 0' : parall + no memory savings
% Setting 'memorySaving = 1' : light parall + memory savings
disp( '=== The variance of noise is 0.1 ===');
EN_ENCAM_S   = 1;
EN_ENCAM_B   = 0;
getGif      = 0; % if save gif result or not
getImage    = 0; % if save the appointed band of the reconstructed MSI as image result or not
mkdir(saveroad);
% rng(0);
randn('seed', 1);
%% initial Data
methodname ={'Nosiy','ENCAMS','ENCAMB'};
Mnum   = length(methodname);
disp(Mnum);
load(dataRoad); % load data
% center=center(1:300,1:300,:);
msi_sz  =  size(center);
[w,h, b] = size(center);
% [60,30,20]
band    = [17,27,7]; %the band to show and save
tem    = center(:,:,band);
maxI    = max(tem(:));
minI    = min(tem(:));

%% Add Gaussian noise
sigma     = sigma_ratio;     % sigma of Gaussian distribution
noisy_msi=zeros(w,h,b);
for i=1:b
%     
    noisy_msi(:, :, i)=center(:, :, i)+randn(size(center(:,:,i)))*sigma;
%    
end
% for i=1:b
% %     randi([15,25],1)/255.0
%     noisy_msi(:, :, i)=center(:, :, i);
%     noisy_msi(:, :, i)=imnoise(noisy_msi(:, :, i),'salt & pepper',randi([10,20],1)/100);
% end
% for bd=70:100
%     num = 19+randperm(21,1);
%     loc = ceil(h*rand(1,num));
%     t = rand(1,length(loc))*0.5-0.25;
%     noisy_msi(:,loc,bd) = bsxfun(@minus,noisy_msi(:,loc,bd),t);
% end
% for i=1:b
%     if mod(i,2)==0
%         noisy_msi(:, :, i) = center(:, :, i) +randi([20,30],1)/255.0*randn(size(center(:, :, i)));
%     
%     else
%         noisy_msi(:, :, i) = center(:, :, i);
%     end
% end

if getGif
    mkdir( [saveroad,'/GIF' ,]);
    togetGif(center,[saveroad,'/GIF/Clean_msi']);
end
if getGif;togetGif(noisy_msi,[saveroad,'/GIF/Noisy_msi']);end
if getImage
    mkdir( [saveroad,'/Image' ,]);
    imwrite(normalized(center(:,:,band)),[saveroad,'/Image/Clean_msi.png']);
end
if getImage; imwrite(noisy_msi(:,:,band),[saveroad,'/Image/Noisy_msi.png']);end

i  = 1;
Re_msi{i} = noisy_msi;
[psnr(i), ssim(i), fsim(i), ergas(i),sam(i)] = MSIQA(center * 255, Re_msi{i}  * 255);
enList = 1;


%% Use ENCAM-S method 
%This model is used to reduce noise at a fixed noise level
i = i+1;
if EN_ENCAM_S
    disp(['performing ',methodname{i}, ' ... ']);
    tic;
    noisy_msi= permute(noisy_msi,[3 2 1]);
    noisy_msi = noisy_msi(:)';
    model=py.model.model_25();%This shows the loading of the ENCAM model.
                              %Now it means loading the model with SIGMA 25 removed. 
                            %If you want to modify the model you want, For example the denoising level of 75.
                            %You can change py.model.model_25() to py.model.model_75()
    out=py.model.tests(noisy_msi,model);
    test=double(py.array.array('d',py.numpy.nditer(out)));
    Re_msi{i}=reshape(test,[w,h,b]);
    Time(i) = toc;
    centers=Re_msi{i};
    save('center_ENCAM','centers');
    [psnr(i), ssim(i), fsim(i), ergas(i),sam(i)] = MSIQA(center * 255, Re_msi{i}  * 255);
    disp([methodname{i}, ' done in ' num2str(Time(i)), ' s.'])
    disp([methodname{i}, ' PSNR ' num2str(psnr(i)), ' .'])
    disp('...')
    if getGif; togetGif(Re_msi{i},[saveroad, '/GIF/', methodname{i}]); end;
    if getImage; imwrite((Re_msi{i}(:,:,band)-minI)/(maxI-minI),[saveroad,'/Image/', methodname{i}, '.png']);end
    enList = [enList,i];
end
%% Use ENCAM-B method
%This model is used to reduce noise at non-fixed noise levels
i = i+1;
if EN_ENCAM_B
    disp(['performing ',methodname{i}, ' ... ']);
    tic;
    noisy_msi= permute(noisy_msi,[3 2 1]);
    noisy_msi = noisy_msi(:)';
    model=py.model.model_real();
    out=py.model.tests(noisy_msi,model);
    test=double(py.array.array('d',py.numpy.nditer(out)));
    Re_msi{i}=reshape(test,[w,h,b]);
    Time(i) = toc;
    [psnr(i), ssim(i), fsim(i), ergas(i),sam(i)] = MSIQA(center * 255, Re_msi{i}  * 255);
    disp([methodname{i}, ' done in ' num2str(Time(i)), ' s.'])
    disp([methodname{i}, ' PSNR ' num2str(psnr(i)), ' .'])
    disp('...')
    if getGif; togetGif(Re_msi{i},[saveroad, '/GIF/', methodname{i}]); end;
    if getImage; imwrite((Re_msi{i}(:,:,band)-minI)/(maxI-minI),[saveroad,'/Image/', methodname{i}, '.png']);end
    enList = [enList,i];
end

%% Show result
fprintf('\n');
fprintf('================== Result =====================\n');
fprintf(' %6.6s    %5.4s    %5.4s    %5.4s    %5.5s   %5.5s  \n','method','PSNR', 'SSIM', 'FSIM', ' ERGAS','SAM');
for i = 1:length(enList)
    fprintf(' %6.6s    %5.3f    %5.3f    %5.3f    %5.3f  %5.5f  \n',...
        methodname{enList(i)},psnr(enList(i)), ssim(enList(i)), fsim(enList(i)), ergas(enList(i)),sam(enList(i)));
end
fprintf('================== Result =====================\n');
close all;
numLine = ceil((length(enList)+1)/5);
if band ==1
    figureName = ['Result on the ',  num2str(band), 'st band'];
elseif band==2
    figureName = ['Result on the ',  num2str(band), 'nd band'];
else
    figureName = ['Result on the ',  num2str(band), 'th band'];
end
figure('units','normalized','position',[0.05,0.482-0.29*numLine/2,0.9,0.29*numLine],'name',figureName);
subplot(numLine,5,1); imshow((center(:,:,band)-minI)/(maxI-minI)),title( 'Clean');
for j = 1:length(enList)
    subplot(numLine,5,j+1);
    imshow((Re_msi{enList(j)}(:,:,band)-minI)/(maxI-minI));title( methodname{enList(j)});
end
save([saveroad,'\Result'], 'psnr','ssim','fsim','ergas','methodname','Re_msi');

delete(gcp)