%% DNN test 1
% Idea start from alexnet
% get features
% Train SVM to indentify each 227 x 227 pix. image
% into 1 of the 6 cat.

% So we don't use spatial localisation.
% Indeed low, coarse or downsampled convolution can be used to identify
% regions
% Adding normalized coordinates can help asswell.
%% Read label contours:
files=dir ('./meta/*.txt');
for i=1:length(files)
    ROI{i} = importdata(['meta/' files(i).name],'\t');
    roinames{i} = files(i).name;
end

figure;el=1;imagesc(poly2mask(ROI{el}(:,1),ROI{el}(:,2),30000,30000))

%% Or using lib to read roi files directly.
files=dir ('./meta/*.roi');
for i=1:length(files)
    roi{i} = ReadImageJROI(['meta/' files(i).name]);
    ROI{i} = roi{i}.mnCoordinates;
    roinames{i} = files(i).name;
end

%% Read the data:
f=imread('B21-1_c4_ORG.tif');
%% Store picture as matlab format file:
save('picture.mat','f','-v7.3'); % -v7.3 to support partial loading
%% map mat file
pict=matfile('picture.mat');
%%
figure;el=1;imagesc(poly2mask(ROI{el}(:,1),ROI{el}(:,2),30000,30000))

% Scale down image:
srcImage=f(1:64:size(f,1),(1:64:size(f,2)));
for i=1:length(roinames)
    ROI{i}=ROI{i}./1;
end

%% Plot contours
figure;
image(srcImage/3);
hold on;
for i=1:length(ROI)
    x=ROI{i}(:,1)/1.25;
    y=ROI{i}(:,2)/1.25;
    plot([x; x(1)],[y; y(1)]);
    hold on;
end


%% Create masks:
for i=1:length(ROI)
    mask{i}=poly2mask(ROI{i}(:,1)/1.25,ROI{i}(:,2)/1.25,size(f,1),size(f,2));
end
%%
save('mask.mat','mask','-v7.3'); % -v7.3 to support partial loading

%%
for i=1:length(files)
    vx{i}=ROI{i}(:,1)-circshift(ROI{i}(:,1),1);
    vy{i}=ROI{i}(:,2)-circshift(ROI{i}(:,2),1);
end
%% Rotate left or right:
for i=1:length(files)
    z=cross([vx vy vy*0] ,circshift([vx vy vy*0],1));
    sz{i}=sum(z);
    if sz{i}>0,
        left{i}=1
    else
        left{i} =0;
    end
end

%% Label:

[FX,FY] = meshgrid(1:size(srcImage,1), 1:size(srcImage,2));
punt = [FX(:) FY(:)];
clear FX FY;
hparts = 20;

for k=1:parts
    for i=1:length(roinames)
        for j=1:length(vx{i})
            origin=ROI{i}(j,:);
            do=punt(1:ceil(end/size(srcImage,1)/hparts),:);
            ismember=do*0;
            do(:,1)=do(:,1) - origin(1); % Delta origin
            do(:,2)=do(:,2) - origin(2);
            ismember = ismember + left{i}*sign((do)*[vx{i}(j) vy{i}(j)]');
        end
    end
end


figure; imagesc(reshape(do,size(srcImage,1),[]))


%% Create Datastore for big data:
