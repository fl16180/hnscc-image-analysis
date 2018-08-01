function resmat = imscore
    img = 'D:\ISB\HNSCC\raw channels\ONCOSEC partII__N4_HP_IM3_5_[36508.7,8557.9]_component_data.tif';

    all_img = filt_stack(img);
    all_img = cat(1,all_img{:});
    all_img = all_img';
    
    combs = nchoosek(1:35,2);
    resmat = zeros(1,length(combs));
    for i = 1:length(combs)
        disp(i);
        tmp = combs(i,:);
        resmat(i) = spearman_score(all_img{tmp(1)},all_img{tmp(2)});
    end
end


function filt_image = gaussfilt(I,sigma)
    imsize = size(I);
    h = fspecial('gaussian',imsize,sigma);
    I = im2double(I);
    filt_image = imfilter(I,h,'conv');
    %figure,imagesc(filt_image),impixelinfo,title('Original Image after Convolving with gaussian'),colormap('gray');
end

function img_array = applyfilt(channel)
    s = [1 5 20 50 100];
    img_array = cell(1,length(s));
    for i=1:length(s)
        img_array{i} = gaussfilt(channel,s(i));
    end
end

function all_img = filt_stack(img)
    imginfo = imfinfo(img);
    a594=imread(img,1);
    coum=imread(img,2);
    cy3=imread(img,3);
    fitc=imread(img,4);
    cy5=imread(img,5);
    dapi=imread(img,6);
    cy55=imread(img,7);
    
    all_img = cell(1,7);
    channels = {a594,coum,cy3,fitc,cy5,dapi,cy55};
    for i=1:7
        all_img{i} = applyfilt(channels{i});
    end
end




function res = spearman_score(A,B)
    res = corr(A(:),B(:),'type','Spearman');
end

