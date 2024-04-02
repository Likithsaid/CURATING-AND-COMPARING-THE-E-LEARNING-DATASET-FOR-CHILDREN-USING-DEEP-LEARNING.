
% cam=webcam;
clear all;
clc;
close all;

cam = webcam;
cc=1;
load('eng1.mat');
while cc==1
    im=cam.snapshot();
    im=imresize(im,[100 100]);
    image(im);
    im=rgb2gray(im);
    label=classify(net,im);
    title(['The recognized emotion is : ',label]);
    drawnow;
   
    kkey = get(gcf,'CurrentCharacter');
     disp(kkey)
        if strcmp(kkey ,'a')==1
            cc=0;
            clear cam;            
        end
end
    