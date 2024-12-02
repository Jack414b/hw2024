inputImage = imread('cream.jpg');% 读取拍摄的照片

figure;
imshow(inputImage);
title('click on the four corners of the desk');
disp('click on the four corners of the desk in clockwise order');
[x,y]=ginput(4);
disp([x,y])

outputcorners =[0,0;157,0;157,184;0,184];

tform =fitgeotrans([x,y],outputcorners,'projective');

outputsize=[184,157];
warpedImage=imwarp(inputImage, tform, 'outputView',imref2d(outputsize));

figure;
imshow(warpedImage);
title('Projected Image');

imwrite(warpedImage, 'reprojected_image1.jpg');
