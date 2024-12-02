% 假设您已经获得了相机参数 cameraParams
% 选择待矫正的图像
imageToCorrect = 'picture/01.jpg';  % 替换为您的图像文件名
originalImage = imread(imageToCorrect);  % 读取待矫正的图像

% 使用内参矫正图像
undistortedImage = undistortImage(originalImage, cameraParams);  % 校正图像

% 显示原图和校正后的图像
figure;
subplot(1, 2, 1);
imshow(originalImage);
title('原图');

subplot(1, 2, 2);
imshow(undistortedImage);
title('校正后的图像');






