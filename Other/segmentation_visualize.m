composed_img = imread('/Users/ewanowara/Desktop/my_img.png');

RGB_img = composed_img(:, 1:size(composed_img,2)/2, :);

seg_img = composed_img(:, size(composed_img,2)/2+1:size(composed_img,2), :);

figure, imshow(RGB_img)

figure, imshow(seg_img)





 
