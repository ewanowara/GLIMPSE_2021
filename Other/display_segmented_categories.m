load('semantic_cagtegories.mat')

detected_categories = [];
for c = 1:length(color_categories)
% check if cth color category was detected in the image
    redValue = color_categories(c,1); 
    greenValue = color_categories(c,2); 
    blueValue = color_categories(c,3); 

    % Extract the individual red, green, and blue color channels.
    redChannel = seg_img(:, :, 1);
    greenChannel = seg_img(:, :, 2);
    blueChannel = seg_img(:, :, 3);
    mask = redChannel == redValue & ...
        greenChannel == greenValue & ...
        blueChannel == blueValue;

    % if category is present
    if sum(mask(:)) > 0
%         figure, imshow(mask);
        % Get (row, column) list
        [rows, columns] = find(mask);

%     figure, imshow(uint8(mask.*double(seg_img)));
    figure, imshow(uint8(mask.*double(RGB_img)));
    title(word_categories{c}, 'FontSize',16)
    saveas(gcf, [num2str(c) '.jpg'], 'jpg')
    close all
%     pause()
    
    
    detected_categories = [detected_categories; c];
    end
end



