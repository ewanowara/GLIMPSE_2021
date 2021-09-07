% num_imgs = 237; % im2gps
num_imgs = 3000; % im2gps3k

clims = [0, round(num_imgs*0.1)]; 

% clims = [0, 150]; 

% figure, imagesc(co_occurence_mat, clims), title('All')

figure, imagesc(co_occurence_mat(1:50, 1:50), clims), set(gca,'FontSize',16) %title('1:50')

% figure, imagesc(co_occurence_mat(51:100, 51:100), clims), title('51:100')
% 
% figure, imagesc(co_occurence_mat(101:150, 101:150), clims), title('101:150')
