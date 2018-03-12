function findEmotion(trainedClassifier, bag)
%FINDEMOTION Summary of this function goes here
[fig, ax1, ax2] = figureSetup(trainedClassifier);

wcam = webcam();
while ishandle(fig)
    img = snapshot(wcam);
    graying = rgb2gray(img);
    
    imagefeatures = double(encode(bag, graying));
    
    [imagepred, probabilities] = predict(trainedClassifier.ClassificationKNN, imagefeatures);
    
    try
        imshow(insertText(img, [640, 1], upper(cellstr(imagepred)), 'AnchorPoint', 'RightTop', 'FontSize', 50, 'BoxColor', 'Green', 'BoxOpacity', 0.4), 'Parent', ax1);
        ax2.Children.YData = probabilities;
        ax2.YLine = [0, 1];
    catch err
    end
    drawnow
end
end

function [fig, ax1, ax2] = figureSetup(trainedClassifier)
set(0, 'defaultfigurewindowstyle', 'docked')
fig = figure('Name', 'Eotion Recognition', 'NumberTitle', 'off');
ax1 = subplot(2, 1, 1);
ax2 = subplot(2, 1, 2);
bar(ax2, zeros(1, numel(trainedClassifier.ClassificationKNN.ClassNames)), 'FaceColor', [0.2 0.6 0.8]);
set(ax2, 'XTickLabel', cellstr(trainedClassifier.ClassificationKNN.ClassNames));
set(0, 'defaultfigurewindowstyle', 'normal')
        

end


