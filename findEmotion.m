function findEmotion(trainedClassifier, bag)
%FINDEMOTION Summary of this function goes here - 
[fig, ax1, ax2] = figureSetup(trainedClassifier);

# to activate inbuilt webcam in your laptop - go to GetHardWare Packages in your matlab toolbar and then install webcams().
# Else it would show an error at this step
# If you are using external webcams like logitech - u gotto install an hardware package that supports that particular webcam. 
# Not all webcams are compatible with MATLAB

# Laptop inbuilt webcam is activated. Read above comments
wcam = webcam();
# or use while(1) - contineous images makes a video, so your webcam is now taking contineous shots and displaying that as a video.
while ishandle(fig)  # Just running forever loop.
    img = snapshot(wcam); # Take an image and then process it
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

# Just creating a right user interface to look at the video produced and the results - For a cleaner representation
# Not compulsory for the project.
function [fig, ax1, ax2] = figureSetup(trainedClassifier)
set(0, 'defaultfigurewindowstyle', 'docked')
fig = figure('Name', 'Eotion Recognition', 'NumberTitle', 'off');
ax1 = subplot(2, 1, 1);
ax2 = subplot(2, 1, 2);
# what ever model u choose in the app, should be seen here, else an error might be caused. In this case i have chosen K- nearest neighbours algorithm.
bar(ax2, zeros(1, numel(trainedClassifier.ClassificationKNN.ClassNames)), 'FaceColor', [0.2 0.6 0.8]); 
set(ax2, 'XTickLabel', cellstr(trainedClassifier.ClassificationKNN.ClassNames));
set(0, 'defaultfigurewindowstyle', 'normal')
        

end


