% Code by Tejas Krishna Reddy 
% Feb - 2018
% This is the main file, run this while you have the other two programs saved in the same directory.


%imageset will read every file in the folder, 'recursive' would enable it to read every single subfolder containing inside 'EMODATB'
imgSet = imageSet('EMODATB', 'recursive');

% bag of features uses "SURF" feature extraction technique and extracts upto 250 point features from each image. 
bag = bagOfFeatures(imgSet, 'VocabularySize', 250, 'PointSelection', 'Detector');

% The extracted features are converted into computable format
features = encode(bag, imgSet);

% These features are now put in a tabular column for further easier computation and analysis.
getfeatures_tab = array2table(features);
getfeatures_tab.emotionsType = getImageLabels(imgSet);%collects repeated features

%% matlab learner - this opens an app - What to do in this app, and how to improve the accuracy is described in detail in description.
classificationLearner

%% Now test the results from the model obtained from the app.  
findEmotion(trainedClassifier, bag);

