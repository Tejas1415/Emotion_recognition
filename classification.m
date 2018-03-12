imgSet = imageSet('EMODATB', 'recursive');
%recur meaning
bag = bagOfFeatures(imgSet, 'VocabularySize', 250, 'PointSelection', 'Detector');
%SURF extraction using predefined detectors
features = encode(bag, imgSet);
getfeatures_tab = array2table(features);
getfeatures_tab.emotionsType = getImageLabels(imgSet);%collects repeated features

%% matlab learner default
classificationLearner

%%  
findEmotion(trainedClassifier, bag);

