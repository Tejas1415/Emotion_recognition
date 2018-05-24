function imageType = getImageLabels(imset)
% Just labelling the images and the features captured and encoded.
% categorial features are non numerical data features which cannot be understood by most machine learning algorithms.
    imageType = categorical(repelem({imset.Description}', ...
        [imset.Count], 1));
end
