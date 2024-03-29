function [trainedClassifier, validationAccuracy] = trainSVM(trainingData)
% [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)
% returns a trained classifier and its accuracy. This code recreates the
% classification model trained in Classification Learner app. Use the
% generated code to automate training the same model with new data, or to
% learn how to programmatically train models.
%
%  Input:
%      trainingData: a table containing the same predictor and response
%       columns as imported into the app.
%
%  Output:
%      trainedClassifier: a struct containing the trained classifier. The
%       struct contains various fields with information about the trained
%       classifier.
%
%      trainedClassifier.predictFcn: a function to make predictions on new
%       data.
%
%      validationAccuracy: a double containing the accuracy in percent. In
%       the app, the History list displays this overall accuracy score for
%       each model.
%
% Use the code to train the model with new data. To retrain your
% classifier, call the function from the command line with your original
% data or new data as the input argument trainingData.
%
% For example, to retrain a classifier trained with the original data set
% T, enter:
%   [trainedClassifier, validationAccuracy] = trainClassifier(T)
%
% To make predictions with the returned 'trainedClassifier' on new data T2,
% use
%   yfit = trainedClassifier.predictFcn(T2)
%
% T2 must be a table containing at least the same predictor columns as used
% during training. For details, enter:
%   trainedClassifier.HowToPredict

% Auto-generated by MATLAB on 12-Jun-2018 17:17:24


% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictorNames = {'sHops', 'TotPkts', 'TotBytes', 'TotAppByte', 'Dur', 'sTtl', 'TcpRtt', 'SynAck', 'SrcPkts', 'DstPkts', 'TotAppByte_1', 'Rate', 'SrcRate', 'DstRate'};
predictors = inputTable(:, predictorNames);
response = inputTable.Label;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationSVM = fitcsvm(...
    predictors, ...
    response, ...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', categorical({'Normal'; 'Botnet'}, {'flow=Background' 'flow=Background-Attempt-cmpgw-CVUT' 'flow=Background-Established-cmpgw-CVUT' 'flow=Background-TCP-Attempt' 'flow=Background-TCP-Established' 'flow=Background-UDP-Attempt' 'flow=Background-UDP-Established' 'flow=Background-UDP-NTP-Established-1' 'flow=Background-ajax.google' 'flow=Background-google-analytics1' 'flow=Background-google-analytics10' 'flow=Background-google-analytics11' 'flow=Background-google-analytics12' 'flow=Background-google-analytics13' 'flow=Background-google-analytics14' 'flow=Background-google-analytics15' 'flow=Background-google-analytics16' 'flow=Background-google-analytics2' 'flow=Background-google-analytics3' 'flow=Background-google-analytics4' 'flow=Background-google-analytics5' 'flow=Background-google-analytics6' 'flow=Background-google-analytics7' 'flow=Background-google-analytics8' 'flow=Background-google-analytics9' 'flow=Background-google-pop' 'flow=Background-google-webmail' 'flow=Background-www.fel.cvut.cz' 'flow=From-Background-CVUT-Proxy' 'flow=From-Botnet-V52-1-ICMP' 'flow=From-Botnet-V52-1-TCP-Established' 'flow=From-Botnet-V52-1-TCP-HTTP-Google-Net-Established-6' 'flow=From-Botnet-V52-1-UDP-Attempt' 'flow=From-Botnet-V52-1-UDP-DNS' 'flow=From-Botnet-V52-2-ICMP' 'flow=From-Botnet-V52-2-TCP-CC106-IRC-Not-Encrypted' 'flow=From-Botnet-V52-2-TCP-HTTP-Google-Net-Established-6' 'flow=From-Botnet-V52-2-UDP-Attempt' 'flow=From-Botnet-V52-2-UDP-DNS' 'flow=From-Botnet-V52-3-TCP-CC106-IRC-Not-Encrypted' 'flow=From-Botnet-V52-3-TCP-HTTP-Google-Net-Established-6' 'flow=From-Botnet-V52-3-UDP-Attempt' 'flow=From-Botnet-V52-3-UDP-DNS' 'flow=From-Normal-V52-CVUT-WebServer' 'flow=From-Normal-V52-Grill' 'flow=From-Normal-V52-Jist' 'flow=From-Normal-V52-MatLab-Server' 'flow=From-Normal-V52-Stribrek' 'flow=From-Normal-V52-UDP-CVUT-DNS-Server' 'flow=Normal-V52-HTTP-windowsupdate' 'flow=To-Background-CVUT-Proxy' 'flow=To-Background-CVUT-WebServer' 'flow=To-Background-Grill' 'flow=To-Background-Jist' 'flow=To-Background-MatLab-Server' 'flow=To-Background-Stribrek' 'flow=To-Background-UDP-CVUT-DNS-Server' 'flow=To-Normal-V52-UDP-NTP-server' 'Normal' 'Background' 'Botnet'}));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'sHops', 'TotPkts', 'TotBytes', 'TotAppByte', 'Dur', 'sTtl', 'TcpRtt', 'SynAck', 'SrcPkts', 'DstPkts', 'TotAppByte_1', 'Rate', 'SrcRate', 'DstRate'};
trainedClassifier.ClassificationSVM = classificationSVM;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2017a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictorNames = {'sHops', 'TotPkts', 'TotBytes', 'TotAppByte', 'Dur', 'sTtl', 'TcpRtt', 'SynAck', 'SrcPkts', 'DstPkts', 'TotAppByte_1', 'Rate', 'SrcRate', 'DstRate'};
predictors = inputTable(:, predictorNames);
response = inputTable.Label;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 5);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
