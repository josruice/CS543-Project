%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  CS 543 - Final project (Spring 2014)  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%% Galaxy shape classifier.
%
%%% Andres Guzman-Ballen (aguzman4 -at- illinois.edu)
%%% Ettienne Montagner (montgnr2 -at- illinois.edu)
%%% Jose Vicente Ruiz Cepeda (ruizcep2 -at- illinois.edu)
%

%%%%%%%%%%%%%%%%%%%%%
%%%   Constants   %%%
%%%%%%%%%%%%%%%%%%%%%

% Data.
root_path = '../CleanDataset'; % Without last slash.

training_names_file_path = '../Markup/training_file_names.txt';

test_names_file_path = '../Markup/test_file_names.txt';

markup_file = '../Markup/binary_training_solutions.csv';

transitions_file = '../Markup/question_transitions.txt';

first_question = '1';

num_questions = 11;

% Determines verbosity of the output.
verbose = 0;

% Classifiers.
feature_method = 'SIFT'; % PHOW, SIFT or DSIFT.

% Number of clusters used in the K-means.
num_clusters = 600; 

% Support Vector Machine (SVM) solver.
solver = 'SDCA'; % SGD or SDCA.

% Lambda value of the SVM.
lambda = 0.000001;


%%%%%%%%%%%%%%%%%%%%%
%%%   Libraries   %%%
%%%%%%%%%%%%%%%%%%%%%

% Load VLFeat library.
run('/Users/Josevi/Libraries/vlfeat-0.9.18/toolbox/vl_setup');


%%%%%%%%%%%%%%%%%%%%
%%%    Script    %%%
%%%%%%%%%%%%%%%%%%%%

%%% TRAINING %%%

% Read the images names from file.
file = fopen(training_names_file_path);
cell_names = textscan(file,'%s\n');
file_names = cell(cell_names{1});

% Get the descriptors of each image of the training dataset.
[file_names, descriptors, total_descriptors] = get_descriptors(root_path, file_names, feature_method);

% Variable to improve code legibility.
num_file_names = length(file_names);

% Quantize the feature descriptors using k-means.
[features_3d] = quantize_feature_vectors (descriptors, total_descriptors, num_clusters);

% Read markup data. The structure of the output cell array will be a column
% vector of cell arrays with one row per different property, where each of the
% elements store the question number, the answer number and a  
% vector with the images that have the property. 
[cell_real_properties] = read_markup(markup_file, num_file_names);

% Permute and reshape the features to fit the Support Vector Machine (SVM)
% requirements: one column per example.
features_2d = permute(features_3d, [3 2 1]);
features_2d = reshape(features_2d, [num_clusters, num_file_names]);
features_2d( find(features_2d) ) = 1; % Binary histograms.

% Cell array formed by cell arrays each with 4 elements: 
% Question number, answer number, SVM weight vector, SVM bias.
num_properties = length(cell_real_properties);
svms = cell(num_properties, 1);

% 2D matrix used to store the real labels of the data with one sample per 
% column.
real_labels = zeros(num_properties, num_file_names);

% Variables used to test the accuracy.
min_accuracy = 1;
max_accuracy = 0;
mean_accuracy = 0;

% Use SVM to create the linear clasifiers for the properties.
for i = 1:num_properties,
    prop = cell_real_properties{i};
    question = prop{1};
    answer = prop{2};
    labels = prop{3};

    % Get real labels of the images for this property.
    real_labels(i,:) = labels';

    % Build the classifier for this property.
    [W,B,~,scores] = vl_svmtrain(features_2d, labels, lambda, 'Solver', solver);

    % Store everything in the cell data structure.
    svms{i} = {question, answer, W, B};

    % Elements with score 0 (on the line of the linear classifier) are
    % the same as negative (don't have the property).
    estimated_labels = sign(scores); % Returns -1, 0 or 1, depending on sign.
    estimated_labels( find(estimated_labels == 0) ) = -1;

    % Testing accuracy in the training set.
    accuracy = sum(labels == sign(estimated_labels')) / length(labels);
    mean_accuracy = mean_accuracy + accuracy;
    min_accuracy = min(min_accuracy, accuracy);
    max_accuracy = max(max_accuracy, accuracy);
end

if verbose >= 1
    % Print the resulting training accuracies.
    mean_accuracy = (mean_accuracy / num_properties);
    fprintf(1, 'TRAINING SET:\n');
    fprintf(1, ' - Mean accuracy: %.2f\n', mean_accuracy * 100);
    fprintf(1, ' - Min accuracy: %.2f\n', min_accuracy * 100);
    fprintf(1, ' - Max accuracy: %.2f\n\n', max_accuracy * 100);
end


%%% SVMs with One Vs. All %%%

% Read the question-transitions file.
transitions_matrix = dlmread(transitions_file);

% Cell array used to store the indices of the images that correspond to each 
% question.
images_per_question = cell( num_questions+1, 1 ); % First value correspond to zero.

% Put all the images in the first question.
images_per_question{2} = [1:num_file_names]; 

% Classify using the SVMs with a real one vs. all.
i = 1;
estimated_labels = -ones( num_properties, num_file_names );
for i = 1:2, % Two iterations are enough to answer all the questions.
    pos = 1;
    while pos <= num_properties,
        question_char = svms{pos}{1}{1};
        question_index = str2num(question_char) + 1;
        question_features_indices = images_per_question{question_index};
        question_features = features_2d(:, question_features_indices);
        num_question_features = length(question_features_indices);
        num_answers = 0;
        initial_pos = pos;

        % Continue if no features in this question.
        if num_question_features == 0
            pos = pos + 1;
            continue;
        end

        % Get the weight vectors and bias of the SVMs of the answers to this 
        % question.
        weights = [];
        bias = [];
        while pos <= num_properties && strcmp(svms{pos}{1}{1}, question_char),
            weights = [weights, svms{pos}{3}];
            bias = [bias; svms{pos}{4}];
            num_answers = num_answers + 1;
            pos = pos + 1;
        end
        
        % Classify the data with the stored SVMs.
        scores = (weights' * question_features) + repmat(bias, [1, num_question_features]);
        [~, I] = max(scores);

        % Update the estimated labels with the results obtained for this question.
        new_labels = -ones( num_answers, num_question_features);
        new_labels( sub2ind(size(new_labels), I, 1:num_question_features) ) = 1;
        estimated_labels( initial_pos:pos-1, question_features_indices ) = new_labels;

        % Move the images to the corresponding next question.
        for j = 1:num_answers,
            next_question_index = transitions_matrix(initial_pos + j - 1, 3) + 1;
            next_question_current_indices = images_per_question{next_question_index};
            images_per_question{next_question_index} = [next_question_current_indices, question_features_indices( find(I == j) )];
        end

        % Remove all the images of this question.
        images_per_question{question_index} = [];
    end
end

% Print the number of correctly classified images.
correctly_classified = find(sum(real_labels == estimated_labels) == num_properties);
fprintf(1, 'Correctly classified images: %d out of %d (%.2f%%)\n', length(correctly_classified), num_file_names, length(correctly_classified)*100/num_file_names);

%keyboard;