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
max_training_samples = 500;

test_names_file_path = '../Markup/test_file_names.txt';
max_test_samples = 200;

markup_file = '../Markup/binary_training_solutions.csv';

transitions_file = '../Markup/question_transitions.txt';

first_question = '1';

num_questions = 11;

plot_descriptors = true;

% Determines verbosity of the output.
verbose = 0;

% Classifiers.
feature_method = 'SIFT';          % HOG, PHOW, SIFT or DSIFT.
max_descriptors_per_image = 1;    % 0 means infinite. 1 only with SIFT or DSIFT.
scale = 10;                       % Scale (only with 1 descriptor and SIFT).
cell_size = 8;                    % Only with HOG.

% Number of clusters used in the K-means.
num_clusters = 300; 

% Use binary histograms.
binary_histograms = 0;   % 0 or 1.

% Support Vector Machine (SVM) solver.
solver = 'SDCA'; % SGD or SDCA.

% Lambda value of the SVM.
lambda = 1e-5;

% Loss parameter of SVM.
loss = 'Logistic';


%%%%%%%%%%%%%%%%%%%%%
%%%   Libraries   %%%
%%%%%%%%%%%%%%%%%%%%%

% Load VLFeat library.
run('/Users/Josevi/Libraries/vlfeat-0.9.18/toolbox/vl_setup');


%%%%%%%%%%%%%%%%%%%%
%%%    Script    %%%
%%%%%%%%%%%%%%%%%%%%

%%% TRAINING %%%
time_start_global = tic;

% Read the images names from file.
file = fopen(training_names_file_path);
cell_names = textscan(file,'%s\n');
file_names = cell(cell_names{1});
file_names = file_names(1:min(length(file_names), max_training_samples), :);

% Get the descriptors of each image of the training dataset.
time_start_descriptors = tic;
[file_names, descriptors, total_descriptors] = get_descriptors(root_path,    ...
    file_names, feature_method, max_descriptors_per_image, plot_descriptors, ...
    scale, cell_size);
time_descriptors = toc(time_start_descriptors);

% Update number of files since some might be missing.
num_file_names = length(file_names);

time_start_quantization = tic;
if max_descriptors_per_image == 1 || strcmpi(feature_method, 'HOG')
    features_3d = single(cell2mat(cellfun(@(x) reshape(x, 1, 1, []), ...
                         descriptors, 'UniformOutput', false)));
    num_clusters = size(features_3d, 3);
else
    % Quantize the feature descriptors using k-means.
    [features_3d, clusters] = quantize_feature_vectors (descriptors, ...
                                        total_descriptors, [], num_clusters);
end
time_quantization = toc(time_start_quantization);

% Read markup data. The structure of the output cell array will be a column
% vector of cell arrays with one row per different property, where each of the
% elements store the question number, the answer number and a  
% vector with the images that have the property. 
[cell_real_properties] = read_markup(markup_file, file_names);
num_properties = length(cell_real_properties);

% Permute and reshape the features to fit the Support Vector Machine (SVM)
% requirements: one column per example.
features_2d = permute(features_3d, [3 2 1]);
features_2d = reshape(features_2d, [num_clusters, num_file_names]);
if binary_histograms
    features_2d( find(features_2d) ) = 1;
end

% Cell array formed by cell arrays each with 4 elements: 
% Question number, answer number, SVM weight vector, SVM bias.
svms = cell(num_properties, 1);

% 2D matrix used to store the real labels of the data with one sample per 
% column.
real_labels = zeros(num_properties, num_file_names);

time_start_svm = tic;
% Use SVM to create the linear clasifiers for the properties.
for i = 1:num_properties,
    prop = cell_real_properties{i};
    question = prop{1};
    answer = prop{2};
    labels = prop{3};

    % Get real labels of the images for this property.
    real_labels(i,:) = labels';

    % Build the classifier for this property.
    [W,B,~,scores] = vl_svmtrain(features_2d, labels, lambda, ...
                                 'Solver', solver,            ...     
                                 'Loss', loss);

    % Store everything in the cell data structure.
    svms{i} = {question, answer, W, B};

    % Elements with score 0 (on the line of the linear classifier) are
    % the same as negative (don't have the property).
    estimated_labels = sign(scores); % Returns -1, 0 or 1, depending on sign.
    estimated_labels( find(estimated_labels == 0) ) = -1;
end
time_svm = toc(time_start_svm);


%%% SVMs with One Vs. All on training data %%%

% Read the question-transitions file.
transitions_matrix = dlmread(transitions_file);

% Cell array used to store the indices of the images that correspond to each 
% question.
images_per_question = cell( num_questions+1, 1 ); % First value correspond to zero.

% Put all the images in the first question.
images_per_question{2} = [1:num_file_names]; 

% Classify using the SVMs with a real one vs. all.
time_start_one_vs_all = tic;
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
time_one_vs_all = toc(time_start_one_vs_all);

% Print results.
correctly_classified = find(sum(real_labels == estimated_labels) == ...
                            num_properties);
accuracy = length(correctly_classified)*100/num_file_names;
time_global = toc(time_start_global);
fprintf(1, ['Training, %s, %d files | '     ...
            '%s, %d maxDesPerImg, %d SIFT scale, %d HOG cellsize, %.2f s | ' ...
            '%d clusters, %d binHist, %.2f s | '    ...
            '%s solver, %s loss, %.0e lambda, %.2f s | ' ...
            '%.2f%% accuracy, %.2f s | '...
            '%.2f s\n'], ...
            root_path(find(root_path=='/',1,'last')+1:end), num_file_names, ...
            feature_method, max_descriptors_per_image, scale, cell_size,    ...
                time_descriptors, ...
            num_clusters, binary_histograms, time_quantization, ...
            solver, loss, lambda, time_svm, ...
            accuracy, time_one_vs_all, ...
            time_global);

% ---------------------------------------------------------------------------- %

%%% TESTING %%%
time_start_global = tic;

% Read the images names from file.
file = fopen(test_names_file_path);
cell_names = textscan(file,'%s\n');
file_names = cell(cell_names{1});
file_names = file_names(1:min(length(file_names), max_test_samples), :);

% Get the descriptors of each image of the test dataset.
time_start_descriptors = tic;
[file_names, descriptors, total_descriptors] = get_descriptors(root_path,    ...
    file_names, feature_method, max_descriptors_per_image, plot_descriptors, ...
    scale, cell_size);
time_descriptors = toc(time_start_descriptors);

% Update number of files since some might be missing.
num_file_names = length(file_names);

time_start_quantization = tic;
if max_descriptors_per_image == 1 || strcmpi(feature_method, 'HOG')
    features_3d = single(cell2mat(cellfun(@(x) reshape(x, 1, 1, []), ...
                         descriptors, 'UniformOutput', false)));
    num_clusters = size(features_3d, 3);
else
    % Quantize the feature descriptors using k-means.
    [features_3d, ~] = quantize_feature_vectors (descriptors, ...
                                    total_descriptors, clusters, num_clusters);
end
time_quantization = toc(time_start_quantization);

% Read markup data. The structure of the output cell array will be a column
% vector of cell arrays with one row per different property, where each of the
% elements store the question number, the answer number and a  
% vector with the images that have the property. 
[cell_real_properties] = read_markup(markup_file, file_names);
num_properties = length(cell_real_properties);

% Permute and reshape the features to fit the Support Vector Machine (SVM)
% requirements: one column per example.
features_2d = permute(features_3d, [3 2 1]);
features_2d = reshape(features_2d, [num_clusters, num_file_names]);
if binary_histograms
    features_2d( find(features_2d) ) = 1;
end

% 2D matrix used to store the real labels of the data with one sample per 
% column.
real_labels = zeros(num_properties, num_file_names);
for i = 1:num_properties,
    prop = cell_real_properties{i};
    labels = prop{3};
    real_labels(i,:) = labels';
end

%%% SVMs with One Vs. All on test data %%%

% Read the question-transitions file.
transitions_matrix = dlmread(transitions_file);

% Cell array used to store the indices of the images that correspond to each 
% question.
images_per_question = cell( num_questions+1, 1 ); % First value correspond to zero.

% Put all the images in the first question.
images_per_question{2} = [1:num_file_names]; 

% Classify using the SVMs with a real one vs. all.
time_start_one_vs_all = tic;
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
time_one_vs_all = toc(time_start_one_vs_all);

% Print results.
correctly_classified = find(sum(real_labels == estimated_labels) == ...
                            num_properties);
accuracy = length(correctly_classified)*100/num_file_names;
time_global = toc(time_start_global);
fprintf(1, ['Testing, %s, %d files | '     ...
            '%s, %d maxDesPerImg, %d SIFT scale, %d HOG cellsize, %.2f s | ' ...
            '%d clusters, %d binHist, %.2f s | '    ...
            '%s solver, %s loss, %.0e lambda, %.2f s | ' ...
            '%.2f%% accuracy, %.2f s | '...
            '%.2f s\n'], ...
            root_path(find(root_path=='/',1,'last')+1:end), num_file_names, ...
            feature_method, max_descriptors_per_image, scale, cell_size,    ...
                time_descriptors, ...
            num_clusters, binary_histograms, time_quantization, ...
            solver, loss, lambda, time_svm, ...
            accuracy, time_one_vs_all, ...
            time_global);

%keyboard;