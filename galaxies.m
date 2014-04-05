%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  CS 543 - Final project (Spring 2014)  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%% Galaxy shape classifier.
%
%%% Andres Guzman-Ballen
%%% Ettienne Montagner
%%% Jose Vicente Ruiz Cepeda (ruizcep2 -at- illinois.edu)
%

%%%%%%%%%%%%%%%%%%%%%
%%%   Constants   %%%
%%%%%%%%%%%%%%%%%%%%%

% Data.
root_path = 'Dataset'; % Without last slash.

names_file_path = 'file_names.txt';

markup_file = '';

% Classifiers.
feature_method = 'SIFT'; % PHOW, SIFT or DSIFT.

% Number of clusters used in the K-means.
num_clusters = 300; 

% Support Vector Machine (SVM) solver.
solver = 'SDCA'; % SGD or SDCA.

% Lambda value of the SVM.
lambda = 0.05;

%%%%%%%%%%%%%%%%%%%%%
%%%   Libraries   %%%
%%%%%%%%%%%%%%%%%%%%%

% Load VLFeat library.
run('/Users/Josevi/Libraries/vlfeat-0.9.18/toolbox/vl_setup');


%%%%%%%%%%%%%%%%%%%%
%%%    Script    %%%
%%%%%%%%%%%%%%%%%%%%

% Read the images names from file.
file = fopen('file_names.txt');
cell_names = textscan(file,'%s\n');
file_names = cell(cell_names{1});

% Variable to improve code legibility.
num_file_names = length(file_names);

%%% TRAINING %%%

% Get the descriptors of each image of the dataset.
[descriptors, total_descriptors] = get_descriptors(root_path, file_names, feature_method);

% Quantize the feature descriptors using k-means.
[features] = quantize_feature_vectors (descriptors, total_descriptors, num_clusters);

keyboard;