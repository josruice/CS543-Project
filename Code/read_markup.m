% TODO: Write proper documentation.

function [cell_properties] = read_markup(file_path, num_file_names)
    % Function constants.
    property_prefix = 'Class';
    property_delim = '.';

    % Open the file with reading permissions and create a file descriptor.
    file = fopen(file_path);

    % Read the headers of the file.
    single_cell_headers = textscan(file, '%s\n', 1);
    cell_headers = strsplit(single_cell_headers{1}{1}, ',');
    cell_headers = cell_headers(2:end); % Remove the id column.

    % Close the file.
    fclose(file);

    % Read the feature vectors.
    labels = csvread(file_path, 1, 1); % Avoid headers row and ids column.
    labels(find(labels == 0)) = -1;    % Substitute zeros by minus ones.

    % All the data will be stored in a cell column vector where each slot will 
    % contain the question number, the answer number and the labels vector.
    num_properties = length(cell_headers);
    cell_properties = cell(num_properties, 1);

    for i = 1:num_properties,
        % Parse the question and answer of the property.
        prop_no_delim = strtok(cell_headers{i}, property_prefix);
        question_and_answer = strsplit(prop_no_delim, property_delim);
        question = question_and_answer(1);
        answer = question_and_answer(2);
        
        % Store everything in the cell array.
        cell_properties{i} = {question, answer, labels(1:num_file_names,i)};
    end
end