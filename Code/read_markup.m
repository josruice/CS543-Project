% TODO: Write proper documentation.

function [cell_properties] = read_markup(file_path, file_names)
    % Function constants.
    property_prefix = 'Class';
    property_delim = '.';

    % Open the file with reading permissions and create a file descriptor.
    file = fopen(file_path);
    num_file_names = length(file_names);
    
    % Read the headers of the file.
    single_cell_headers = textscan(file, '%s\n', 1);
    cell_headers = strsplit(single_cell_headers{1}{1}, ',');
    cell_headers = cell_headers(2:end); % Remove the id column.

    % Close the file.
    fclose(file);

    % Read the file ids and feature vectors.
    ids_and_labels = csvread(file_path, 1, 0); % Avoid headers.
    ids = ids_and_labels(:,1);
    labels = ids_and_labels(:,2:end);
    
    file_names_ids = cellfun(@(x) str2double(x(1:find(x=='.',1)-1)), file_names);
    labels = labels(ismember(ids, file_names_ids), :);
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
        cell_properties{i} = {question, answer, labels(:,i)};
    end
end