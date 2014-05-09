% feature_method can be 'PHOW', 'SIFT', 'DSIFT'.

function [read_images, descriptors, total_descriptors] = get_descriptors(root_path, file_names, feature_method, max_descriptors_per_image, plot_descriptors)
    % Variable to improve code legibility.
    num_file_names = length(file_names);

    % Create a counter and a cell array to store the descriptor matrices.
    total_descriptors = 0;
    descriptors = cell(1, num_file_names);

    % Keep track of images not read.
    not_read = [];

    % Iterate through all the folder images.
    for i = 1:num_file_names,
        % Build image path.
        image_path = sprintf('%s/%s', root_path, file_names{i});

        % Load the image.
        try
            image = imread(image_path);
        catch err
            % Error reading the image.
            not_read(end+1) = i;
            continue;
        end

        % Split the image so that only the required part is used.
        [h w d] = size(image); % Height, width and depth.
        half_w = floor(w/2);

        % Convert the image to SINGLE (feature extractors requirements).
        image = im2single(image);

        % Get frames and descriptors using a feature extraction method.
        %  - The size of matrix D will be [128 num_descriptors].
        
        switch upper(feature_method)
        case 'PHOW'
            [F D] = vl_phow(image);
        case 'SIFT'
            if ndims(image) == 3
                image = rgb2gray(image); % Gray scale. SIFT requirement.
            end
            if max_descriptors_per_image == 1
                [F,D] = vl_sift(image, 'frames', [size(image)/2, 14, 0]', ...
                                       'orientations');
            else
                [F D] = vl_sift(image);
            end
        case 'DSIFT'
            if ndims(image) == 3
                image = rgb2gray(image); % Gray scale. DSIFT requirement.
            end
            if max_descriptors_per_image == 1
                [F D] = vl_dsift(image, 'Size', size(image,1)/4, ...
                                        'Step', size(image,1),   ...
                                        'Geometry', [4 4 8]);
            else
                [F D] = vl_dsift(image);
            end
        end 

        % Sample the output descriptors if required.
        num_descriptors = size(D,2);
        if num_descriptors > max_descriptors_per_image && ...
            max_descriptors_per_image ~= 0
            indices = randperm(num_descriptors, max_descriptors_per_image);
            D = D(:, indices);
            F = F(:, indices);
        end

        % Plot the image with the descriptors if required.
        if plot_descriptors
            fig = figure('Name', 'SIFT Descriptors', ...
                         'Position', [200 200 600 600]);
            imshow(image, 'Border', 'tight', 'InitialMagnification', 300);
            hold on;
            h1 = vl_plotframe(F);
            h2 = vl_plotframe(F);
            set(h1, 'color', 'k', 'linewidth', 3);
            set(h2, 'color', 'y', 'linewidth', 2);
            pause;
            h3 = vl_plotsiftdescriptor(D, F);
            set(h3, 'color', 'g');
            pause;
            close(fig);
        end

        % Update the counter and store the descriptors.
        total_descriptors = total_descriptors + size(D,2);
        descriptors{i} = D;
    end

    % Return a cell array with the names of the read images.
    read_images = file_names;
    read_images(not_read) = [];

    % Detele the cells of the images not read.
    descriptors(not_read) = [];
end