function [x_vertices, y_vertices] = create_circle_vertices(center, radius, num_points)
% CREATE_CIRCLE_VERTICES Generates vertices for plotting a filled circle.
%
% Inputs:
%   center     - [x; y] coordinates of the circle center.
%   radius     - Radius of the circle.
%   num_points - (Optional) Number of points to define the circle boundary.
%                Defaults to 30.
%
% Outputs:
%   x_vertices - X-coordinates of the vertices.
%   y_vertices - Y-coordinates of the vertices.

    if nargin < 3
        num_points = 1000; % Default number of points for smoothness
    end 
    
    % Ensure center is a column vector
    if size(center, 1) == 1
        center = center';
    end
    
    angles = linspace(0, 2*pi, num_points+1); % Generate angles
    x_vertices = center(1) + radius * cos(angles); % Calculate x coordinates
    y_vertices = center(2) + radius * sin(angles); % Calculate y coordinates
end 