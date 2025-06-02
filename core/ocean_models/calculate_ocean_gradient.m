% calculate_ocean_gradient.m
function grad_U = calculate_ocean_gradient(position, time, current_params, epsilon)
% CALCULATE_OCEAN_GRADIENT Computes the spatial gradient (Jacobian) of the
% ocean current velocity field using central differences.
%
% grad_U = [du/dx, du/dy; dv/dx, dv/dy]
%
% Inputs:
%   position       - [x; y] coordinates.
%   time           - Current simulation time.
%   current_params - Structure with vortex definitions, type.
%   epsilon        - (Optional) Step size for finite difference. Defaults to 1e-4.
%
% Outputs:
%   grad_U         - 2x2 matrix representing the gradient.

    if nargin < 4
        epsilon = 1e-4; % Small step for numerical differentiation
    end
    if epsilon <= 0
        error('Epsilon must be positive for finite differences.');
    end

    x = position(1);
    y = position(2);

    % Create matrix of the 4 points needed for central differences
    points_to_sample = [
        x + epsilon, x - epsilon, x          , x;
        y          , y          , y + epsilon, y - epsilon
    ]; % 2x4 matrix

    % Call the vectorized function ONCE for these 4 points
    [U_sampled, V_sampled] = calculate_ocean_current_vectorized(points_to_sample, time, current_params);
    % U_sampled = [u_xp, u_xm, u_yp, u_ym]
    % V_sampled = [v_xp, v_xm, v_yp, v_ym]

    % Compute partial derivatives
    du_dx = (U_sampled(1) - U_sampled(2)) / (2 * epsilon); % (u_xp - u_xm) / (2*eps)
    du_dy = (U_sampled(3) - U_sampled(4)) / (2 * epsilon); % (u_yp - u_ym) / (2*eps)
    dv_dx = (V_sampled(1) - V_sampled(2)) / (2 * epsilon); % (v_xp - v_xm) / (2*eps)
    dv_dy = (V_sampled(3) - V_sampled(4)) / (2 * epsilon); % (v_yp - v_ym) / (2*eps)

    % Assemble the gradient matrix (Jacobian)
    grad_U = [du_dx, du_dy;
              dv_dx, dv_dy];

end 