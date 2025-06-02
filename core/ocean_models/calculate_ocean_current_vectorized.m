% calculate_ocean_current_vectorized.m
function [U_total, V_total] = calculate_ocean_current_vectorized(positions, time, current_params)
% CALCULATE_OCEAN_CURRENT_VECTORIZED Computes ocean current for multiple positions.
%
% Inputs:
%   positions      - 2xN matrix of [x; y] coordinates.
%   time           - Current simulation time.
%   current_params - Structure with vortex definitions, type.
%
% Outputs:
%   U_total        - 1xN vector of u components of velocity.
%   V_total        - 1xN vector of v components of velocity.

    N_points = size(positions, 2);
    U_total = zeros(1, N_points);
    V_total = zeros(1, N_points);
    pos_x = positions(1, :); % Row vector of x coordinates
    pos_y = positions(2, :); % Row vector of y coordinates

    for i = 1:length(current_params.vortices)
        vortex = current_params.vortices(i);
        x0 = vortex.center(1);
        y0 = vortex.center(2);
        Gamma = vortex.strength;
        R_vortex = vortex.core_radius; % Renamed to avoid conflict with distance R

        % Adjust vortex parameters for time variance
        if strcmp(current_params.type, 'time_varying')
             x0 = x0 + 5 * sin(0.1 * time);
             y0 = y0 + 3 * cos(0.07 * time);
             Gamma = Gamma * (1 + 0.1*cos(0.05*time));
        end

        % Calculate distances for all points relative to this vortex
        dx = pos_x - x0; % 1xN
        dy = pos_y - y0; % 1xN
        r_sq = dx.^2 + dy.^2; % 1xN

        % Calculate tangential velocity (Lamb-Oseen profile) - Handle singularity
        v_theta = zeros(1, N_points);
        valid_idx = r_sq > eps; % Indices where r_sq is not effectively zero
        r_sq_valid = r_sq(valid_idx);
        v_theta(valid_idx) = (Gamma ./ (2 * pi * sqrt(r_sq_valid))) .* (1 - exp(-r_sq_valid / R_vortex^2));

        % Convert tangential velocity to Cartesian components (vectorized)
        % u = -v_theta * sin(theta) = -v_theta * (dy / r)
        % v =  v_theta * cos(theta) =  v_theta * (dx / r)
        u_vortex = zeros(1, N_points);
        v_vortex = zeros(1, N_points);
        r_valid = sqrt(r_sq_valid);
        u_vortex(valid_idx) = -v_theta(valid_idx) .* dy(valid_idx) ./ r_valid;
        v_vortex(valid_idx) =  v_theta(valid_idx) .* dx(valid_idx) ./ r_valid;

        % Accumulate contributions from this vortex
        U_total = U_total + u_vortex;
        V_total = V_total + v_vortex;
    end
end 