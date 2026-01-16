% create_symbolic_ocean_func.m
function [ocean_func, ocean_gradient_func] = create_symbolic_ocean_func(current_params, use_mx)
% function [ocean_func] = create_symbolic_ocean_func(current_params, use_mx)
% CREATE_SYMBOLIC_OCEAN_FUNC Creates symbolic CasADi functions for ocean current calculation.
%
% Inputs:
%   current_params - Structure containing current parameters
%   use_mx        - Boolean flag (true for MX symbols, false for SX)
%
% Outputs:
%   ocean_func              - CasADi function that calculates current at given positions
%   ocean_linear_approx_func - CasADi function that calculates linear approximation of current

    import casadi.* % Import CasADi package

    % --- Create Symbolic Variables ---
    if use_mx
        V = MX; % Use MX directly after import
    else
        V = SX; % Use SX directly after import
    end

    % Symbolic variables
    P = V.sym('P', 2, 1);   % Current position (2x1)
    % P0 = V.sym('P0', 2, 1); % Reference position for linearization (2x1)
    t = V.sym('t', 1, 1);   % Time

    current_vec = cell(1, current_params.num_ensemble_members+1+current_params.num_ensemble_members_test);
    J = cell(1, current_params.num_ensemble_members+1+current_params.num_ensemble_members_test);
    outputCurrentNames = cell(1, current_params.num_ensemble_members+1+current_params.num_ensemble_members_test);
    outputGradientNames = cell(1, current_params.num_ensemble_members+1+current_params.num_ensemble_members_test);

    current_vec{current_params.num_ensemble_members+1} = V.zeros(2, 1);
    J{current_params.num_ensemble_members+1} = V.zeros(2, 2);

    for j = 1:current_params.num_ensemble_members+1+current_params.num_ensemble_members_test
        % Initialize total velocity
        u_total = 0;
        v_total = 0;

        % --- Calculate Current for Each Vortex ---
        for i = 1:length(current_params.vortices)
            vortex = current_params.vortices(i);
            
            % Get base vortex parameters
            x0_base = vortex.center(1);
            y0_base = vortex.center(2);
            Gamma_base = vortex.strength;
            R_vortex = vortex.core_radius;

            % Handle time-varying behavior if needed
            if strcmp(current_params.type, 'time_varying')
                % Example: Oscillating vortex center and strength
                vortex_end = current_params.vortices_end(i);   
                
                x0_end = vortex_end.center(1);
                y0_end = vortex_end.center(2);
                Gamma_end = vortex_end.strength;
                R_vortex_end = vortex_end.core_radius;

                
                x0 = x0_base + (x0_end - x0_base) * t/current_params.T_final;
                y0 = y0_base + (y0_end - y0_base) * t/current_params.T_final;
                Gamma = Gamma_base + (Gamma_end - Gamma_base) * t/current_params.T_final;
            else
                x0 = x0_base;
                y0 = y0_base;
                Gamma = Gamma_base;
            end

            % Calculate position relative to vortex center
            dx = P(1) - x0;
            dy = P(2) - y0;
            r_sq = dx^2 + dy^2;

            % Calculate vortex-induced velocity (Lamb-Oseen profile)
            % Use if_else for symbolic conditional
            r = sqrt(r_sq + eps); % Add small constant to avoid division by zero
            v_theta = (Gamma / (2 * pi * r)) * (1 - exp(-r_sq / R_vortex^2));
            
            % Convert to Cartesian components
            u_vortex = -v_theta * dy / r;
            v_vortex = v_theta * dx / r;
            
            % Add to total velocity
            u_total = u_total + u_vortex;
            v_total = v_total + v_vortex;
        end

        % --- Create Original Function ---
        if j <= current_params.num_ensemble_members
            current_vec{j} = (eye(2) + diag(randn(2, 1) * current_params.noise_level)) *[u_total; v_total] ;
            % --- Compute Jacobian with respect to position P ---
            J{j} = jacobian(current_vec{j}, P);  % 2x2 matrix
            outputCurrentNames{j} = ['current_out_', num2str(j)];
            outputGradientNames{j} = ['gradient_out_', num2str(j)];

            current_vec{current_params.num_ensemble_members+1} = current_vec{current_params.num_ensemble_members+1} + current_vec{j}/current_params.num_ensemble_members;
            J{current_params.num_ensemble_members+1} = J{current_params.num_ensemble_members+1} + J{j}/current_params.num_ensemble_members;
        else
            current_vec{j+1} = (eye(2) + diag(randn(2, 1) * current_params.noise_level)) *[u_total; v_total] ;
            % --- Compute Jacobian with respect to position P ---
            J{j+1} = jacobian(current_vec{j+1}, P);  % 2x2 matrix
            outputCurrentNames{j+1} = ['current_test_', num2str(j-current_params.num_ensemble_members)];
            outputGradientNames{j+1} = ['gradient_test_', num2str(j-current_params.num_ensemble_members)];
        end
    end

    outputCurrentNames{current_params.num_ensemble_members+1} = 'current_out_avg';
    outputGradientNames{current_params.num_ensemble_members+1} = 'gradient_out_avg';

    ocean_func_single = Function('ocean_current_single', ...
        {P, t}, ...
        current_vec, ...
        {'pos_in', 't_in'}, ...
        outputCurrentNames);
    
    % For linear approximation, we need to evaluate current and Jacobian at P0
    % Create the linear approximation expression directly
    % f(P) ≈ f(P0) + J(P0) * (P - P0)
    
    % Create functions to evaluate current and Jacobian at any point
    % current_at_point = Function('current_at_point', ...
    %     {P, t}, ...
    %     {current_vec}, ...
    %     {'pos_in', 't_in'}, ...
    %     {'current_out'});
    % jacobian_at_point = Function('jacobian_at_point', ...
    %     {P, t}, ...
    %     {J}, ...
    %     {'pos_in', 't_in'}, ...
    %     {'jacobian_out'});
    
    % Create linear approximation using the reference point P0
    % current_at_P0 = current_at_point(P0, t);  % f(P0, t)
    % J_at_P0 = jacobian_at_point(P0, t);       % J(P0, t)
    
    % Linear approximation: f(P0,t) + J(P0,t) * (P - P0)
    % linear_approx = current_at_P0 + J_at_P0 * (P - P0);
    % linear_approx = substitute(current_vec, P, P0) + substitute(J, P, P0) * (P - P0);
    
    % ocean_linear_func_single = Function('ocean_current_linear_single', ...
    %     {P, P0, t}, ...
    %     {linear_approx}, ...
    %     {'pos_in', 'pos_ref', 't_in'}, ...
    %     {'current_out'});

    ocean_gradient_func = Function('ocean_gradient_single', ... % Name the base function
        {P, t}, ...
        J, ...
        {'pos_in', 't_in'}, ...
        outputGradientNames);

    % --- Create Mapped Functions ---
    % Map over the first input ('pos_in') using serial execution
    ocean_func = ocean_func_single.map(1, 'serial');
    % ocean_linear_approx_func = ocean_linear_func_single.map(1, 'serial');
    ocean_gradient_func = ocean_gradient_func.map(1, 'serial');

    % n_threads = feature('numcores');
    % ocean_func = ocean_func_single.map(4, 'thread', n_threads);
    % ocean_gradient_func = ocean_gradient_func.map(4, 'thread', n_threads);

end 