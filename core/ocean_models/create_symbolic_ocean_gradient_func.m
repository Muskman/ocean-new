% create_symbolic_ocean_gradient_func.m
function ocean_gradient_func = create_symbolic_ocean_gradient_func(current_params, use_mx)
% CREATE_SYMBOLIC_OCEAN_GRADIENT_FUNC Creates a symbolic CasADi function for ocean current gradient.
%
% Inputs:
%   current_params - Structure containing current parameters
%   use_mx        - Boolean flag (true for MX symbols, false for SX)
%
% Output:
%   ocean_gradient_func - CasADi function that calculates current gradient at given positions

    import casadi.* % Import CasADi package

    % --- Create Symbolic Variables ---
    if use_mx
        V = MX; % Use MX directly after import
    else
        V = SX; % Use SX directly after import
    end

    % Single position input (2x1)
    P = V.sym('P', 2, 1);
    t = V.sym('t', 1, 1);

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
            x0 = x0_base + 5 * sin(0.1 * t);
            y0 = y0_base + 3 * cos(0.07 * t);
            Gamma = Gamma_base * (1 + 0.1*cos(0.05*t));
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

    % --- Calculate Gradient ---
    % Create Jacobian matrix
    J = jacobian([u_total; v_total], P); % Use jacobian directly after import

    % --- Create Function ---
    % Create function that takes a single position and returns gradient
    ocean_gradient_func = Function('ocean_gradient_single', ... % Name the base function
        {P, t}, ...
        {J}, ...
        {'pos_in', 't_in'}, ...
        {'gradient_out'});

    % --- Create Mapped Function ---
    % Map over the first input ('pos_in') using serial execution.
    % The simpler map signature usually handles name inheritance.
    ocean_gradient_func_mapped = ocean_gradient_func.map(1, 'serial');

    % --- Return Mapped Function ---
    ocean_gradient_func = ocean_gradient_func_mapped;

end 