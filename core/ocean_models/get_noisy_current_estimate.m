% get_noisy_current_estimate.m - CORRECTED
function [estimated_current_cell, estimated_gradient_cell] = get_noisy_current_estimate(positions, time, current_params, env_params, calculate_gradient)
% GET_NOISY_CURRENT_ESTIMATE Calculates noisy current and optionally gradient for multiple positions using symbolic functions.
%
% Inputs:
%   positions          - 2xN matrix of [x; y] coordinates.
%   time               - Current simulation time.
%   current_params     - Structure with current parameters (incl. noise levels).
%   env_params         - Structure with environment parameters.
%   calculate_gradient - Boolean flag (true to calculate gradient, false otherwise).
%
% Outputs:
%   estimated_current_cell  - 1xN cell array, each cell contains [u; v] current estimate.
%   estimated_gradient_cell - 1xN cell array, each cell contains 2x2 gradient estimate if calculate_gradient is true, otherwise contains [].

    if nargin < 5
        calculate_gradient = true; % Default
    end

    N_points = size(positions, 2);

    % --- Create/Cache Symbolic Functions --- 
    persistent sym_ocean_current_func sym_ocean_gradient_func func_params_hash
    current_params_json = jsonencode(current_params); % Use JSON for reliable hashing
    new_hash = simple_string_hash(current_params_json);

    % Recreate functions if parameters changed or functions are empty
    recreate_current = isempty(sym_ocean_current_func) || isempty(func_params_hash) || ~strcmp(new_hash, func_params_hash);
    recreate_gradient = calculate_gradient && (isempty(sym_ocean_gradient_func) || isempty(func_params_hash) || ~strcmp(new_hash, func_params_hash));

    if recreate_current
        fprintf('Creating/Recreating symbolic CasADi current function...\n');
        [sym_ocean_current_func, sym_ocean_gradient_func] = create_symbolic_ocean_func(current_params, false); % Use SX for evaluation
    end
    if recreate_gradient
         fprintf('Creating/Recreating symbolic CasADi gradient function...\n');
        %  sym_ocean_gradient_func = create_symbolic_ocean_gradient_func(current_params, false); % Use SX for evaluation
    elseif ~calculate_gradient
         sym_ocean_gradient_func = []; % Ensure it's empty if not needed
    end
    if recreate_current || recreate_gradient
         func_params_hash = new_hash; % Update hash if anything was recreated
    end
    % --- End Function Creation/Caching ---

    % --- Evaluate Functions --- 
    % Evaluate current function (numerically)
    current_eval = sym_ocean_current_func('pos_in', positions, 't_in', time);
    true_current_matrix = full(current_eval.current_out_1); % taking the first current ensemble member

    % Evaluate gradient function if requested
    true_gradient_cell = cell(1, N_points);
    if calculate_gradient && ~isempty(sym_ocean_gradient_func)
        try
            gradient_eval = sym_ocean_gradient_func('pos_in', positions, 't_in', time);
            temp_gradient_cell = gradient_eval.gradient_out_1; % taking the first gradient ensemble member
            % Convert CasADi cell to MATLAB cell, applying full()
            for i = 1:N_points
                true_gradient_cell{i} = full(temp_gradient_cell{i});
            end
        catch ME_grad % Catch potential evaluation errors
            fprintf('ERROR evaluating symbolic gradient function: %s\n', ME_grad.message);
            calculate_gradient = false; % Disable further gradient attempts if evaluation fails
             for i = 1:N_points; true_gradient_cell{i} = []; end % Ensure output is empty
        end
    elseif calculate_gradient && isempty(sym_ocean_gradient_func)
         fprintf('Warning: Gradient requested but symbolic function handle is empty.\n');
         calculate_gradient = false; % Cannot calculate
         for i = 1:N_points; true_gradient_cell{i} = []; end % Ensure output is empty
    end
    % --- End Evaluation ---


    % Initialize output cell arrays
    estimated_current_cell = cell(1, N_points);
    estimated_gradient_cell = cell(1, N_points); % Initialize empty again

    % Noise parameters
    current_noise_std = current_params.noise_level;
    gradient_noise_std = current_params.gradient_noise_level;

    % --- Add Noise --- 
    for i = 1:N_points
        % Noise for current
        current_noise = current_noise_std * randn(2, 1);
        estimated_current_cell{i} = true_current_matrix(:, i) + current_noise;

        % Noise for gradient (only if calculated successfully)
        if calculate_gradient && ~isempty(true_gradient_cell{i}) % Check if gradient was actually computed
            gradient_noise = gradient_noise_std * randn(2, 2);
            estimated_gradient_cell{i} = true_gradient_cell{i} + gradient_noise;
        else
            estimated_gradient_cell{i} = []; % Ensure it's empty otherwise
        end
    end
    % --- End Noise Addition ---

end

% Simple hashing function (replace with a more robust one if needed)
function hash_str = simple_string_hash(str)
    try
        % Use MATLAB's digest function if available (more robust)
        hash_bytes = mlreportgen.utils.hash(str, 'SHA-256');
        hash_str = sprintf('%02x', hash_bytes);
    catch
        % Fallback simple hash (less reliable for complex changes)
        hash_val = 0;
        for k = 1:length(str)
            hash_val = mod(hash_val * 31 + double(str(k)), 2^32);
        end
        hash_str = sprintf('%x', hash_val);
    end
end 