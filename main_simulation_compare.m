% main_simulation.m
clear; clc; close all;

% --- Load Configuration ---
% Load all simulation parameters from configuration function
% This replaces the hardcoded parameter definitions
[sim_params, env_params, current_params, agent_params, num_agents, video_params] = simulation_config();

% --- Handle Multiple Algorithms ---
if iscell(sim_params.algo)
    algorithms_to_run = sim_params.algo;
else
    algorithms_to_run = {sim_params.algo};
end

% Store results for all algorithms
all_metrics = struct();
all_final_agents = cell(length(algorithms_to_run), 1);
all_state_histories = cell(length(algorithms_to_run), 1);

% --- Start Profiling ---
% Apply profiling configuration
% profile on -detail builtin -timer performance;

fprintf('Running comparison for %d algorithm(s): %s\n', length(algorithms_to_run), strjoin(algorithms_to_run, ', '));

% --- Run Simulation for Each Algorithm ---
for algo_idx = 1:length(algorithms_to_run)
    current_algo = algorithms_to_run{algo_idx};
    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('Running Algorithm: %s (%d/%d)\n', current_algo, algo_idx, length(algorithms_to_run));
    fprintf('%s\n', repmat('=', 1, 60));

    % Set current algorithm in sim_params
    sim_params.algo = current_algo;

    % Disable video for all algorithms
    video_params.enabled = false;

    % --- Setup Random Seed ---
    % Apply random seed configuration
    rng(8, "philox");  % Values from config could be used here in future


    fprintf('Initializing environment, agents, and currents...\n');

    % --- Initialization ---
    % Pass sim_params and agent_params to initializer
    agents = initialize_agents(num_agents, agent_params, env_params, sim_params);
    state_history = cell(num_agents, 1);
    for i = 1:num_agents
        state_history{i} = nan(2, sim_params.time_steps + 1);
        state_history{i}(:, 1) = agents(i).position;
        agents(i).current_plan = []; agents(i).plan_start_step = -inf;
    end

    % --- Initialize Visualization ---
    visualizer = SimulationVisualizer(sim_params, env_params, current_params, agent_params, num_agents, video_params);
    visualizer.initialize(agents);

    fprintf('Starting simulation loop...\n');

% --- Simulation Loop ---
for t_idx = 1:sim_params.time_steps
    current_time = t_idx * sim_params.dt;

    % --- Get Noisy Estimates for ALL Agents at Once ---
    % 1a. Collect current positions of all agents
    current_agent_positions = cat(2, agents.position); % 2xN matrix

    % 1b. Call the vectorized noisy estimation function
    [estimated_currents_cell, estimated_gradients_cell] = ...
        get_noisy_current_estimate(current_agent_positions, current_time, current_params, env_params); % Pass flag
    % estimated_gradients_cell contains 2x2 matrices or []

    % 1c. Distribute the estimates back to individual agents
    for i = 1:num_agents
        agents(i).estimated_current = estimated_currents_cell{i};
        % Assign gradient if calculated, otherwise assign a default (e.g., zeros)
        if ~isempty(estimated_gradients_cell{i})
            agents(i).estimated_gradient = estimated_gradients_cell{i};
        else
            agents(i).estimated_gradient = zeros(2, 2); % Default value when not calculated
            % Alternatively use NaN(2,2) if that's more appropriate for planner
            % agents(i).estimated_gradient = NaN(2, 2);
        end
    end
    % --- End Noisy Estimate Block ---

    % 2. Plan Trajectories
    if mod(t_idx - 1, sim_params.replan_interval) == 0 || t_idx == 1
        fprintf('Step %d (t=%.1f): Re-planning...\n', t_idx, current_time);

        % Select planner based on algorithm parameter
        switch sim_params.algo
            case 'fullOpt'
                [planned_trajectories, metrics] = fullopt_multi_agent_planner(agents, env_params, current_params, sim_params, agent_params);
            case {'sca', 'ssca'}
                [planned_trajectories, metrics] = sca_multi_agent_planner(agents, env_params, current_params, sim_params, agent_params);
            otherwise
                error('Unknown algorithm: %s. Valid options: fullOpt, sca, ssca', sim_params.algo);
        end
        
        % Stop and view results
        % profile off
        % profile viewer
        % profview  % Alternative viewer
        % profsave

        % Distribute the plan (or fallback)
        for i = 1:num_agents
            if ~isempty(planned_trajectories{i}) && isfield(planned_trajectories{i}, 'planned_positions') && size(planned_trajectories{i}.planned_positions, 2) > 1
                agents(i).current_plan = planned_trajectories{i}.planned_positions;
                agents(i).plan_start_step = t_idx;
            else
                 % Planner failed or returned empty, maintain previous state/stop
                 fprintf('Warning: No valid plan assigned for agent %d at step %d.\n', i, t_idx);
                 if isempty(agents(i).current_plan); agents(i).control_velocity = [0;0]; end % Stop if never planned
            end
        end
    end

    % 3. Determine Control Velocity from Plan
    for i = 1:num_agents
        plan = agents(i).current_plan; start_step = agents(i).plan_start_step; time_into_plan = t_idx - start_step;
        if ~isempty(plan) && time_into_plan < size(plan, 2) -1
            target_pos = plan(:, time_into_plan + 2);
            required_displacement = target_pos - agents(i).position;
            required_ground_velocity = required_displacement / sim_params.dt;
            required_control_velocity = required_ground_velocity - agents(i).estimated_current;
            speed = norm(required_control_velocity);
            if speed > agent_params.max_speed; agents(i).control_velocity = required_control_velocity * (agent_params.max_speed / speed);
            else; agents(i).control_velocity = required_control_velocity; end
        else; agents(i).control_velocity = [0; 0]; agents(i).current_plan = []; end
    end

    % 4. Update Agent States
    for i = 1:num_agents
        % agents(i) = update_agent_state(agents(i), current_time, sim_params.dt, current_params, agent_params);
        plan = agents(i).current_plan;
        agents(i).position = plan(:, time_into_plan + 2);
        state_history{i}(:, t_idx + 1) = agents(i).position;
    end

    % 5. Collision Check (TODO)
    % 6. Goal Check (TODO)

    % 7. Visualization
    visualizer.update(agents, current_time, t_idx, state_history);

    % Progress Indicator
    if mod(t_idx, 100) == 0; fprintf('Simulated %.1f seconds...\n', current_time); end
end % End simulation loop

    fprintf('Algorithm %s finished after %.1f seconds.\n', current_algo, sim_params.T_final);

    % Store results for this algorithm
    all_metrics.(current_algo) = metrics;
    all_final_agents{algo_idx} = agents;
    all_state_histories{algo_idx} = state_history;

    % --- Final Visualization ---
    visualizer.finalize(agents, state_history);

end % End algorithm loop

% --- Print Comparison Metrics ---
fprintf('\n%s\n', repmat('=', 1, 120));
fprintf('                                      ALGORITHM COMPARISON METRICS\n');
fprintf('%s\n', repmat('=', 1, 120));

% Create header with algorithm names
header_format = '| %-25s ';
data_format = '| %-25s ';
for i = 1:length(algorithms_to_run)
    header_format = [header_format, '| %-18s '];
    data_format = [data_format, '| %18.4f '];
end
header_format = [header_format, '|\n'];
data_format = [data_format, '|\n'];

% Print header
fprintf(header_format, 'Metric', algorithms_to_run{:});
fprintf('|%s%s%s|\n', repmat('-', 1, 27), repmat('+', 1, 1), repmat('-', 1, 18*length(algorithms_to_run)+1));

% Energy (Training)
energy_vals = cellfun(@(algo) all_metrics.(algo).training_energy, algorithms_to_run);
fprintf(data_format, 'Energy (Training)', energy_vals);

% Energy (Testing)
energy_test_vals = cellfun(@(algo) all_metrics.(algo).testing_energy, algorithms_to_run);
fprintf(data_format, 'Energy (Testing)', energy_test_vals);

% Control Violations (Training)
control_viol_train = cellfun(@(algo) all_metrics.(algo).training_control_constraint_violations, algorithms_to_run);
fprintf('| %-25s ', 'Control Viol (Train)');
for i = 1:length(algorithms_to_run)
    fprintf('| %18d ', control_viol_train(i));
end
fprintf('|\n');

% Control Violations (Testing)
control_viol_test = cellfun(@(algo) all_metrics.(algo).testing_control_constraint_violations, algorithms_to_run);
fprintf('| %-25s ', 'Control Viol (Test)');
for i = 1:length(algorithms_to_run)
    fprintf('| %18d ', control_viol_test(i));
end
fprintf('|\n');

fprintf('%s\n', repmat('-', 1, 120));

% Constraint Violations
constraint_viol = cellfun(@(algo) all_metrics.(algo).training_constraint_violations, algorithms_to_run);
fprintf('| %-25s ', 'Constraint Violations');
for i = 1:length(algorithms_to_run)
    fprintf('| %18d ', constraint_viol(i));
end
fprintf('|\n');

% Formulation Time
form_time = cellfun(@(algo) all_metrics.(algo).formulation_time, algorithms_to_run);
fprintf(data_format, 'Formulation Time (s)', form_time);

% Training Time
train_time = cellfun(@(algo) all_metrics.(algo).training_time, algorithms_to_run);
fprintf(data_format, 'Training Time (s)', train_time);

fprintf('%s\n\n', repmat('=', 1, 120));