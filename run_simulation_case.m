function all_metrics = run_simulation_case(sim_params, env_params, current_params, agent_params, video_params)
    % RUN_SIMULATION_CASE  Run a full multi-algorithm simulation for one configuration.
    %
    % Inputs:
    %   sim_params     - Simulation parameters (from simulation_config)
    %   env_params     - Environment parameters
    %   current_params - Ocean current parameters
    %   agent_params   - Agent parameters (includes agent_params.num_agents)
    %   video_params   - Video export parameters
    %
    % Output:
    %   all_metrics    - Struct with per-algorithm metrics plus case metadata

    num_agents = agent_params.num_agents;

    % --- Handle Multiple Algorithms ---
    algorithms_to_run = sim_params.algo;
    
    % Store results — include case metadata so the batch script has full context
    all_metrics = struct();
    all_metrics.num_agents             = num_agents;
    all_metrics.num_ensemble_members   = current_params.num_ensemble_members;
    all_metrics.noise_level            = current_params.noise_level;

    all_final_agents    = cell(length(algorithms_to_run), 1);
    all_state_histories = cell(length(algorithms_to_run), 1);

    fprintf('Running comparison for %d algorithm(s): %s\n', length(algorithms_to_run), strjoin(algorithms_to_run, ', '));
    random_seed = randi(1000);

    % --- Run Simulation for Each Algorithm ---
    for algo_idx = 1:length(algorithms_to_run)
        current_algo = algorithms_to_run{algo_idx};
        fprintf('\n%s\n', repmat('=', 1, 60));
        fprintf('Running Algorithm: %s (%d/%d)\n', current_algo, algo_idx, length(algorithms_to_run));
        fprintf('%s\n', repmat('=', 1, 60));

        sim_params.algo = current_algo;

        rng(random_seed, "philox");

        try
            fprintf('Initializing environment, agents, and currents...\n');

            % --- Initialization ---
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

                % 1. Get Noisy Estimates for All Agents
                current_agent_positions = cat(2, agents.position);
                [estimated_currents_cell, estimated_gradients_cell] = ...
                    get_noisy_current_estimate(current_agent_positions, current_time, current_params, env_params);

                for i = 1:num_agents
                    agents(i).estimated_current = estimated_currents_cell{i};
                    if ~isempty(estimated_gradients_cell{i})
                        agents(i).estimated_gradient = estimated_gradients_cell{i};
                    else
                        agents(i).estimated_gradient = zeros(2, 2);
                    end
                end

                % 2. Plan Trajectories
                if mod(t_idx - 1, sim_params.replan_interval) == 0 || t_idx == 1
                    fprintf('Step %d (t=%.1f): Re-planning...\n', t_idx, current_time);

                    switch sim_params.algo
                        case 'fullOpt'
                            [planned_trajectories, metrics] = fullopt_multi_agent_planner(agents, env_params, current_params, sim_params, agent_params);
                        case {'sca', 'ssca'}
                            [planned_trajectories, metrics] = sca_multi_agent_planner(agents, env_params, current_params, sim_params, agent_params);
                        case 'dssca'
                            [planned_trajectories, metrics] = dssca_multi_agent_planner(agents, env_params, current_params, sim_params, agent_params);
                        case 'eesto'
                            [planned_trajectories, metrics] = eesto_planner(agents, env_params, current_params, sim_params, agent_params);
                        case 'stomp'
                            [planned_trajectories, metrics] = stomp_planner(agents, env_params, current_params, sim_params, agent_params);
                        case 'astar'
                            [planned_trajectories, metrics] = astar_planner(agents, env_params, current_params, sim_params, agent_params);
                        otherwise
                            error('Unknown algorithm: %s. Valid options: fullOpt, sca, ssca, dssca, eesto, stomp, astar', sim_params.algo);
                    end

                    for i = 1:num_agents
                        if ~isempty(planned_trajectories{i}) && isfield(planned_trajectories{i}, 'planned_positions') && size(planned_trajectories{i}.planned_positions, 2) > 1
                            agents(i).current_plan = planned_trajectories{i}.planned_positions;
                            agents(i).plan_start_step = t_idx;
                        else
                            fprintf('Warning: No valid plan assigned for agent %d at step %d.\n', i, t_idx);
                            if isempty(agents(i).current_plan); agents(i).control_velocity = [0; 0]; end
                        end
                    end
                end

                % 3. Determine Control Velocity from Plan
                for i = 1:num_agents
                    plan = agents(i).current_plan; start_step = agents(i).plan_start_step; time_into_plan = t_idx - start_step;
                    if ~isempty(plan) && time_into_plan < size(plan, 2) - 1
                        target_pos = plan(:, time_into_plan + 2);
                        required_displacement = target_pos - agents(i).position;
                        required_ground_velocity = required_displacement / sim_params.dt;
                        required_control_velocity = required_ground_velocity - agents(i).estimated_current;
                        speed = norm(required_control_velocity);
                        if speed > agent_params.max_speed; agents(i).control_velocity = required_control_velocity * (agent_params.max_speed / speed);
                        else; agents(i).control_velocity = required_control_velocity; end
                    else; agents(i).control_velocity = [0; 0]; end
                end

                % 4. Update Agent States
                for i = 1:num_agents
                    plan = agents(i).current_plan;
                    agents(i).position = plan(:, time_into_plan + 2);
                    state_history{i}(:, t_idx + 1) = agents(i).position;
                end

                % 5. Visualization
                visualizer.update(agents, current_time, t_idx, state_history);

                if mod(t_idx, 100) == 0; fprintf('Simulated %.1f seconds...\n', current_time); end
            end % End simulation loop

            fprintf('Algorithm %s finished after %.1f seconds.\n', current_algo, sim_params.T_final);

            metrics.success       = true;
            metrics.error_message = '';

            all_final_agents{algo_idx}    = agents;
            all_state_histories{algo_idx} = state_history;

            visualizer.finalize(agents, state_history);

        catch ME
            warning('run_simulation_case:AlgorithmFailed', ...
                'Algorithm %s failed: %s', current_algo, ME.message);
            metrics               = struct();
            metrics.success       = false;
            metrics.error_message = ME.message;
        end

        all_metrics.(current_algo) = metrics;

    end % End algorithm loop

    print_metrics(all_metrics, algorithms_to_run);
end
