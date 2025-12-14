% test_multi_agent_planner.m
function [planned_trajectories, metrics] = fullopt_multi_agent_planner(agents, env_params, current_params, sim_params, agent_params)
    % solve full optimization problem for multi-agent planning
    
    import casadi.*
    
    % --- Create Problem Builder ---
    config = ProblemBuilder.getDefaultConfig();
    % You can customize configuration here if needed
    % config.collision_method = 'pairwise'; % or 'minimum_distance'
    % config.enable_formation_constraints = false;
    
    builder = ProblemBuilder(agents, env_params, current_params, sim_params, agent_params, config);

    % --- Solver Options ---
    opts = struct;
    opts.ipopt.print_level = 0;     % 0=quiet, 3=default, 5=verbose
    opts.ipopt.max_iter = 2000;     % Limit iterations
    opts.ipopt.tol = 1e-6;          % Solver tolerance
    opts.print_time = 1;
    opts.ipopt.warm_start_init_point = 'yes';
    opts.expand = true;

    % --- Start timing ---
    tic;
    
    % --- Build Parameterized NLP ---
    nlp = builder.getParameterizedNLP();
    [lbg, ubg] = builder.getParameterizedConstraintBounds();
    w0 = builder.getInitialGuess();
    builder.ensemble_samples = ones(current_params.num_ensemble_members, 1);
    
    solver = nlpsol('solver', 'ipopt', nlp, opts);
    formulation_time = toc;
    fprintf('Time taken to formulate problem: %.2f seconds\n', formulation_time);
    % --- Solve the Parameterized NLP ---
    planned_trajectories = cell(length(agents), 1); % Initialize output
    try
        p0 = [w0; builder.ensemble_samples];
        sol = solver('x0', w0, 'lbx', builder.lbx, 'ubx', builder.ubx, 'p', p0, 'lbg', lbg, 'ubg', ubg);
        
        % Record training time
        training_time = toc-formulation_time;
        
        % --- Process Solution ---
        stats = solver.stats();
        if stats.success || strcmp(stats.return_status, 'Solve_Succeeded') || strcmp(stats.return_status, 'Solved_To_Acceptable_Level')
            if ~stats.success
                fprintf('ProblemBuilder Planner: Warning! Solved to acceptable level.\n');
            end
            fprintf('ProblemBuilder Planner: Success! Objective: %.4f\n', full(sol.f));
            w_opt = full(sol.x);
            
            % Get problem dimensions
            N_agents = length(agents);
            T = sim_params.planning_horizon;
            P_opt = reshape(w_opt, 2*N_agents, T+1);
            
            % Format output for the simulation
            for i = 1:N_agents
                agent_traj = P_opt(2*i-1 : 2*i, :); % Extract 2x(T+1) trajectory
                planned_trajectories{i} = struct('planned_positions', agent_traj);
            end
            
            builder.updateReferenceTrajectory(P_opt);
            builder.buildBenchmarkingExpressions();
            metrics = builder.getBenchmarkingMetrics();
            metrics.formulation_time = formulation_time;
            metrics.training_time = training_time;
        else
            fprintf('ProblemBuilder Planner: Solver FAILED! Status: %s\n', stats.return_status);
            % Fallback: Keep previous plan or stop (return empty/zero velocity plan)
            % Returning empty will cause agent to stop based on main loop logic
            for i = 1:length(agents)
                planned_trajectories{i} = struct('planned_positions', []);
            end
        end
        
    catch ME
        fprintf('ProblemBuilder Planner: Error during solve! %s\n', ME.message);
        % Fallback on error
        for i = 1:length(agents)
            planned_trajectories{i} = struct('planned_positions', []);
        end
    end
    
end % End of function 