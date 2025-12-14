% sca_multi_agent_planner.m
function [planned_trajectories, metrics] = sca_multi_agent_planner(agents, env_params, current_params, sim_params, agent_params)
    % SCA and stochastic-SCAalgorithm for multi-agent trajectory planning
    
    import casadi.*
    
    % --- Create Problem Builder ---
    config = ProblemBuilder.getDefaultConfig();
    % You can customize configuration here if needed
    config.use_linear_approximation = true;
     
    builder = ProblemBuilder(agents, env_params, current_params, sim_params, agent_params, config);
    
    % --- Solver Options ---
    opts = struct;
    opts.ipopt.print_level = 0;     % 0=quiet, 3=default, 5=verbose
    % opts.ipopt.max_iter = 2000;     % Limit iterations
    % opts.ipopt.tol = 1e-6;          % Solver tolerance
    % opts.print_time = 1;
    % opts.ipopt.warm_start_init_point = 'yes';
    opts.expand = true;


    % opts.fatrop.print_level = 0;     % 0=quiet, 3=default, 5=verbose
    % opts.debug = true;
    % opts.jit = true;
    % opts.jit_options.flags = '-O3';
    % opts.jit_options.compiler = 'clang';

    max_outer_iterations = sim_params.max_outer_iterations;
    learning_rate = sim_params.learning_rate;

    % --- Start timing ---
    tic; solve_time = 0;
    

    for iter = 1:max_outer_iterations
        fprintf('-------------------------\n')
        fprintf('Outer Iteration %d\n',iter)
        fprintf('-------------------------\n')

        % --- Build Parameterized NLP ---
        if iter == 1
            % Build parameterized NLP once
            nlp = builder.getParameterizedNLP();
            % keyboard;
            [lbg, ubg] = builder.getParameterizedConstraintBounds();
            w0 = builder.getInitialGuess();
            
            % Create solver once
            solver = nlpsol('solver', 'ipopt', nlp, opts);
            fprintf('Parameterized solver created once for all iterations.\n');
            formulation_time = toc;
            fprintf('Time taken to formulate problem: %.2f seconds\n', formulation_time);
        else
            % Update reference trajectory (P0) for subsequent iterations
            % No need to rebuild NLP - just update P0 parameter
            P_current = builder.P0 + learning_rate * (P_opt - builder.P0);
            builder.updateReferenceTrajectory(P_current);
            w0 = P_current(:); % Use previous solution as initial guess
            fprintf('Using previous solution as warm start.\n');
        end
        
        % --- Solve the Parameterized NLP ---
        planned_trajectories = cell(length(agents), 1); % Initialize output
        try
            % Pass current P0 as parameter to solver
            if strcmp(sim_params.algo, 'ssca')
                builder.ensemble_samples = zeros(current_params.num_ensemble_members, 1);
                sample_idx = randperm(current_params.num_ensemble_members,1);
                builder.ensemble_samples(sample_idx) = 1;
            else
                builder.ensemble_samples = ones(current_params.num_ensemble_members, 1);
            end

            p0 = [w0; builder.ensemble_samples]; tic_solve = tic;
            sol = solver('x0', w0, 'lbx', builder.lbx, 'ubx', builder.ubx, 'p', p0, 'lbg', lbg, 'ubg', ubg);
            solve_time = solve_time + toc(tic_solve);

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
                
                if iter == max_outer_iterations
                    P_current = builder.P0 + learning_rate * (P_opt - builder.P0);
                    builder.updateReferenceTrajectory(P_current);
                    training_time = solve_time;
                    builder.buildBenchmarkingExpressions();
                    metrics = builder.getBenchmarkingMetrics();
                    metrics.formulation_time = formulation_time;
                    metrics.training_time = training_time;

                    % Format output for the simulation
                    for i = 1:N_agents
                        agent_traj = P_current(2*i-1 : 2*i, :); % Extract 2x(T+1) trajectory
                        planned_trajectories{i} = struct('planned_positions', agent_traj);
                    end
                end
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
    end
    
end % End of function 