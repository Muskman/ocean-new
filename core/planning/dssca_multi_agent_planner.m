% sca_multi_agent_planner.m
function [planned_trajectories, metrics] = dssca_multi_agent_planner(agents, env_params, current_params, sim_params, agent_params)
    % SCA and stochastic-SCAalgorithm for multi-agent trajectory planning
    
    import casadi.*
    
    % --- Create Problem Builder ---
    config = ProblemBuilder.getDefaultConfig();
    % You can customize configuration here if needed
    config.use_linear_approximation = true;
    config.enable_formation_constraints = false;
    % config.enable_collision_constraints = false;
     
    builder = ProblemBuilderD(agents, env_params, current_params, sim_params, agent_params, config);
    
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

    max_outer_iterations = sim_params.max_outer_iterations * length(agents);
    
    % --- Start timing ---
    tic_formulation = tic; formulation_time = 0; solve_time = 0;
    

    for iter = 1:max_outer_iterations
        fprintf('-------------------------\n')
        fprintf('Outer Iteration %d\n',iter)
        fprintf('-------------------------\n')

        % --- Select an agent for planning ---
        % Case 1:first half of iterations, plan for agent 1, second half of iterations, plan for agent 2
        % if iter<=max_outer_iterations/2
        %     idx_agent = 1;
        % else
        %     idx_agent = 2;
        % end
        
        % Case 2: select agent based on iteration number
        idx_agent = mod(iter+1, length(agents)) + 1;
        
        % Case 3: random selection of agent
        % idx_agent = randi(length(agents));
        
        fprintf('Selected agent %d for planning.\n', idx_agent);

        % --- Build Parameterized NLP ---
        if iter == 1
            % Build parameterized NLP
            nlp = builder.getParameterizedNLP(idx_agent);
            % keyboard;
            [lbg, ubg] = builder.getParameterizedConstraintBounds();
            w0 = builder.getInitialGuess(idx_agent);
            
            % Create solver once
            solver = nlpsol('solver', 'ipopt', nlp, opts);
            fprintf('Parameterized solver created for agent %d.\n', idx_agent);
            formulation_time = formulation_time + toc(tic_formulation);
            fprintf('Time taken to formulate problem: %.2f seconds\n', formulation_time);

            % keyboard;
            % [H, g] = builder.getQPMatrices();
        else
            % Update reference trajectory (P0) for subsequent iterations
            % No need to rebuild NLP - just update P0 parameter
            % w0 = builder.P0(2*idx_agent-1:2*idx_agent, :); % Use previous solution as initial guess
            % w0 = w0(:);
            w0 = builder.getInitialGuess(idx_agent);
            fprintf('Using previous solution as warm start.\n');

            tic_formulation = tic;
            nlp = builder.getParameterizedNLP(idx_agent, sample_idx);
            solver = nlpsol('solver', 'ipopt', nlp, opts);
            formulation_time = formulation_time + toc(tic_formulation);
        end
        
        % --- Solve the Parameterized NLP ---
        planned_trajectories = cell(length(agents), 1); % Initialize output
        try
            % Pass current P0 as parameter to solver
            if any(strcmp(sim_params.algo, 'dssca'))
                builder.ensemble_samples = zeros(current_params.num_ensemble_members, 1);
                sample_idx = randperm(current_params.num_ensemble_members,1);
                builder.ensemble_samples(sample_idx) = 1;
            else
                builder.ensemble_samples = ones(current_params.num_ensemble_members, 1);
            end

            % p0 = [w0; builder.ensemble_samples]; 
            tic_solve = tic;
            % sol = solver('x0', w0, 'lbx', builder.lbx, 'ubx', builder.ubx, 'p', p0, 'lbg', lbg, 'ubg', ubg);
            sol = solver('x0', w0, 'lbx', builder.lbx, 'ubx', builder.ubx, 'lbg', lbg, 'ubg', ubg);
            solve_time = solve_time + toc(tic_solve);

            % --- Process Solution ---
            stats = solver.stats();
            if stats.success || strcmp(stats.return_status, 'Solve_Succeeded') || strcmp(stats.return_status, 'Solved_To_Acceptable_Level')
                if ~stats.success
                    fprintf('ProblemBuilder Planner: Warning! Solved to acceptable level.\n');
                end
                fprintf('ProblemBuilder Planner: Success! Objective: %.4f\n', full(sol.f));
                fprintf('ProblemBuilder Planner: Learning rate: %.4f | Gradient tracking weight: %.4f\n', builder.learning_rate(idx_agent), builder.gradient_tracking_weight(idx_agent));
                fprintf('ProblemBuilder Planner: Stochastic gradient norm: %.4f\n', builder.stochastic_gradient_norm);
                fprintf('ProblemBuilder Planner: Step %.4f\n', norm(builder.P0(:)-builder.P0_old{idx_agent}(:)));
                w_opt = full(sol.x);
                
                % Get problem dimensions
                N_agents = length(agents);
                T = sim_params.planning_horizon;
                P_opt_agent = reshape(w_opt, 2, T+1);
                
                % Update reference trajectory
                P_current = builder.P0;
                P_current(2*idx_agent-1:2*idx_agent,:) = builder.P0(2*idx_agent-1:2*idx_agent,:) + builder.learning_rate(idx_agent) * (P_opt_agent - builder.P0(2*idx_agent-1:2*idx_agent,:));
                builder.updateReferenceTrajectory(P_current, idx_agent);
                
                if iter == max_outer_iterations
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
                metrics.optimization_status = stats.return_status;
            end
            
        catch ME
            fprintf('ProblemBuilder Planner: Error during solve! %s\n', ME.message);
            % Fallback on error
            for i = 1:length(agents)
                planned_trajectories{i} = struct('planned_positions', []);
            end
            metrics.inner_error_message = ME.message;
        end
    end
    
end % End of function 