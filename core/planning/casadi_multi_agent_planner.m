% casadi_multi_agent_planner.m
function planned_trajectories = casadi_multi_agent_planner(agents, env_params, current_params, sim_params, agent_params)
% CASADI_MULTI_AGENT_PLANNER Solves the trajectory planning problem using CasADi NLP.
% Formulation: Direct Transcription (positions are decision variables).

    import casadi.*

    % --- Problem Setup ---
    N_agents = length(agents);
    T = sim_params.planning_horizon; % Number of control intervals
    dt = sim_params.dt;

    % Create Symbolic Ocean Current Function (Mapped version)
    ocean_current_func = create_symbolic_ocean_func(current_params, true); % Use MX

    % --- Decision Variables ---
    % P_sym: 2*N_agents x (T+1) matrix. Column k+1 is state [x1;y1;x2;y2...] at time k*dt
    P_sym = MX.sym('P', 2*N_agents, T+1);

    % --- Objective Function (Minimize Squared Control Effort Proxy) ---
    J = 0; % Initialize objective
    for k = 0:T-1 % Loop through time intervals 0 to T-1
        P_k = P_sym(:, k+1);        % State vector at start of interval k
        P_k_plus_1 = P_sym(:, k+2); % State vector at end of interval k

        % Reshape state vector to 2xN_agents matrix for current function
        Pos_k_matrix = reshape(P_k, 2, N_agents);

        % Calculate currents at time k*dt for all agents
        Currents_k_matrix = ocean_current_func(Pos_k_matrix, k*dt);
        Currents_k_vec = reshape(Currents_k_matrix, 2*N_agents, 1);

        % Required ground displacement during interval k
        disp_ground = P_k_plus_1 - P_k;

        % Implied control displacement (proportional to control velocity * dt)
        disp_control = disp_ground - Currents_k_vec * dt;

        % Accumulate squared L2 norm of control displacement
        J = J + sumsqr(disp_control);
    end

    % --- Constraints --- 
    g = {}; % Constraint expressions cell array
    lbg = []; ubg = []; % Lower and upper bounds for constraints

    % 1. Initial Position Constraint
    initial_pos_vec = reshape(cat(2, agents.position), 2*N_agents, 1);
    g = {g{:}, P_sym(:, 1) - initial_pos_vec};
    lbg = [lbg; zeros(2*N_agents, 1)-eps]; % Equality constraint LBG=UBG=0
    ubg = [ubg; zeros(2*N_agents, 1)+eps];

    % 2. Final Position Constraint (Individual or Formation Goal)
    final_pos_vec = reshape(cat(2, agents.goal), 2*N_agents, 1);
    g = {g{:}, P_sym(:, T+1) - final_pos_vec};
    lbg = [lbg; zeros(2*N_agents, 1)-eps]; % Equality constraint
    ubg = [ubg; zeros(2*N_agents, 1)+eps];

    % 3. Maximum Displacement / Control Constraint
    max_control_disp_sq = (agent_params.max_speed * dt)^2;
    for k = 0:T-1 % Loop intervals
        P_k = P_sym(:, k+1); P_k_plus_1 = P_sym(:, k+2);
        Pos_k_matrix = reshape(P_k, 2, N_agents);
        Currents_k_matrix = ocean_current_func(Pos_k_matrix, k*dt);
        Currents_k_vec = reshape(Currents_k_matrix, 2*N_agents, 1);
        disp_ground = P_k_plus_1 - P_k;
        disp_control = disp_ground - Currents_k_vec * dt;
        disp_control_matrix = reshape(disp_control, 2, N_agents);

        for i = 1:N_agents % Loop agents
            disp_control_norm_sq_per_agent = sumsqr(disp_control_matrix(:, i)); % 1xN_agents
            g = {g{:}, disp_control_norm_sq_per_agent};
            lbg = [lbg; zeros(1, 1)]; % Lower bound is 0
            ubg = [ubg; max_control_disp_sq * ones(1, 1)]; % Upper bound
        end
    end

    % 4. Obstacle Avoidance Constraints
    if ~isempty(env_params.obstacles)
        num_obstacles = length(env_params.obstacles);
        obs_centers_matrix = cat(2, env_params.obstacles.center); % 2xNumObs
        obs_radii = cat(2, env_params.obstacles.radius);       % 1xNumObs

        for k = 1:T % Loop time steps k=1...T (state index k+1)
            P_k_plus_1 = P_sym(:, k+1);
            Pos_k_plus_1_matrix = reshape(P_k_plus_1, 2, N_agents);
            for i = 1:N_agents % Loop agents
                Agent_pos_ik = Pos_k_plus_1_matrix(:, i);
                % Vectorized distance check for all obstacles
                dist_sq_all_obs = sumsqr(Agent_pos_ik - obs_centers_matrix);
                min_dist_sq_all_obs = (agent_params.radius + obs_radii + agent_params.safety_margin).^2;

                g = {g{:}, dist_sq_all_obs};
                lbg = [lbg; min_dist_sq_all_obs']; % Must be >= min_dist_sq (ensure column)
                ubg = [ubg; inf*ones(num_obstacles,1)];       % No upper bound
            end
        end
    end

    % 5.1 Inter-Agent Collision Avoidance Constraints (All Pairs) 
    % Create indices for all agent pairs (upper triangular)
    % min_dist_agent_sq = (2 * agent_params.radius + agent_params.safety_margin)^2;
    % [i_idx, j_idx] = find(triu(ones(N_agents), 1));
    % n_pairs = length(i_idx);
    % % For each time step, compute all distances at once
    % for k = 1:T
    %     P_k_plus_1 = P_sym(:, k+1);
    %     Pos_matrix = reshape(P_k_plus_1, 2, N_agents);
        
    %     % Vectorized distance computation
    %     pos_i = Pos_matrix(:, i_idx);  % 2 x n_pairs
    %     pos_j = Pos_matrix(:, j_idx);  % 2 x n_pairs
    %     dist_sq_vec = sum((pos_i - pos_j).^2, 1);  % 1 x n_pairs
        
    %     % Add all constraints for this time step at once
    %     g = {g{:}, dist_sq_vec'};
    %     lbg = [lbg; min_dist_agent_sq * ones(n_pairs, 1)]; % Must be >= min_dist_agent_sq
    %     ubg = [ubg; inf * ones(n_pairs, 1)];
    % end

    % 5.2 Inter-Agent Collision Avoidance Constraints (Minimum distance)
    % min_dist_agent_sq = (2 * agent_params.radius + agent_params.safety_margin)^2;
    % for k = 1:T
    %     Pos_k = reshape(P_sym(:, k+1), 2, N_agents);
    %     for i = 1:N_agents-1
    %         % Distances to all other agents
    %         other_indices = setdiff(i:N_agents, i);
    %         distances_sq = sum((Pos_k(:,i) - Pos_k(:,other_indices)).^2, 1);
    %         min_dist_sq = min(distances_sq);
            
    %         g = {g{:}, min_dist_sq};
    %         lbg = [lbg; min_dist_agent_sq];
    %         ubg = [ubg; inf];
    %     end
    % end

    % 6. Formation Constraints (Optional - if formation_enabled)
    if sim_params.formation_enabled && N_agents > 1
        formation_tolerance_sq_sum = (agent_params.formation_tolerance^2) * N_agents; % Example overall tolerance
        Target_relative = agent_params.formation_relative_positions; % 2xN_agents
        for k = 1:T % Loop time steps (can exclude final step if only final formation matters)
            P_k_plus_1 = P_sym(:, k+1);
            Pos_k_plus_1_matrix = reshape(P_k_plus_1, 2, N_agents);
            % Agent Centroid Constraint (Simplified: pull towards mean)
            % Calculate the centroid of all agents at this step using CasADi functions
            Centroid_k = sum2(Pos_k_plus_1_matrix) / N_agents;
            Relative_k = Pos_k_plus_1_matrix - Centroid_k; % Current relative positions
            dev_sq = sumsqr(Relative_k - Target_relative); % Squared deviation summed over agents
            g = {g{:}, dev_sq};
            lbg = [lbg; 0]; % Lower bound 0
            ubg = [ubg; formation_tolerance_sq_sum]; % Upper bound based on tolerance
        end
    end


    % --- NLP Definition --- 
    % Decision variables vector
    w = P_sym(:);
    nlp = struct('f', J, 'x', w, 'g', vertcat(g{:}));

    % --- Solver Options --- 
    opts = struct;
    opts.ipopt.print_level = 3; % 0=quiet, 3=default, 5=verbose
    opts.ipopt.max_iter = 2000;   % Limit iterations (Increased slightly)
    opts.ipopt.tol = 1e-10;      % Solver tolerance
    % opts.ipopt.acceptable_tol = 1e-3; % Allow slightly looser tolerance if struggling
    opts.print_time = 1;
    % Add warm start options if available solver supports them
    % opts.ipopt.warm_start_init_point = 'yes';
    % opts.ipopt.warm_start_bound_push = 1e-9;
    % opts.ipopt.warm_start_mult_bound_push = 1e-9;

    solver = nlpsol('solver', 'ipopt', nlp, opts);

    % --- Bounds on Decision Variables --- 
    lbx = -inf(size(w)); % Lower bounds
    ubx = inf(size(w));  % Upper bounds
    % Apply environment bounds
    for k = 0:T % Loop time points
        for i = 1:N_agents
            idx_x = (k * 2 * N_agents) + 2*i - 1;
            idx_y = (k * 2 * N_agents) + 2*i;
            lbx(idx_x) = env_params.x_limits(1)+agent_params.radius; ubx(idx_x) = env_params.x_limits(2)-agent_params.radius;
            lbx(idx_y) = env_params.y_limits(1)+agent_params.radius; ubx(idx_y) = env_params.y_limits(2)-agent_params.radius;
        end
    end

    % --- Initial Guess --- 
    w0 = zeros(size(w));
    P0 = zeros(2*N_agents, T+1);
    for i = 1:N_agents
        start_pos = agents(i).position;
        goal_pos = agents(i).goal;
        % Linear interpolation (can be improved with fancier guess)
        interp_x = linspace(start_pos(1), goal_pos(1), T+1);
        interp_y = linspace(start_pos(2), goal_pos(2), T+1);
        P0(2*i-1, :) = interp_x;
        P0(2*i, :) = interp_y;
    end
    w0 = P0(:);

    % --- Solve the NLP --- 
    planned_trajectories = cell(N_agents, 1); % Initialize output
    try
        sol = solver('x0', w0, 'lbx', lbx, 'ubx', ubx, 'lbg', lbg, 'ubg', ubg);

        % --- Process Solution --- 
        stats = solver.stats();
        if stats.success || strcmp(stats.return_status, 'Solve_Succeeded') || strcmp(stats.return_status, 'Solved_To_Acceptable_Level')
            if ~stats.success
                 fprintf('CasADi Planner: Warning! Solved to acceptable level.\n');
            end
            fprintf('CasADi Planner: Success! Objective: %.4f\n', full(sol.f));
            w_opt = full(sol.x);
            P_opt = reshape(w_opt, 2*N_agents, T+1);

            % Format output for the simulation
            for i = 1:N_agents
                agent_traj = P_opt(2*i-1 : 2*i, :); % Extract 2x(T+1) trajectory
                planned_trajectories{i} = struct('planned_positions', agent_traj);
            end
        else
            fprintf('CasADi Planner: Solver FAILED! Status: %s\n', stats.return_status);
            % Fallback: Keep previous plan or stop (return empty/zero velocity plan)
            % Returning empty will cause agent to stop based on main loop logic
            for i = 1:N_agents; planned_trajectories{i} = struct('planned_positions', []); end
        end

    catch ME
        fprintf('CasADi Planner: Error during solve! %s\n', ME.message);
        % Fallback on error
        for i = 1:N_agents; planned_trajectories{i} = struct('planned_positions', []); end
    end

end % End of function 