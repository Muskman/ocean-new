

function [planned_trajectories, metrics] = stomp_planner(agents, env_params, current_params, sim_params, agent_params)

    v_max = agent_params.max_speed;
    T = sim_params.planning_horizon;
    
    % num_its = sim_params.stomp_num_its;
    % decay_fact = sim_params.stomp_decay_fact;
    % N = sim_params.stomp_number_of_trajectories;
    % threshold = sim_params.stomp_threshold;
    % mag_step = sim_params.stomp_mag_step;
    % c_d = agent_params.drag_coefficient;

    num_its = 100;
    decay_fact = 0.99;
    N = 50;
    threshold = 1e-6;
    mag_step = 5*1e-2;
    c_d = 3;
    sim_params.initial_guess = 'straightline';
    
    opts.vis_baselines = false;
    opts.current_params = current_params;
    
    opts.n_obs = length(env_params.obstacles);
    opts.x_obs = cat(2,env_params.obstacles.center);
    opts.r_obs = cat(2,env_params.obstacles.radius);
    opts.r_a = agent_params.radius;

    % x_min = env_params.x_limits(1);
    % x_max = env_params.x_limits(2);
    % y_min = env_params.y_limits(1);
    % y_max = env_params.y_limits(2);

    x_min = 1; y_min = 1;
    x_max = 50; y_max = 50;
    
    % Initialize Ocean Current Grid
    nx_contour = 50; ny_contour = 50;
    x_vec_contour = linspace(env_params.x_limits(1), env_params.x_limits(2), nx_contour);
    y_vec_contour = linspace(env_params.y_limits(1), env_params.y_limits(2), ny_contour);
    [X_grid_contour, Y_grid_contour] = meshgrid(x_vec_contour, y_vec_contour);

    % --- Vectorized Calculation for Contour --- 
    contour_positions = [X_grid_contour(:)'; Y_grid_contour(:)']; % Create 2xN matrix
    [U_grid_contour_flat, V_grid_contour_flat] = calculate_ocean_current_vectorized(contour_positions, 0, current_params);
    Current_Mag_flat = sqrt(U_grid_contour_flat.^2 + V_grid_contour_flat.^2);
    % Reshape back to grid format
    U = reshape(U_grid_contour_flat, size(X_grid_contour));
    V = reshape(V_grid_contour_flat, size(Y_grid_contour));
    mag = reshape(Current_Mag_flat, size(X_grid_contour));

    builder = ProblemBuilder(agents, env_params, current_params, sim_params, agent_params, ProblemBuilder.getDefaultConfig());
    P0_final = builder.P0;

    planned_trajectories = cell(length(agents), 1);
    training_time = 0;
    
    for i = 1:length(agents)
        P0 = builder.P0(2*i-1:2*i,:); % P0(:, 0.5*T+1) = [];
        path = [P0; repmat(sim_params.dt, 1, T+1)]';
        % path = flipud(path);

        t_start = tic;
        
        [STOMP_path, STOMP_energy,cost_STOMP,V_rel_i,V_abs_i] = STOMP_dylan_mohan(v_max,N,num_its,decay_fact,...
                                                T+1,threshold,mag_step,X_grid_contour,Y_grid_contour,U,V,mag,path,c_d,...
                                                x_min,x_max,y_min,y_max,U,V,opts);

        training_time = training_time + toc(t_start);

        planned_trajectories{i} = struct('planned_positions', STOMP_path(:,1:2)');
        P0_final(2*i-1:2*i,:) = STOMP_path(:,1:2)';                                             
    end
    
    builder.updateReferenceTrajectory(P0_final);
    builder.buildAllConstraintsAndBounds();
    builder.buildBenchmarkingExpressions();
    metrics = builder.getBenchmarkingMetrics();
    metrics.training_time = training_time;
    metrics.formulation_time = 0;
    metrics.cost_STOMP = cost_STOMP;
end
