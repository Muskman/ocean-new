% Start profiling
profile on -detail builtin -timer performance

% rng
rng(3,"philox")

% main_simulation.m
clear; clc; close all;

fprintf('Setting up simulation parameters...\n');

% --- Simulation Parameters ---
sim_params.dt = 1;                       % Simulation time step (s)
sim_params.T_final = 200;                % Total simulation time (s)
sim_params.time_steps = floor(sim_params.T_final / sim_params.dt);
sim_params.visualization = true;         % Enable/disable visualization
sim_params.vis_interval = 20;             % Update visualization every N steps (adjust as needed)
sim_params.vis_vector_scale = 1;         % Scaling factor for velocity vector visualization
sim_params.formation_enabled = true;    % *** SET TO true TO ENABLE FORMATION ***
sim_params.planning_horizon = sim_params.T_final/sim_params.dt;       % Number of steps planner looks ahead
sim_params.replan_interval = sim_params.T_final/sim_params.dt;        % Replan interval (adjust as needed)
sim_params.gradient_required_by_planner = true; % *** SET true/false AS NEEDED ***

% --- Environment Parameters ---
env_params.x_limits = [-50, 50];         % Environment boundaries (m)
env_params.y_limits = [-50, 50];         % Environment boundaries (m)
env_params.obstacles = struct('center', {}, 'radius', {}); % Obstacles structure
% env_params.obstacles(1) = struct('center', [15; 15], 'radius', 15);
% env_params.obstacles(2) = struct('center', [-20; 15], 'radius', 7);
% env_params.obstacles(3) = struct('center', [25; -20], 'radius', 6);

% --- Ocean Current Parameters ---
current_params.type = 'static';           % 'static' or 'time_varying'
current_params.vortices = struct('center', {}, 'strength', {}, 'core_radius', {});
current_params.vortices(1) = struct('center', [10; 10], 'strength', 25*4, 'core_radius', 20);
current_params.vortices(2) = struct('center', [-15; -15], 'strength', -20*4, 'core_radius', 30);
current_params.num_ensemble_members = 10;
current_params.noise_level = 0.1;         % Standard deviation of noise added to current estimate
current_params.gradient_noise_level = 0; % *** ADDED: Std dev of noise for each gradient component ***

% --- Agent Parameters ---
num_agents = 4;
agent_params.radius = 1.5; 
agent_params.max_speed = 5.0;
agent_params.safety_margin = 0.2;
agent_params.color = lines(num_agents); % Assign distinct colors

% --- Formation Parameters ---
agent_params.formation_inter_agent_distance = 8.0;
agent_params.formation_tolerance = 1e-2;
agent_params.formation_weight = 0.5;
d = agent_params.formation_inter_agent_distance;
if num_agents == 3
    h = d * sqrt(3)/2; R = h * 2/3;
    agent_params.formation_relative_positions = [0, R; -d/2, -h/3; d/2, -h/3]';
elseif num_agents == 4
     agent_params.formation_relative_positions = [-d/2, d/2; d/2, d/2; -d/2, -d/2; d/2, -d/2]';
else
    fprintf('Warning: Predefined formation not set for %d agents.\n', num_agents);
    positions = zeros(2, num_agents);
    for i = 1:num_agents; positions(1, i) = (i - (num_agents+1)/2) * d; end
    agent_params.formation_relative_positions = positions;
end
% Normalize relative positions so the mean is [0;0] if not already centered
agent_params.formation_relative_positions = agent_params.formation_relative_positions - mean(agent_params.formation_relative_positions, 2);

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

% Setup Graphics Handles
ax = []; quiver_h = []; agent_plots = []; path_plots = []; plan_plots = [];
control_vel_quivers = []; current_est_quivers = []; formation_links = []; % Handle for formation links

if sim_params.visualization
    figure('Position', [100, 100, 800, 700]); % Slightly larger figure
    ax = axes;
    hold(ax, 'on');
    axis(ax, [env_params.x_limits, env_params.y_limits]);
    axis(ax, 'equal'); grid(ax, 'on'); box on;
    title(ax, 'Multi-Agent Ocean Navigation (Initializing)', 'FontSize', 12);
    xlabel(ax, 'X (m)', 'FontSize', 10); ylabel(ax, 'Y (m)', 'FontSize', 10);

    % Initialize Ocean Current Grid (finer grid for contour)
    nx_contour = 50; ny_contour = 50;
    x_vec_contour = linspace(env_params.x_limits(1), env_params.x_limits(2), nx_contour);
    y_vec_contour = linspace(env_params.y_limits(1), env_params.y_limits(2), ny_contour);
    [X_grid_contour, Y_grid_contour] = meshgrid(x_vec_contour, y_vec_contour);

    % --- Vectorized Calculation for Contour --- 
    contour_positions = [X_grid_contour(:)'; Y_grid_contour(:)']; % Create 2xN matrix
    [U_grid_contour_flat, V_grid_contour_flat] = calculate_ocean_current_vectorized(contour_positions, 0, current_params);
    Current_Mag_flat = sqrt(U_grid_contour_flat.^2 + V_grid_contour_flat.^2);
    % Reshape back to grid format
    Current_Mag = reshape(Current_Mag_flat, size(X_grid_contour));
    % --- End Vectorized Calculation ---

    % Initialize Contour Plot
    [~, contour_h] = contourf(ax, X_grid_contour, Y_grid_contour, Current_Mag, 10, 'LineStyle', 'none');
    
    colmap = flipud(cmap('b',100,65,5));
    % caxis([0,max(max(mag))]); colormap (colmap); %parul
    
    colormap(ax, colmap); % Choose a colormap
    colorbar(ax); % Add colorbar
    % Adjust caxis slightly - use prctile to avoid extreme outliers affecting scale
    mag_limits = prctile(Current_Mag(:), [1, 99]); % Get 1st and 99th percentile
    caxis(ax, [0, max(mag_limits(2), eps)*1.1]);

    % Initialize Current Vector Field (Quiver) on a potentially coarser grid
    nx_quiver = 20; ny_quiver = 20;
    x_vec_quiver = linspace(env_params.x_limits(1), env_params.x_limits(2), nx_quiver);
    y_vec_quiver = linspace(env_params.y_limits(1), env_params.y_limits(2), ny_quiver);
    [X_grid_quiver, Y_grid_quiver] = meshgrid(x_vec_quiver, y_vec_quiver);

    % --- Vectorized Calculation for Quiver --- 
    quiver_positions = [X_grid_quiver(:)'; Y_grid_quiver(:)']; % Create 2xN matrix
    [U_grid_quiver_flat, V_grid_quiver_flat] = calculate_ocean_current_vectorized(quiver_positions, 0, current_params);
    % Mask values inside obstacles for quiver display
    for k = 1:numel(X_grid_quiver)
        pos = quiver_positions(:, k);
        for obs_idx = 1:length(env_params.obstacles)
            if norm(pos - env_params.obstacles(obs_idx).center) < env_params.obstacles(obs_idx).radius
                U_grid_quiver_flat(k) = NaN; V_grid_quiver_flat(k) = NaN; break;
            end
        end
    end
    % Reshape back to grid format
    U_grid_quiver = reshape(U_grid_quiver_flat, size(X_grid_quiver));
    V_grid_quiver = reshape(V_grid_quiver_flat, size(X_grid_quiver));
    % --- End Vectorized Calculation ---

    quiver_h = quiver(ax, X_grid_quiver, Y_grid_quiver, U_grid_quiver, V_grid_quiver, 'AutoScaleFactor', 1.5, 'Color', 'k', 'LineWidth', 0.5);

    % Plot Static Environment (Obstacles)
    plot_environment(ax, env_params, agents);
    obstacle_handles = findobj(ax, 'Type', 'patch'); % Find obstacle patches just plotted

    % Initialize Agent Graphics Handles (using fill)
    agent_fill_plots = gobjects(num_agents, 1); path_plots = gobjects(num_agents, 1); plan_plots = gobjects(num_agents, 1);
    control_vel_quivers = gobjects(num_agents, 1); current_est_quivers = gobjects(num_agents, 1);

    for i = 1:num_agents
        [x_verts, y_verts] = create_circle_vertices(agents(i).position, agent_params.radius);
        agent_fill_plots(i) = fill(ax, x_verts, y_verts, agent_params.color(i,:), 'EdgeColor', 'k', 'LineWidth', 1); % Use assigned color
        path_plots(i) = plot(ax, agents(i).position(1), agents(i).position(2), '-', 'Color', [agent_params.color(i,:)*0.8, 0.7], 'LineWidth', 1.5); % Path slightly darker/transparent
        plan_plots(i) = plot(ax, nan, nan, 'k:', 'LineWidth', 0.5);
        control_vel_quivers(i) = quiver(ax, agents(i).position(1), agents(i).position(2), 0, 0, 'Color', 'c', 'LineWidth', 1.5, 'AutoScale', 'off', 'MaxHeadSize', 0.5);
        current_est_quivers(i) = quiver(ax, agents(i).position(1), agents(i).position(2), 0, 0, 'Color', 'm', 'LineWidth', 1.5, 'AutoScale', 'off', 'MaxHeadSize', 0.5);
    end

    % Initialize Formation Links Plot
    if sim_params.formation_enabled && num_agents > 1
        num_links = nchoosek(num_agents, 2); formation_links = gobjects(num_links, 1); link_idx = 1;
        for i = 1:num_agents; for j = i+1:num_agents
            formation_links(link_idx) = plot(ax, [agents(i).position(1), agents(j).position(1)], [agents(i).position(2), agents(j).position(2)], '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 0.5); link_idx = link_idx + 1;
        end; end
    end

    % Set Stacking Order (Bottom to Top)
    
    uistack(obstacle_handles, 'bottom'); % Obstacles above quiver
    uistack(quiver_h, 'bottom');       % Quiver above contour
    uistack(contour_h, 'bottom');
    % Paths/Plans/Links will plot above obstacles by default here
    for i = 1:num_agents               % Agents and vectors on very top
        uistack(agent_fill_plots(i), 'top');
        uistack(control_vel_quivers(i), 'top');
        uistack(current_est_quivers(i), 'top');
    end
    drawnow;
end

fprintf('Starting simulation loop...\n');
first_viz_step = true;

% --- Simulation Loop ---
for t_idx = 1:sim_params.time_steps
    current_time = t_idx * sim_params.dt;

    % --- Get Noisy Estimates for ALL Agents at Once ---
    % 1a. Collect current positions of all agents
    current_agent_positions = cat(2, agents.position); % 2xN matrix

    % 1b. Call the vectorized noisy estimation function
    [estimated_currents_cell, estimated_gradients_cell] = ...
        get_noisy_current_estimate(current_agent_positions, current_time, current_params, env_params, sim_params.gradient_required_by_planner); % Pass flag
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

        planned_trajectories = sca_multi_agent_planner(agents, env_params, current_params, sim_params, agent_params);
        
        % Stop and view results
        profile off
        profile viewer
        profview  % Alternative viewer
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
        agents(i) = update_agent_state(agents(i), current_time, sim_params.dt, current_params, agent_params);
        state_history{i}(:, t_idx + 1) = agents(i).position;
    end

    % 5. Collision Check (TODO)
    % 6. Goal Check (TODO)

    % 7. Visualization
    if sim_params.visualization && mod(t_idx, sim_params.vis_interval) == 0
        if ~ishandle(ax); fprintf('Vis Warning: Axes closed.\n'); sim_params.visualization = false;
        else
            % --- Update Ocean Current Contour and Quiver ---
            if strcmp(current_params.type, 'time_varying') || first_viz_step
                % --- Vectorized Update --- 
                [U_grid_contour_flat, V_grid_contour_flat] = calculate_ocean_current_vectorized(contour_positions, current_time, current_params);
                Current_Mag_flat = sqrt(U_grid_contour_flat.^2 + V_grid_contour_flat.^2);
                Current_Mag = reshape(Current_Mag_flat, size(X_grid_contour));

                [U_grid_quiver_flat, V_grid_quiver_flat] = calculate_ocean_current_vectorized(quiver_positions, current_time, current_params);
                 % Mask quiver values inside obstacles
                 for k = 1:numel(X_grid_quiver)
                     pos = quiver_positions(:, k);
                     for obs_idx = 1:length(env_params.obstacles)
                         if norm(pos - env_params.obstacles(obs_idx).center) < env_params.obstacles(obs_idx).radius
                             U_grid_quiver_flat(k) = NaN; V_grid_quiver_flat(k) = NaN; break;
                         end
                     end
                 end
                 U_grid_quiver = reshape(U_grid_quiver_flat, size(X_grid_quiver));
                 V_grid_quiver = reshape(V_grid_quiver_flat, size(X_grid_quiver));
                 % --- End Vectorized Update ---

                if ishandle(contour_h); set(contour_h, 'ZData', Current_Mag); end
                % Update color axis dynamically based on current view
                current_mags_valid = Current_Mag(~isnan(Current_Mag));
                if ~isempty(current_mags_valid)
                   mag_limits = prctile(current_mags_valid, [1, 99]);
                   caxis(ax, [0, max(mag_limits(2), eps)*1.1]);
                end

                 if ishandle(quiver_h); set(quiver_h, 'UData', U_grid_quiver, 'VData', V_grid_quiver); end
                 first_viz_step = false;
            end

            % --- Update Agents and Agent-Specific Plots ---
            agent_positions = cat(2, agents.position); % Get all current positions [2xN]
            for i = 1:num_agents
                pos_i = agent_positions(:, i);
                % Agent Fill
                if ishandle(agent_fill_plots(i))
                    [x_verts, y_verts] = create_circle_vertices(pos_i, agent_params.radius);
                    set(agent_fill_plots(i), 'XData', x_verts, 'YData', y_verts);
                end
                % Path
                if ishandle(path_plots(i)); set(path_plots(i), 'XData', state_history{i}(1, 1:t_idx+1), 'YData', state_history{i}(2, 1:t_idx+1)); end
                % Plan
                plan = agents(i).current_plan; start_step = agents(i).plan_start_step; time_into_plan = t_idx - start_step; N_horizon = sim_params.planning_horizon;
                if ishandle(plan_plots(i))
                     if ~isempty(plan) && time_into_plan < N_horizon
                         future_plan_col_indices = (time_into_plan + 2) : min(N_horizon + 1, size(plan,2)); % Ensure index doesn't exceed plan length
                         if ~isempty(future_plan_col_indices) && all(future_plan_col_indices <= size(plan, 2))
                            plan_points_to_plot = [pos_i, plan(:, future_plan_col_indices)];
                            set(plan_plots(i), 'XData', plan_points_to_plot(1,:), 'YData', plan_points_to_plot(2,:));
                         else; set(plan_plots(i), 'XData', nan, 'YData', nan); end
                     else; set(plan_plots(i), 'XData', nan, 'YData', nan); end
                end
                % Control Velocity Quiver
                if ishandle(control_vel_quivers(i)); ctrl_vel = agents(i).control_velocity; set(control_vel_quivers(i), 'XData', pos_i(1), 'YData', pos_i(2), 'UData', ctrl_vel(1)*sim_params.vis_vector_scale, 'VData', ctrl_vel(2)*sim_params.vis_vector_scale); end
                % Estimated Current Quiver
                if ishandle(current_est_quivers(i)); est_curr = agents(i).estimated_current; set(current_est_quivers(i), 'XData', pos_i(1), 'YData', pos_i(2), 'UData', est_curr(1)*sim_params.vis_vector_scale, 'VData', est_curr(2)*sim_params.vis_vector_scale); end

                 % Ensure agent elements stay on top
                 if ishandle(agent_fill_plots(i)); uistack(agent_fill_plots(i), 'top'); end
                 if ishandle(control_vel_quivers(i)); uistack(control_vel_quivers(i), 'top'); end
                 if ishandle(current_est_quivers(i)); uistack(current_est_quivers(i), 'top'); end
            end

            % Update Formation Links (if enabled)
            if sim_params.formation_enabled && num_agents > 1 && ~isempty(formation_links) && all(ishandle(formation_links))
                 link_idx = 1;
                 for i = 1:num_agents; for j = i+1:num_agents
                    set(formation_links(link_idx), 'XData', [agent_positions(1, i), agent_positions(1, j)], 'YData', [agent_positions(2, i), agent_positions(2, j)]); link_idx = link_idx + 1;
                 end; end
            end

            title(ax, sprintf('Multi-Agent Ocean Navigation (t = %.1f s)', current_time));
            drawnow limitrate;
        end
    end

    % Progress Indicator
    if mod(t_idx, 100) == 0; fprintf('Simulated %.1f seconds...\n', current_time); end
end % End simulation loop

fprintf('Simulation finished after %.1f seconds.\n', sim_params.T_final);

% --- Final Visualization ---
if sim_params.visualization && ishandle(ax)
    for i = 1:num_agents
        % Update final agent positions using fill
        if ishandle(agent_fill_plots(i))
            [x_verts, y_verts] = create_circle_vertices(agents(i).position, agent_params.radius);
            set(agent_fill_plots(i), 'XData', x_verts, 'YData', y_verts);
        end
        % Ensure full path is shown
        if ishandle(path_plots(i)); set(path_plots(i), 'XData', state_history{i}(1, :), 'YData', state_history{i}(2, :)); end
        % Plot goal marker
        plot(ax, agents(i).goal(1), agents(i).goal(2), 'x', 'Color', [0 0.7 0], 'MarkerSize', 10, 'LineWidth', 2); % Re-plot goal marker for visibility
        % Hide dynamic elements
        if ishandle(plan_plots(i)); set(plan_plots(i), 'Visible', 'off'); end
        if ishandle(control_vel_quivers(i)); set(control_vel_quivers(i), 'Visible', 'off'); end
        if ishandle(current_est_quivers(i)); set(current_est_quivers(i), 'Visible', 'off'); end
    end
    % Hide formation links
    if sim_params.formation_enabled && ~isempty(formation_links) && all(ishandle(formation_links))
        set(formation_links, 'Visible', 'off');
    end
    title(ax, sprintf('Multi-Agent Ocean Navigation (Finished, T = %.1f s)', sim_params.T_final), 'FontSize', 12);
    hold(ax, 'off');
end