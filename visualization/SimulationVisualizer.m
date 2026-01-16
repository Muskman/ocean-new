classdef SimulationVisualizer < handle
    % SIMULATIONVISUALIZER Handles all visualization for the ocean multi-agent simulation
    % 
    % This class encapsulates all visualization logic that was previously inline
    % in main_simulation.m. It maintains exact same functionality and behavior.
    
    properties (Access = private)
        % Configuration parameters
        sim_params
        env_params
        current_params
        agent_params
        num_agents
        video_params
        
        % Graphics handles
        figure_handle
        ax                          % Main axes handle
        contour_h                   % Contour plot handle
        quiver_h                    % Quiver plot handle
        agent_fill_plots            % Agent circle fill plots
        path_plots                  % Agent path plots
        plan_plots                  % Agent plan plots
        control_vel_quivers         % Control velocity quiver plots
        current_est_quivers         % Current estimate quiver plots
        formation_links             % Formation link plots
        obstacle_handles            % Obstacle patch handles
        
        % Ocean visualization grid data
        X_grid_contour, Y_grid_contour          % Contour grid
        contour_positions                       % Contour positions for vectorized calc
        X_grid_quiver, Y_grid_quiver           % Quiver grid  
        quiver_positions                        % Quiver positions for vectorized calc
        
        % State tracking
        first_viz_step              % Flag for first visualization step
        
        % Video export properties
        video_writer                % VideoWriter object
        video_enabled               % Flag for video export enabled
        video_file_path             % Full path to video file
    end
    
    methods
        function obj = SimulationVisualizer(sim_params, env_params, current_params, agent_params, num_agents, video_params)
            % Constructor - store configuration parameters
            obj.sim_params = sim_params;
            obj.env_params = env_params;
            obj.current_params = current_params;
            obj.agent_params = agent_params;
            obj.num_agents = num_agents;
            obj.video_params = video_params;
            
            % Initialize state
            obj.first_viz_step = true;
            
            % Initialize video properties
            obj.video_enabled = video_params.enabled;
            obj.video_writer = [];
            obj.video_file_path = '';
            
            % Initialize handle arrays as empty
            obj.ax = [];
            obj.quiver_h = [];
            obj.agent_fill_plots = [];
            obj.path_plots = [];
            obj.plan_plots = [];
            obj.control_vel_quivers = [];
            obj.current_est_quivers = [];
            obj.formation_links = [];
            obj.obstacle_handles = [];
        end
        
        function initialize(obj, agents)
            % Setup initial visualization
            
            if ~obj.sim_params.visualization
                return; % Skip if visualization is disabled
            end
            
            % Create figure and axes optimized for video capture
            obj.figure_handle = figure('Position', [100, 100, 1000, 800], ... % Larger figure for better video quality
                                      'Color', 'white', ...                    % White background
                                      'MenuBar', 'none', ...                   % Clean appearance
                                      'ToolBar', 'none');                      % No toolbar for cleaner video
            obj.ax = axes('Parent', obj.figure_handle);
            hold(obj.ax, 'on');
            axis(obj.ax, [obj.env_params.x_limits, obj.env_params.y_limits]);
            axis(obj.ax, 'equal'); grid(obj.ax, 'on'); box on;
            set(obj.ax, 'FontSize', 12, 'FontWeight', 'bold');
            title(obj.ax, 'Multi-Agent Ocean Navigation (Initializing)', 'FontSize', 14, 'FontWeight', 'bold');
            xlabel(obj.ax, 'X (m)', 'FontSize', 12); ylabel(obj.ax, 'Y (m)', 'FontSize', 12, 'FontWeight', 'bold');

            % Initialize Ocean Current Grid
            nx_contour = 50; ny_contour = 50;
            x_vec_contour = linspace(obj.env_params.x_limits(1), obj.env_params.x_limits(2), nx_contour);
            y_vec_contour = linspace(obj.env_params.y_limits(1), obj.env_params.y_limits(2), ny_contour);
            [obj.X_grid_contour, obj.Y_grid_contour] = meshgrid(x_vec_contour, y_vec_contour);

            % --- Vectorized Calculation for Contour --- 
            obj.contour_positions = [obj.X_grid_contour(:)'; obj.Y_grid_contour(:)']; % Create 2xN matrix
            [U_grid_contour_flat, V_grid_contour_flat] = calculate_ocean_current_vectorized(obj.contour_positions, 0, obj.current_params);
            Current_Mag_flat = sqrt(U_grid_contour_flat.^2 + V_grid_contour_flat.^2);
            % Reshape back to grid format
            Current_Mag = reshape(Current_Mag_flat, size(obj.X_grid_contour));

            % Initialize Contour Plot
            [~, obj.contour_h] = contourf(obj.ax, obj.X_grid_contour, obj.Y_grid_contour, Current_Mag, 10, 'LineStyle', 'none');
            
            colmap = flipud(cmap('b',100,65,5));
            % caxis([0,max(max(mag))]); colormap (colmap); %parul
            
            colormap(obj.ax, colmap); % Choose a colormap
            cb = colorbar(obj.ax); % Add colorbar
            cb.Label.String = 'Ocean Current Speed (m/s)';
            cb.Label.FontSize = 14;
            cb.Label.FontWeight = 'bold';
            % Adjust caxis slightly - use prctile to avoid extreme outliers affecting scale
            mag_limits = prctile(Current_Mag(:), [1, 99]); % Get 1st and 99th percentile
            caxis(obj.ax, [0, max(mag_limits(2), eps)*1.1]);

            % Initialize Current Vector Field (Quiver) on a potentially coarser grid 
            nx_quiver = 20; ny_quiver = 20;
            x_vec_quiver = linspace(obj.env_params.x_limits(1), obj.env_params.x_limits(2), nx_quiver);
            y_vec_quiver = linspace(obj.env_params.y_limits(1), obj.env_params.y_limits(2), ny_quiver);
            [obj.X_grid_quiver, obj.Y_grid_quiver] = meshgrid(x_vec_quiver, y_vec_quiver);

            % --- Vectorized Calculation for Quiver --- 
            obj.quiver_positions = [obj.X_grid_quiver(:)'; obj.Y_grid_quiver(:)']; % Create 2xN matrix
            [U_grid_quiver_flat, V_grid_quiver_flat] = calculate_ocean_current_vectorized(obj.quiver_positions, 0, obj.current_params);
            % Mask values inside obstacles for quiver display
            for k = 1:numel(obj.X_grid_quiver)
                pos = obj.quiver_positions(:, k);
                for obs_idx = 1:length(obj.env_params.obstacles)
                    if norm(pos - obj.env_params.obstacles(obs_idx).center) < obj.env_params.obstacles(obs_idx).radius
                        U_grid_quiver_flat(k) = NaN; V_grid_quiver_flat(k) = NaN; break;
                    end
                end
            end
            % Reshape back to grid format
            U_grid_quiver = reshape(U_grid_quiver_flat, size(obj.X_grid_quiver));
            V_grid_quiver = reshape(V_grid_quiver_flat, size(obj.X_grid_quiver));
            % --- End Vectorized Calculation ---

            % Create quiver plot 
            obj.quiver_h = quiver(obj.ax, obj.X_grid_quiver, obj.Y_grid_quiver, U_grid_quiver, V_grid_quiver, 'AutoScaleFactor', 1.5, 'Color', 'k', 'LineWidth', 0.5);

            % Plot Static Environment (Obstacles) 
            plot_environment(obj.ax, obj.env_params, agents);
            obj.obstacle_handles = findobj(obj.ax, 'Type', 'patch'); % Find obstacle patches just plotted

            % Initialize Agent Graphics Handles (using fill) 
            obj.agent_fill_plots = gobjects(obj.num_agents, 1); 
            obj.path_plots = gobjects(obj.num_agents, 1); 
            obj.plan_plots = gobjects(obj.num_agents, 1);
            obj.control_vel_quivers = gobjects(obj.num_agents, 1); 
            obj.current_est_quivers = gobjects(obj.num_agents, 1);

            for i = 1:obj.num_agents
                [x_verts, y_verts] = create_circle_vertices(agents(i).position, obj.agent_params.radius);
                obj.agent_fill_plots(i) = fill(obj.ax, x_verts, y_verts, obj.agent_params.color(i,:), 'EdgeColor', 'k', 'LineWidth', 1); % Use assigned color
                obj.path_plots(i) = plot(obj.ax, agents(i).position(1), agents(i).position(2), '-', 'Color', [obj.agent_params.color(i,:)*0.8, 0.7], 'LineWidth', 1.5); % Path slightly darker/transparent
                obj.plan_plots(i) = plot(obj.ax, nan, nan, 'k:', 'LineWidth', 0.5);
                obj.control_vel_quivers(i) = quiver(obj.ax, agents(i).position(1), agents(i).position(2), 0, 0, 'Color', 'c', 'LineWidth', 1.5, 'AutoScale', 'off', 'MaxHeadSize', 0.5);
                obj.current_est_quivers(i) = quiver(obj.ax, agents(i).position(1), agents(i).position(2), 0, 0, 'Color', 'm', 'LineWidth', 1.5, 'AutoScale', 'off', 'MaxHeadSize', 0.5);
            end

            % Initialize Formation Links Plot 
            if obj.sim_params.formation_enabled && obj.num_agents > 1
                num_links = nchoosek(obj.num_agents, 2); 
                obj.formation_links = gobjects(num_links, 1); 
                link_idx = 1;
                for i = 1:obj.num_agents
                    for j = i+1:obj.num_agents
                        obj.formation_links(link_idx) = plot(obj.ax, [agents(i).position(1), agents(j).position(1)], [agents(i).position(2), agents(j).position(2)], '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 0.5); 
                        link_idx = link_idx + 1;
                    end
                end
            end

            % Set Stacking Order (Bottom to Top) 
            uistack(obj.obstacle_handles, 'bottom'); % Obstacles above quiver
            uistack(obj.quiver_h, 'bottom');        % Quiver above contour
            uistack(obj.contour_h, 'bottom');
            % Paths/Plans/Links will plot above obstacles by default here
            for i = 1:obj.num_agents                % Agents and vectors on very top
                uistack(obj.agent_fill_plots(i), 'top');
                uistack(obj.control_vel_quivers(i), 'top');
                uistack(obj.current_est_quivers(i), 'top');
            end
            drawnow;
            
            % Setup video capture if enabled
            obj.setupVideoCapture();
        end
        
        function update(obj, agents, current_time, t_idx, state_history)
            % Runtime visualization updates
            
            if ~obj.sim_params.visualization || mod(t_idx, obj.sim_params.vis_interval) ~= 0
                return; % Skip if visualization disabled or not update interval
            end
            
            if ~ishandle(obj.ax)
                fprintf('Vis Warning: Axes closed.\n'); 
                obj.sim_params.visualization = false;
                return;
            end
            
            % --- Update Ocean Current Contour and Quiver --- 
            if strcmp(obj.current_params.type, 'time_varying') || obj.first_viz_step
                % --- Vectorized Update --- 
                [U_grid_contour_flat, V_grid_contour_flat] = calculate_ocean_current_vectorized(obj.contour_positions, current_time, obj.current_params);
                Current_Mag_flat = sqrt(U_grid_contour_flat.^2 + V_grid_contour_flat.^2);
                Current_Mag = reshape(Current_Mag_flat, size(obj.X_grid_contour));

                [U_grid_quiver_flat, V_grid_quiver_flat] = calculate_ocean_current_vectorized(obj.quiver_positions, current_time, obj.current_params);
                 % Mask quiver values inside obstacles
                 for k = 1:numel(obj.X_grid_quiver)
                     pos = obj.quiver_positions(:, k);
                     for obs_idx = 1:length(obj.env_params.obstacles)
                         if norm(pos - obj.env_params.obstacles(obs_idx).center) < obj.env_params.obstacles(obs_idx).radius
                             U_grid_quiver_flat(k) = NaN; V_grid_quiver_flat(k) = NaN; break;
                         end
                     end
                 end
                 U_grid_quiver = reshape(U_grid_quiver_flat, size(obj.X_grid_quiver));
                 V_grid_quiver = reshape(V_grid_quiver_flat, size(obj.X_grid_quiver));
                 % --- End Vectorized Update ---

                if ishandle(obj.contour_h); set(obj.contour_h, 'ZData', Current_Mag); end
                % Update color axis dynamically based on current view
                current_mags_valid = Current_Mag(~isnan(Current_Mag));
                if ~isempty(current_mags_valid)
                   mag_limits = prctile(current_mags_valid, [1, 99]);
                   caxis(obj.ax, [0, max(mag_limits(2), eps)*1.1]);
                end

                 if ishandle(obj.quiver_h); set(obj.quiver_h, 'UData', U_grid_quiver, 'VData', V_grid_quiver); end
                 obj.first_viz_step = false;
            end

            % --- Update Agents and Agent-Specific Plots --- 
            agent_positions = cat(2, agents.position); % Get all current positions [2xN]
            for i = 1:obj.num_agents
                pos_i = agent_positions(:, i);
                % Agent Fill
                if ishandle(obj.agent_fill_plots(i))
                    [x_verts, y_verts] = create_circle_vertices(pos_i, obj.agent_params.radius);
                    set(obj.agent_fill_plots(i), 'XData', x_verts, 'YData', y_verts);
                end
                % Path
                if ishandle(obj.path_plots(i)); set(obj.path_plots(i), 'XData', state_history{i}(1, 1:t_idx+1), 'YData', state_history{i}(2, 1:t_idx+1)); end
                % Plan
                plan = agents(i).current_plan; start_step = agents(i).plan_start_step; time_into_plan = t_idx - start_step; N_horizon = obj.sim_params.planning_horizon;
                if ishandle(obj.plan_plots(i))
                    if ~isempty(plan) && time_into_plan < N_horizon
                        future_plan_col_indices = (time_into_plan + 2) : min(N_horizon + 1, size(plan,2)); % Ensure index doesn't exceed plan length
                        if ~isempty(future_plan_col_indices) && all(future_plan_col_indices <= size(plan, 2))
                           plan_points_to_plot = [pos_i, plan(:, future_plan_col_indices)];
                           set(obj.plan_plots(i), 'XData', plan_points_to_plot(1,:), 'YData', plan_points_to_plot(2,:));
                        else
                           set(obj.plan_plots(i), 'XData', nan, 'YData', nan);
                        end
                    else
                       set(obj.plan_plots(i), 'XData', nan, 'YData', nan);
                    end
                end
                % Control Velocity Quiver
                if ishandle(obj.control_vel_quivers(i)); ctrl_vel = agents(i).control_velocity; set(obj.control_vel_quivers(i), 'XData', pos_i(1), 'YData', pos_i(2), 'UData', ctrl_vel(1)*obj.sim_params.vis_vector_scale, 'VData', ctrl_vel(2)*obj.sim_params.vis_vector_scale); end
                % Estimated Current Quiver
                if ishandle(obj.current_est_quivers(i)); est_curr = agents(i).estimated_current; set(obj.current_est_quivers(i), 'XData', pos_i(1), 'YData', pos_i(2), 'UData', est_curr(1)*obj.sim_params.vis_vector_scale, 'VData', est_curr(2)*obj.sim_params.vis_vector_scale); end

                 % Ensure agent elements stay on top
                 if ishandle(obj.agent_fill_plots(i)); uistack(obj.agent_fill_plots(i), 'top'); end
                 if ishandle(obj.control_vel_quivers(i)); uistack(obj.control_vel_quivers(i), 'top'); end
                 if ishandle(obj.current_est_quivers(i)); uistack(obj.current_est_quivers(i), 'top'); end
            end

            % Update Formation Links (if enabled) 
            if obj.sim_params.formation_enabled && obj.num_agents > 1 && ~isempty(obj.formation_links) && all(ishandle(obj.formation_links))
                 link_idx = 1;
                 for i = 1:obj.num_agents
                     for j = i+1:obj.num_agents
                        set(obj.formation_links(link_idx), 'XData', [agent_positions(1, i), agent_positions(1, j)], 'YData', [agent_positions(2, i), agent_positions(2, j)]); 
                        link_idx = link_idx + 1;
                     end
                 end
            end

            % Update title and refresh display 
            title(obj.ax, sprintf('Multi-Agent Ocean Navigation (t = %3.1f s)', current_time), 'FontSize', 14, 'FontWeight', 'bold');
            drawnow limitrate;
            
            % Capture video frame if enabled (synced with visualization updates)
            obj.captureVideoFrame();
        end
        
        function finalize(obj, agents, state_history)
            % Final visualization cleanup
            
            if ~obj.sim_params.visualization || ~ishandle(obj.ax)
                return; % Skip if visualization disabled or axes closed
            end
            
            for i = 1:obj.num_agents
                % Update final agent positions using fill
                if ishandle(obj.agent_fill_plots(i))
                    [x_verts, y_verts] = create_circle_vertices(agents(i).position, obj.agent_params.radius);
                    set(obj.agent_fill_plots(i), 'XData', x_verts, 'YData', y_verts);
                end
                % Ensure full path is shown
                if ishandle(obj.path_plots(i)); set(obj.path_plots(i), 'XData', state_history{i}(1, :), 'YData', state_history{i}(2, :)); end
                % Plot goal marker
                plot(obj.ax, agents(i).goal(1), agents(i).goal(2), 'x', 'Color', [0 0.7 0], 'MarkerSize', 10, 'LineWidth', 2); % Re-plot goal marker for visibility
                % Hide dynamic elements
                if ishandle(obj.plan_plots(i)); set(obj.plan_plots(i), 'Visible', 'off'); end
                if ishandle(obj.control_vel_quivers(i)); set(obj.control_vel_quivers(i), 'Visible', 'off'); end
                if ishandle(obj.current_est_quivers(i)); set(obj.current_est_quivers(i), 'Visible', 'off'); end
            end
            % Hide formation links
            if obj.sim_params.formation_enabled && ~isempty(obj.formation_links) && all(ishandle(obj.formation_links))
                set(obj.formation_links, 'Visible', 'off');
            end
            title(obj.ax, sprintf('Multi-Agent Ocean Navigation (Finished, T = %3.1f s)', obj.sim_params.T_final), 'FontSize', 14, 'FontWeight', 'bold');
            hold(obj.ax, 'off');
            
            % Capture final frames for a brief pause (professional video ending)
            if obj.video_enabled
                % Hold final frame for 2 seconds (framerate * 2)
                pause_frames = obj.video_params.framerate * 2;
                for i = 1:pause_frames
                    obj.captureVideoFrame();
                end
            end
            
            % Finalize video capture
            obj.finalizeVideoCapture();
        end
        
        function is_enabled = isVisualizationEnabled(obj)
            % Helper method to check if visualization is enabled
            is_enabled = obj.sim_params.visualization;
        end
        
        function is_valid = isVisualizationValid(obj)
            % Helper method to check if visualization is still valid
            is_valid = obj.sim_params.visualization && ishandle(obj.ax);
        end
        
        function setupVideoCapture(obj)
            % Setup video capture with auto-generated path and filename
            if ~obj.video_enabled || ~obj.sim_params.visualization
                return;
            end
            
            try
                % Generate auto-generated folder structure and filename with .avi extension
                obj.video_file_path = [obj.generateVideoFilePath(), '.avi'];
                
                % Create directory if it doesn't exist
                [video_dir, ~, ~] = fileparts(obj.video_file_path);
                if ~exist(video_dir, 'dir')
                    mkdir(video_dir);
                end
                
                % Create VideoWriter object using Motion JPEG AVI (most compatible)
                obj.video_writer = VideoWriter(obj.video_file_path, 'Motion JPEG AVI');
                obj.video_writer.FrameRate = obj.video_params.framerate;
                obj.video_writer.Quality = obj.video_params.quality;
                
                % Open video file
                open(obj.video_writer);
                
                fprintf('Video export enabled: %s\n', obj.video_file_path);
                
            catch ME
                warning('SimulationVisualizer:VideoSetupFailed', 'Failed to setup video capture: %s. Video export disabled.', ME.message);
                obj.video_enabled = false;
                obj.video_writer = [];
            end
        end
        
        function captureVideoFrame(obj)
            % Capture current frame for video export
            if ~obj.video_enabled || isempty(obj.video_writer) || ~ishandle(obj.figure_handle)
                return;
            end
            
            try
                % Ensure figure is properly rendered before capture
                figure(obj.figure_handle); % Bring figure to front
                drawnow; % Force complete rendering
                
                % Capture frame from entire figure (includes title, labels, colorbar, etc.)
                frame = getframe(obj.figure_handle);
                
                % Write frame to video
                writeVideo(obj.video_writer, frame);
                
            catch ME
                warning('SimulationVisualizer:FrameCaptureFailed', 'Failed to capture video frame: %s', ME.message);
            end
        end
        
        function finalizeVideoCapture(obj)
            % Finalize and close video file
            if ~obj.video_enabled || isempty(obj.video_writer)
                return;
            end
            
            try
                % Close video file
                close(obj.video_writer);
                obj.video_writer = [];
                
                fprintf('Video export completed: %s\n', obj.video_file_path);
                
            catch ME
                warning('SimulationVisualizer:VideoFinalizeFailed', 'Failed to finalize video capture: %s', ME.message);
            end
        end
        
        function file_path = generateVideoFilePath(obj)
            % Generate auto-generated folder structure and filename
            
            % Level 1: Date folder (YYYY-MM-DD)
            date_folder = datestr(now, 'yyyy-mm-dd');
            
            % Level 2: Configuration folder (formation_enabled/disabled + current type)
            if obj.sim_params.formation_enabled
                formation_str = 'formation_enabled';
            else
                formation_str = 'formation_disabled';
            end
            config_folder = sprintf('%s_%s', formation_str, obj.current_params.type);
            
            % Level 3: Auto-generated filename with algorithm prefix
            % Format numbers compactly (no trailing zeros)
            dt_str = obj.formatNumber(obj.sim_params.dt);
            T_final_str = obj.formatNumber(obj.sim_params.T_final);
            noise_str = obj.formatNumber(obj.current_params.noise_level);
            
            filename = sprintf('%s_dt%s_T%s_ens%d_noise%s_agents%d', ...
                obj.sim_params.algo, ...
                dt_str, ...
                T_final_str, ...
                obj.current_params.num_ensemble_members, ...
                noise_str, ...
                obj.num_agents);
            
            % Construct full path (without extension - will be added based on video profile)
            file_path = fullfile('results', date_folder, config_folder, filename);
        end
        
        function str = formatNumber(~, num)
            % Format number compactly (remove trailing zeros)
            % Using %g automatically handles both integers and decimals compactly
            str = sprintf('%g', num);
        end
        
    end
end
