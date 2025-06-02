% initialize_agents.m
function agents = initialize_agents(num_agents, agent_params, env_params, sim_params) % Added sim_params
    agents = struct('id', {}, 'position', {}, 'goal', {}, ...
                    'control_velocity', {}, 'estimated_current', {}, ...
                    'estimated_gradient', {}, ...
                    'radius', {}, 'current_plan', {}, 'plan_start_step', {});

    agent_radius = agent_params.radius;
    min_dist_obs_factor = 1.5; % Multiplier for radius to keep away from obstacle center
    min_dist_agent = 2.5 * agent_radius; % Min distance between agents centers

    if ~sim_params.formation_enabled
        fprintf('Initializing agents with random start/goal positions.\n');
        % --- Standard Random Initialization ---
        positions = zeros(2, num_agents);
        goals = zeros(2, num_agents);
        for i = 1:num_agents
            valid_pos = false;
            while ~valid_pos % Find valid start position
                pos = [env_params.x_limits(1) + rand() * diff(env_params.x_limits);
                       env_params.y_limits(1) + rand() * diff(env_params.y_limits)];
                obs_collision = false;
                if ~isempty(env_params.obstacles) % Check only if obstacles exist
                    obs_collision = any(vecnorm(pos - cat(2, env_params.obstacles.center)) < cat(2, env_params.obstacles.radius) + agent_radius * min_dist_obs_factor);
                end
                agent_collision = any(vecnorm(pos - positions(:, 1:i-1)) < min_dist_agent);
                if ~obs_collision && ~agent_collision; valid_pos = true; positions(:, i) = pos; end
            end
            valid_goal = false;
            while ~valid_goal % Find valid goal position
                goal = [env_params.x_limits(1) + rand() * diff(env_params.x_limits);
                        env_params.y_limits(1) + rand() * diff(env_params.y_limits)];
                obs_collision = false;
                 if ~isempty(env_params.obstacles) % Check only if obstacles exist
                    obs_collision = any(vecnorm(goal - cat(2, env_params.obstacles.center)) < cat(2, env_params.obstacles.radius) + agent_radius); % Goal edge outside obs
                 end
                goal_too_close_to_start = norm(goal - pos) < 0.4 * min(diff(env_params.x_limits), diff(env_params.y_limits));
                goal_too_close_to_other_goals = any(vecnorm(goal - goals(:, 1:i-1)) < min_dist_agent);
                if ~obs_collision && ~goal_too_close_to_start && ~goal_too_close_to_other_goals; valid_goal = true; goals(:, i) = goal; end
            end
        end
         % --- End Random Init ---

    else % --- Formation Initialization ---
        fprintf('Initializing agents in formation.\n');
        relative_pos = agent_params.formation_relative_positions; % [2xN]

        valid_formation_placement = false;
        max_placement_tries = 100;
        placement_tries = 0;

        while ~valid_formation_placement && placement_tries < max_placement_tries
            placement_tries = placement_tries + 1;

            % Find valid START centroid
            valid_start_centroid = false;
            while ~valid_start_centroid
                 start_centroid = [env_params.x_limits(1) + rand() * diff(env_params.x_limits);
                                   env_params.y_limits(1) + rand() * diff(env_params.y_limits)];
                 % Calculate potential absolute start positions
                 potential_start_positions = start_centroid + relative_pos;
                 % Check obstacle collision for ALL agents in formation
                 obs_collision_start = false;
                 if ~isempty(env_params.obstacles)
                     for i = 1:num_agents
                         if any(vecnorm(potential_start_positions(:,i) - cat(2, env_params.obstacles.center)) < cat(2, env_params.obstacles.radius) + agent_radius * min_dist_obs_factor)
                             obs_collision_start = true; break;
                         end
                     end
                 end
                 % Check if any part of formation is out of bounds
                 out_of_bounds_start = any(potential_start_positions(1,:) < env_params.x_limits(1) | potential_start_positions(1,:) > env_params.x_limits(2) | ...
                                           potential_start_positions(2,:) < env_params.y_limits(1) | potential_start_positions(2,:) > env_params.y_limits(2));

                 if ~obs_collision_start && ~out_of_bounds_start
                     valid_start_centroid = true;
                     positions = potential_start_positions;
                 end
            end

            % Find valid GOAL centroid
            valid_goal_centroid = false;
            while ~valid_goal_centroid
                 goal_centroid = [env_params.x_limits(1) + rand() * diff(env_params.x_limits);
                                  env_params.y_limits(1) + rand() * diff(env_params.y_limits)];
                 % Calculate potential absolute goal positions
                 potential_goal_positions = goal_centroid + relative_pos;
                 % Check obstacle collision for ALL agents in formation
                 obs_collision_goal = false;
                 if ~isempty(env_params.obstacles)
                     for i = 1:num_agents
                         if any(vecnorm(potential_goal_positions(:,i) - cat(2, env_params.obstacles.center)) < cat(2, env_params.obstacles.radius) + agent_radius) % Goal edge outside obs
                             obs_collision_goal = true; break;
                         end
                     end
                 end
                 % Check if any part of formation is out of bounds
                 out_of_bounds_goal = any(potential_goal_positions(1,:) < env_params.x_limits(1) | potential_goal_positions(1,:) > env_params.x_limits(2) | ...
                                          potential_goal_positions(2,:) < env_params.y_limits(1) | potential_goal_positions(2,:) > env_params.y_limits(2));


                 % Ensure goal centroid is far enough from start centroid
                 centroids_too_close = norm(goal_centroid - start_centroid) < 0.5 * min(diff(env_params.x_limits), diff(env_params.y_limits));

                 if ~obs_collision_goal && ~out_of_bounds_goal && ~centroids_too_close
                     valid_goal_centroid = true;
                     goals = potential_goal_positions;
                 end
            end
            valid_formation_placement = true; % If we reached here, both are valid
        end % End while trying placement

        if ~valid_formation_placement
            error('Could not find valid start/goal formation placement after %d tries. Check parameters/obstacle density.', max_placement_tries);
        end
         % --- End Formation Init ---

    end

    % Assign properties to agents structure
    for i = 1:num_agents
        agents(i).id = i;
        agents(i).position = positions(:, i);
        agents(i).goal = goals(:, i);
        agents(i).control_velocity = [0; 0];
        agents(i).estimated_current = [0; 0];
        agents(i).estimated_gradient = zeros(2, 2);
        agents(i).radius = agent_radius;
        agents(i).current_plan = []; % Initialize plan fields
        agents(i).plan_start_step = -inf;
    end
end