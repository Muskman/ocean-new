classdef ProblemBuilder < handle
    properties (Access = private)
        % Symbolic variables
        P_sym           % Decision variables (2*N_agents x T+1)
        
        % Problem parameters
        N_agents, T, dt
        agents, env_params, current_params, sim_params, agent_params
        
        % Ocean functions
        ocean_current_func, ocean_gradient_func %, ocean_linear_approx_func
        
        % Reference trajectory
        P0              % Reference for linearization
        
        % Accumulated problem components
        objective       % Objective function expression
        constraints     % Cell array of constraint expressions
        lbg, ubg       % Constraint bounds
        
        % Configuration
        config          % What to include/exclude
        
        % Safety margin
        safety_margin
    end
    
    properties (Access = public)
        % Bounds on decision variables
        lbx, ubx
    end
    
    methods
        function obj = ProblemBuilder(agents, env_params, current_params, sim_params, agent_params, config)
            % Constructor - Initialize all parameters and setup
            import casadi.*
            
            % Store parameters
            obj.agents = agents;
            obj.env_params = env_params;
            obj.current_params = current_params;
            obj.sim_params = sim_params;
            obj.agent_params = agent_params;
            
            % Set configuration
            if nargin < 6
                obj.config = obj.getDefaultConfig();
            else
                obj.config = config;
            end
            
            % Problem dimensions
            obj.N_agents = length(agents);
            obj.T = sim_params.planning_horizon;
            obj.dt = sim_params.dt;
            
            % Safety margin (from original code)
            obj.safety_margin = 0.2;
            
            % Initialize containers
            obj.constraints = {};
            obj.lbg = [];
            obj.ubg = [];
            
            % Setup problem components
            obj.setupSymbolicVariables();
            obj.setupOceanFunctions();
            obj.generateReferenceTrajectory();
        end
        
        function setupSymbolicVariables(obj)
            % Create symbolic decision variables and bounds
            import casadi.*
            
            % P_sym: 2*N_agents x (T+1) matrix
            obj.P_sym = MX.sym('P', 2*obj.N_agents, obj.T+1);
            
            % Setup bounds on decision variables
            w_size = numel(obj.P_sym);
            obj.lbx = -inf(w_size, 1);
            obj.ubx = inf(w_size, 1);
            
            % Apply environment bounds
            for k = 0:obj.T
                for i = 1:obj.N_agents
                    idx_x = (k * 2 * obj.N_agents) + 2*i - 1;
                    idx_y = (k * 2 * obj.N_agents) + 2*i;
                    obj.lbx(idx_x) = obj.env_params.x_limits(1) + obj.agent_params.radius;
                    obj.ubx(idx_x) = obj.env_params.x_limits(2) - obj.agent_params.radius;
                    obj.lbx(idx_y) = obj.env_params.y_limits(1) + obj.agent_params.radius;
                    obj.ubx(idx_y) = obj.env_params.y_limits(2) - obj.agent_params.radius;
                end
            end
        end
        
        function setupOceanFunctions(obj)
            % Create ocean current functions
            % [obj.ocean_current_func, obj.ocean_gradient_func] = ...
            %     create_symbolic_ocean_func(obj.current_params, true);
            obj.ocean_current_func = create_symbolic_ocean_func(obj.current_params, true);
            obj.ocean_gradient_func = create_symbolic_ocean_gradient_func(obj.current_params, true);
        end
        
        function generateReferenceTrajectory(obj)
            % Generate reference trajectory P0 using linear interpolation
            obj.P0 = zeros(2*obj.N_agents, obj.T+1);
            for i = 1:obj.N_agents
                start_pos = obj.agents(i).position;
                goal_pos = obj.agents(i).goal;
                % Linear interpolation
                interp_x = linspace(start_pos(1), goal_pos(1), obj.T+1);
                interp_y = linspace(start_pos(2), goal_pos(2), obj.T+1);
                obj.P0(2*i-1, :) = interp_x;
                obj.P0(2*i, :) = interp_y;
            end
        end
        
        function buildObjective(obj)
            % Build objective function (minimize squared control effort using linear approximation)
            import casadi.*
            obj.objective = 0;
            
            for k = 0:obj.T-1
                P_k = obj.P_sym(:, k+1);        % State vector at start of interval k
                P_k_plus_1 = obj.P_sym(:, k+2); % State vector at end of interval k
                P0_k = obj.P0(:, k+1);          % Reference position for linearization

                % Reshape state vector to 2xN_agents matrix for current function
                Pos_k_matrix = reshape(P_k, 2, obj.N_agents);
                P0_k_matrix = reshape(P0_k, 2, obj.N_agents);

                % Calculate currents at time k*dt for all agents
                if obj.config.use_linear_approximation
                    Currents_k_matrix = MX.zeros(2, obj.N_agents);
                    Current_at_P0 = obj.ocean_current_func(P0_k_matrix, k*obj.dt);
                    jacobians = obj.ocean_gradient_func(P0_k_matrix, k*obj.dt);
                    pose_diff = Pos_k_matrix - P0_k_matrix;
                    
                    for i = 1:obj.N_agents
                        Currents_k_matrix(:, i) = Current_at_P0(:, i) + jacobians(:, 2*(i-1)+1:2*i) * pose_diff(:, i);
                    end
                else
                    Currents_k_matrix = obj.ocean_current_func(Pos_k_matrix, k*obj.dt);
                end
                
                Currents_k_vec = reshape(Currents_k_matrix, 2*obj.N_agents, 1);

                % Required ground displacement during interval k
                disp_ground = P_k_plus_1 - P_k;

                % Implied control displacement (proportional to control velocity * dt)
                disp_control = disp_ground - Currents_k_vec * obj.dt;

                % Accumulate squared L2 norm of control displacement
                obj.objective = obj.objective + sumsqr(disp_control);
            end
        end
        
        function addInitialConstraints(obj)
            % Add initial position constraints
            if obj.config.enable_initial_constraints
                initial_pos_vec = reshape(cat(2, obj.agents.position), 2*obj.N_agents, 1);
                obj.constraints{end+1} = obj.P_sym(:, 1) - initial_pos_vec;
                obj.lbg = [obj.lbg; zeros(2*obj.N_agents, 1)-eps];
                obj.ubg = [obj.ubg; zeros(2*obj.N_agents, 1)+eps];
            end
        end
        
        function addFinalConstraints(obj)
            % Add final position constraints
            if obj.config.enable_final_constraints
                final_pos_vec = reshape(cat(2, obj.agents.goal), 2*obj.N_agents, 1);
                obj.constraints{end+1} = obj.P_sym(:, obj.T+1) - final_pos_vec;
                obj.lbg = [obj.lbg; zeros(2*obj.N_agents, 1)-eps];
                obj.ubg = [obj.ubg; zeros(2*obj.N_agents, 1)+eps];
            end
        end
        
        function addControlConstraints(obj)
            % Add maximum displacement/control constraints
            import casadi.*
            if obj.config.enable_control_constraints
                max_control_disp_sq = (obj.agent_params.max_speed * obj.dt)^2;
                
                for k = 0:obj.T-1
                    P_k = obj.P_sym(:, k+1);
                    P_k_plus_1 = obj.P_sym(:, k+2);
                    P0_k = obj.P0(:, k+1);
                    
                    Pos_k_matrix = reshape(P_k, 2, obj.N_agents);
                    P0_k_matrix = reshape(P0_k, 2, obj.N_agents);
                    
                    % Calculate currents at time k*dt for all agents
                    if obj.config.use_linear_approximation
                        Currents_k_matrix = MX.zeros(2, obj.N_agents);
                        Current_at_P0 = obj.ocean_current_func(P0_k_matrix, k*obj.dt);
                        jacobians = obj.ocean_gradient_func(P0_k_matrix, k*obj.dt);
                        pose_diff = Pos_k_matrix - P0_k_matrix;
                        
                        for i = 1:obj.N_agents
                            Currents_k_matrix(:, i) = Current_at_P0(:, i) + jacobians(:, 2*(i-1)+1:2*i) * pose_diff(:, i);
                        end
                    else
                        Currents_k_matrix = obj.ocean_current_func(Pos_k_matrix, k*obj.dt);
                    end
                    
                    Currents_k_vec = reshape(Currents_k_matrix, 2*obj.N_agents, 1);
                    
                    disp_ground = P_k_plus_1 - P_k;
                    disp_control = disp_ground - Currents_k_vec * obj.dt;
                    disp_control_matrix = reshape(disp_control, 2, obj.N_agents);

                    for i = 1:obj.N_agents
                        disp_control_norm_sq_per_agent = sumsqr(disp_control_matrix(:, i));
                        obj.constraints{end+1} = disp_control_norm_sq_per_agent;
                        obj.lbg = [obj.lbg; 0];
                        obj.ubg = [obj.ubg; max_control_disp_sq];
                    end
                end
            end
        end
        
        function addObstacleConstraints(obj)
            % Add obstacle avoidance constraints
            if obj.config.enable_obstacle_constraints && ~isempty(obj.env_params.obstacles)
                num_obstacles = length(obj.env_params.obstacles);
                obs_centers_matrix = cat(2, obj.env_params.obstacles.center); % 2xNumObs
                obs_radii = cat(2, obj.env_params.obstacles.radius);         % 1xNumObs

                for k = 1:obj.T % Loop time steps k=1...T (state index k+1)
                    P_k_plus_1 = obj.P_sym(:, k+1);
                    Pos_k_plus_1_matrix = reshape(P_k_plus_1, 2, obj.N_agents);
                    P0_k_matrix = reshape(obj.P0(:, k+1), 2, obj.N_agents);
                    
                    for i = 1:obj.N_agents
                        Agent_pos_ik = Pos_k_plus_1_matrix(:, i);
                        Agent_pos_ik_0 = P0_k_matrix(:, i);
                        % Vectorized distance check for all obstacles
                        if obj.config.use_linear_approximation
                            dist_sq_all_obs = sumsqr(Agent_pos_ik_0 - obs_centers_matrix) + 2*(Agent_pos_ik_0 - obs_centers_matrix)'*(Agent_pos_ik - Agent_pos_ik_0);
                        else
                            dist_sq_all_obs = sumsqr(Agent_pos_ik - obs_centers_matrix);
                        end
                        
                        min_dist_sq_all_obs = (obj.agent_params.radius + obs_radii + obj.safety_margin).^2;

                        obj.constraints{end+1} = dist_sq_all_obs;
                        obj.lbg = [obj.lbg; min_dist_sq_all_obs'];
                        obj.ubg = [obj.ubg; inf*ones(num_obstacles,1)];
                    end
                end
            end
        end
        
        function addCollisionConstraints(obj)
            % Add inter-agent collision avoidance constraints
            if obj.config.enable_collision_constraints && obj.N_agents > 1
                switch obj.config.collision_method
                    case 'minimum_distance'
                        obj.addMinDistanceCollisionConstraints();
                    case 'pairwise'
                        obj.addPairwiseCollisionConstraints();
                    otherwise
                        obj.addMinDistanceCollisionConstraints(); % Default
                end
            end
        end
        
        function addMinDistanceCollisionConstraints(obj)
            % Add minimum distance collision constraints (current implementation)
            min_dist_agent_sq = (2 * obj.agent_params.radius + obj.safety_margin)^2;
            
            for k = 1:obj.T
                Pos_k = reshape(obj.P_sym(:, k+1), 2, obj.N_agents);
                for i = 1:obj.N_agents-1
                    % Distances to all other agents
                    other_indices = setdiff(i:obj.N_agents, i);
                    distances_sq = sum((Pos_k(:,i) - Pos_k(:,other_indices)).^2, 1);
                    min_dist_sq = min(distances_sq);
                    
                    obj.constraints{end+1} = min_dist_sq;
                    obj.lbg = [obj.lbg; min_dist_agent_sq];
                    obj.ubg = [obj.ubg; inf];
                end
            end
        end
        
        function addPairwiseCollisionConstraints(obj)
            % Add pairwise collision constraints (commented out version)
            min_dist_agent_sq = (2 * obj.agent_params.radius + obj.safety_margin)^2;
            [i_idx, j_idx] = find(triu(ones(obj.N_agents), 1));
            n_pairs = length(i_idx);
            
            for k = 1:obj.T
                P_k_plus_1 = obj.P_sym(:, k+1);
                Pos_matrix = reshape(P_k_plus_1, 2, obj.N_agents);
                
                % Vectorized distance computation
                pos_i = Pos_matrix(:, i_idx);  % 2 x n_pairs
                pos_j = Pos_matrix(:, j_idx);  % 2 x n_pairs
                dist_sq_vec = sum((pos_i - pos_j).^2, 1);  % 1 x n_pairs
                
                % Add all constraints for this time step at once
                obj.constraints{end+1} = dist_sq_vec';
                obj.lbg = [obj.lbg; min_dist_agent_sq * ones(n_pairs, 1)];
                obj.ubg = [obj.ubg; inf * ones(n_pairs, 1)];
            end
        end
        
        function addFormationConstraints(obj)
            % Add formation constraints
            if obj.config.enable_formation_constraints && obj.sim_params.formation_enabled && obj.N_agents > 1
                formation_tolerance_sq_sum = (obj.agent_params.formation_tolerance^2) * obj.N_agents;
                Target_relative = obj.agent_params.formation_relative_positions; % 2xN_agents
                
                for k = 1:obj.T
                    P_k_plus_1 = obj.P_sym(:, k+1);
                    Pos_k_plus_1_matrix = reshape(P_k_plus_1, 2, obj.N_agents);
                    
                    % Calculate the centroid of all agents at this step using CasADi functions
                    Centroid_k = sum2(Pos_k_plus_1_matrix) / obj.N_agents;
                    Relative_k = Pos_k_plus_1_matrix - Centroid_k; % Current relative positions
                    dev_sq = reshape(Relative_k - Target_relative,2*obj.N_agents,1); 
                    % dev_sq = sumsqr(Relative_k - Target_relative); % Squared deviation summed over agents

                    obj.constraints{end+1} = dev_sq;
                    obj.lbg = [obj.lbg; 0*ones(2*obj.N_agents,1)];
                    obj.ubg = [obj.ubg; formation_tolerance_sq_sum*ones(2*obj.N_agents,1)];
                end
            end
        end
        
        function buildAllConstraints(obj)
            % Master method that calls all constraint builders
            obj.addInitialConstraints();
            obj.addFinalConstraints();
            obj.addControlConstraints(); 
            obj.addObstacleConstraints();
            obj.addCollisionConstraints();
            obj.addFormationConstraints();
        end
        
        function J = getObjective(obj)
            % Get objective function (build if not already built)
            if isempty(obj.objective)
                obj.buildObjective();
            end
            J = obj.objective;
        end
        
        function nlp = getNLP(obj)
            % Assemble complete NLP structure
            import casadi.*
            
            if isempty(obj.objective)
                obj.buildObjective();
            end
            if isempty(obj.constraints)
                obj.buildAllConstraints();
            end
            
            w = obj.P_sym(:);  % Decision variables vector
            g = vertcat(obj.constraints{:});  % All constraints
            
            nlp = struct('f', obj.objective, 'x', w, 'g', g);
        end
        
        function [lbg, ubg] = getConstraintBounds(obj)
            % Get constraint bounds (build constraints if not already built)
            if isempty(obj.constraints)
                obj.buildAllConstraints();
            end
            lbg = obj.lbg;
            ubg = obj.ubg;
        end
        
        function w0 = getInitialGuess(obj)
            % Return flattened P0 as initial guess
            w0 = obj.P0(:);
        end
    end
    
    methods (Static)
        function config = getDefaultConfig()
            % Get default configuration
            config = struct();
            config.enable_initial_constraints = true;
            config.enable_final_constraints = true;
            config.enable_control_constraints = true;
            config.enable_obstacle_constraints = true;
            config.enable_collision_constraints = false;
            config.enable_formation_constraints = true;
            config.collision_method = 'minimum_distance'; % 'minimum_distance' or 'pairwise'
            config.use_linear_approximation = true;
        end
    end
end 