classdef ProblemBuilder < handle
    properties (Access = private)
        % Symbolic variables
        P_sym                               % Decision variables (2*N_agents x T+1)
        
        % Problem parameters
        N_agents, T, dt
        agents, env_params, current_params, sim_params, agent_params
        
        % Ocean functions
        ocean_current_func, ocean_gradient_func
        
        % Objective function expression
        objective       % Objective function expression
        
        % Symbolic template system
        P0_sym                              % Symbolic reference trajectory (2*N_agents x T+1)
        ensemble_samples_sym                % Symbolic ensemble sample indicator (num_ensemble_members x 1)
        symbolic_objective_template         % Symbolic objective f(P_sym, P0_sym, ensemble_samples_sym)
        symbolic_control_constraints_template   % Cell array of symbolic constraints g(P_sym, P0_sym)

        constraint_exprs

        % Template metadata
        templates_built                     % Boolean flag
        
        % Cached constraint bounds (built together with constraints)
        constraint_bounds_lbg               % Lower bounds for all constraints
        constraint_bounds_ubg               % Upper bounds for all constraints
        
        % Configuration
        config                              % What to include/exclude
        
        % Safety margin
        safety_margin
    end
    
    properties (Access = public)
        % Bounds on decision variables
        lbx, ubx

        energy_training
        constraints_training
        control_constraints_training
        lbg_training
        ubg_training

        energy_testing
        constraints_testing
        control_constraints_testing
        lbg_testing
        ubg_testing
        
        % Reference trajectory (needed for parameterized NLP)
        P0              % Reference for linearization

        % Ensemble samples
        ensemble_samples
        ensemble_sample_idx
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
            obj.ensemble_samples_sym = [];
            obj.ensemble_sample_idx = 0;
            
            % Safety margin (from original code)
            obj.safety_margin = 0.2;
            
            % Initialize symbolic template system
            obj.P0_sym = [];
            obj.symbolic_objective_template = [];
            obj.symbolic_control_constraints_template = {};
            obj.templates_built = false;
            
            % Initialize cached constraint bounds
            obj.constraint_bounds_lbg = [];
            obj.constraint_bounds_ubg = [];
            
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
            [obj.ocean_current_func, obj.ocean_gradient_func] = create_symbolic_ocean_func(obj.current_params, true);
            % obj.ocean_current_func = create_symbolic_ocean_func(obj.current_params, true);
            % obj.ocean_gradient_func = create_symbolic_ocean_gradient_func(obj.current_params, true);
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

        function updateReferenceTrajectory(obj, P)
            % Update reference trajectory
            obj.P0 = P;
            
            if obj.config.use_linear_approximation && obj.templates_built
                fprintf('Reference trajectory updated for linear approximation.\n');
            else
                fprintf('Reference trajectory updated.\n');
            end
        end
        
        function buildSymbolicTemplates(obj)
            % Build symbolic templates for objective and constraints
            % These templates are parameterized by P0_sym and can be materialized quickly
            import casadi.*
            
            fprintf('Building symbolic templates for objective and constraints...\n');
            
            % Create symbolic reference trajectory
            obj.P0_sym = MX.sym('P0', 2*obj.N_agents, obj.T+1);
            obj.ensemble_samples_sym = MX.sym('ensemble_samples', Sparsity.dense(obj.current_params.num_ensemble_members,1));

            % Build symbolic objective and control constraints template
            symbolic_objective_template = MX.zeros(obj.current_params.num_ensemble_members, 1);
            if obj.config.use_linear_approximation
                obj.symbolic_control_constraints_template = cell(1, obj.N_agents*obj.T);
            else
                obj.symbolic_control_constraints_template = cell(1, obj.N_agents*obj.T*obj.current_params.num_ensemble_members);
            end
    
            for k = 0:obj.T-1
                Currents_at_P0_sym = cell(1, obj.current_params.num_ensemble_members+1);
                Js_at_P0_sym = cell(1, obj.current_params.num_ensemble_members+1);
                Currents_k_matrix = cell(1, obj.current_params.num_ensemble_members+1);
                Currents_k_matrix_avg = MX.zeros(2, obj.N_agents);
                
                % Extract symbolic states and references
                P_k = obj.P_sym(:, k+1);        % Current state
                P_k_plus_1 = obj.P_sym(:, k+2); % Next state
                P0_k = obj.P0_sym(:, k+1);      % Symbolic reference
                disp_ground = P_k_plus_1 - P_k;
                
                % Reshape to matrix form for ocean functions
                Pos_k_matrix = reshape(P_k, 2, obj.N_agents);
                P0_k_matrix = reshape(P0_k, 2, obj.N_agents);
                pose_diff = Pos_k_matrix - P0_k_matrix;
                
                if obj.config.use_linear_approximation
                    % Symbolic ocean evaluations at reference point
                    [Currents_at_P0_sym{:}] = obj.ocean_current_func(P0_k_matrix, k*obj.dt);
                    [Js_at_P0_sym{:}] = obj.ocean_gradient_func(P0_k_matrix, k*obj.dt);
                else
                    % Exact Symbolic ocean evaluations
                    [Currents_k_matrix{:}] = obj.ocean_current_func(Pos_k_matrix, k*obj.dt);
                end

                for i = 1:obj.current_params.num_ensemble_members
                    
                    % Get current ensemble sample indicator
                    % s = obj.ensemble_samples_sym(i)/sum(obj.ensemble_samples_sym);
                    % s = 1/sum(obj.ensemble_samples_sym);

                    % Calculate currents using linear approximation template
                    if obj.config.use_linear_approximation
                        % Template Per-agent linear approximation
                        Currents_k_matrix{i} = MX.zeros(2, obj.N_agents);
                        for j = 1:obj.N_agents
                            Currents_k_matrix{i}(:, j) = Currents_at_P0_sym{i}(:, j) + Js_at_P0_sym{i}(:, 2*(j-1)+1:2*j) * pose_diff(:, j);
                        end
                    end
                    
                    % Build template objective contribution
                    Currents_k_vec = reshape(Currents_k_matrix{i}, 2*obj.N_agents, 1);
                    disp_control = disp_ground - Currents_k_vec * obj.dt;
                    disp_control_matrix = reshape(disp_control, 2, obj.N_agents);
                    
                    % Accumulate objective in symbolic template
                    % Skip if ensemble sample indicator is 0
                    % obj.symbolic_objective_template = obj.symbolic_objective_template + if_else(obj.ensemble_samples_sym(i)==1, s*sumsqr(disp_control_matrix), 0);
                    symbolic_objective_template(i) = symbolic_objective_template(i) + sumsqr(disp_control_matrix);

                    % Per-agent control constraints
                    if ~obj.config.use_linear_approximation
                        for j = 1:obj.N_agents
                            c_idx = obj.current_params.num_ensemble_members*obj.N_agents*k+obj.N_agents*(i-1)+j;
                            obj.symbolic_control_constraints_template{c_idx} = sumsqr(disp_control_matrix(:, j));
                        end
                    end
                end

                obj.symbolic_objective_template = symbolic_objective_template'*obj.ensemble_samples_sym/sum(obj.ensemble_samples_sym);
                
                if obj.config.use_linear_approximation
                    for j = 1:obj.N_agents
                        Currents_k_matrix_avg(:, j) = Currents_at_P0_sym{end}(:, j) + Js_at_P0_sym{end}(:, 2*(j-1)+1:2*j) * pose_diff(:, j);
                    end
                    Currents_k_vec_avg = reshape(Currents_k_matrix_avg, 2*obj.N_agents, 1);
                    disp_control_avg = disp_ground - Currents_k_vec_avg * obj.dt;
                    disp_control_matrix_avg = reshape(disp_control_avg, 2, obj.N_agents);
                    for j = 1:obj.N_agents
                        c_idx = obj.N_agents*k+j;
                        obj.symbolic_control_constraints_template{c_idx} = sumsqr(disp_control_matrix_avg(:, j));
                    end
                end
            end
            
            obj.templates_built = true;
            fprintf('Symbolic templates built successfully.\n');
        end

        function buildBenchmarkingExpressions(obj)
            % Build symbolic benchmarking templates for objective and constraints
            import casadi.*
            
            fprintf('Building benchmarking expressions for objective and constraints...\n');
            
            % Build symbolic objective and control constraints template
            obj.energy_training = 0; obj.energy_testing = 0;
            obj.control_constraints_training = cell(1, obj.N_agents*obj.current_params.num_ensemble_members*obj.T); 
            obj.control_constraints_testing = cell(1, obj.N_agents*obj.current_params.num_ensemble_members_test*obj.T);
    
            for k = 0:obj.T-1
                Currents_k_matrix = cell(1, obj.current_params.num_ensemble_members+1+obj.current_params.num_ensemble_members_test);  
                
                % Extract symbolic states and references
                P_k = obj.P0(:, k+1);        % Current state
                P_k_plus_1 = obj.P0(:, k+2); % Next state
                disp_ground = P_k_plus_1 - P_k;
                
                % Reshape to matrix form for ocean functions
                Pos_k_matrix = reshape(P_k, 2, obj.N_agents);
                
                % Exact Symbolic ocean evaluations
                [Currents_k_matrix{:}] = obj.ocean_current_func(Pos_k_matrix, k*obj.dt);

                for i = 1:obj.current_params.num_ensemble_members+1+obj.current_params.num_ensemble_members_test
                    
                    % Build objective contribution
                    Currents_k_vec = reshape(Currents_k_matrix{i}, 2*obj.N_agents, 1);
                    disp_control = disp_ground - Currents_k_vec * obj.dt;
                    disp_control_matrix = reshape(disp_control, 2, obj.N_agents);
                    
                    % Accumulate objective and control constraints
                    if i <= obj.current_params.num_ensemble_members
                        obj.energy_training = obj.energy_training + sumsqr(disp_control_matrix)/obj.current_params.num_ensemble_members;
                        for j = 1:obj.N_agents
                            c_idx = obj.current_params.num_ensemble_members*obj.N_agents*k+obj.N_agents*(i-1)+j;
                            obj.control_constraints_training{c_idx} = sumsqr(disp_control_matrix(:, j));
                        end
                    elseif i > obj.current_params.num_ensemble_members+1
                        obj.energy_testing = obj.energy_testing + sumsqr(disp_control_matrix)/obj.current_params.num_ensemble_members_test;
                        for j = 1:obj.N_agents
                            c_idx = obj.current_params.num_ensemble_members_test*obj.N_agents*k+obj.N_agents*(i-obj.current_params.num_ensemble_members-2)+j;
                            obj.control_constraints_testing{c_idx} = sumsqr(disp_control_matrix(:, j));
                        end
                    end
                end
            end
            obj.buildBenchmarkingConstraintsAndBounds();
            
            fprintf('Benchmarking expressions built successfully.\n');
        end
        
        function nlp = getParameterizedNLP(obj)
            % Get parameterized NLP structure with P0 as parameter
            % Uses symbolic expressions (not Function objects) as required by CasADi
            import casadi.*
            
            % Build symbolic templates if not already built
            if ~obj.templates_built
                obj.buildSymbolicTemplates();
            end
            
            % Decision variables (MX)
            w = obj.P_sym(:);
            
            % Parameters (MX) - reference trajectory P0 and ensemble sample indicator
            p = [obj.P0_sym(:); obj.ensemble_samples_sym(:)];
            
            % Objective function (MX expression, not Function)
            f = obj.symbolic_objective_template;
            
            % Build constraints and bounds together - store for bounds method
            if size(obj.constraint_bounds_lbg, 1) == 0
                obj.buildAllConstraintsAndBounds();
            end
            
            % Combine all constraints into single expression
            if ~isempty(obj.constraint_exprs)
                g = vertcat(obj.constraint_exprs{:});
            else
                g = [];
            end
            
            % Create parameterized NLP structure (all symbolic expressions)
            if isempty(g)
                nlp = struct('f', f, 'x', w, 'p', p);
            else
                nlp = struct('f', f, 'x', w, 'p', p, 'g', g);
            end
            
            fprintf('  Total constraint expressions: %d\n', length(obj.constraint_exprs));
            if ~isempty(g)
                fprintf('  Total constraints after vertcat: %d\n', numel(g));
            end
            fprintf('Parameterized NLP created with %d parameters (symbolic expressions).\n', numel(p));
        end
        
        function buildAllConstraintsAndBounds(obj)
            % Build ALL constraints and bounds in a single unified pass
            % This ensures perfect synchronization between constraints and bounds
            
            obj.constraint_exprs = {};
            obj.constraint_bounds_lbg = [];
            obj.constraint_bounds_ubg = [];
            
            fprintf('Building constraints and bounds together (debug info):\n');
            
            % --- Control Constraints (Parameterized) ---
            control_count = 0;
            if obj.config.enable_control_constraints && ~isempty(obj.symbolic_control_constraints_template)
                control_count = length(obj.symbolic_control_constraints_template);
                obj.constraint_exprs = [obj.constraint_exprs, obj.symbolic_control_constraints_template];
                
                % Control constraint bounds
                if obj.config.use_linear_approximation
                    max_control_disp_sq = ((1-3*obj.current_params.noise_level)*obj.agent_params.max_speed * obj.dt)^2;
                else
                    max_control_disp_sq = (obj.agent_params.max_speed * obj.dt)^2;
                end
                obj.constraint_bounds_lbg = [obj.constraint_bounds_lbg; zeros(control_count, 1)];
                obj.constraint_bounds_ubg = [obj.constraint_bounds_ubg; max_control_disp_sq * ones(control_count, 1)];

                fprintf('  Control constraints: %d\n', control_count);
            end

            % --- Initial Constraints ---
            if obj.config.enable_initial_constraints
                initial_pos_vec = reshape(cat(2, obj.agents.position), 2*obj.N_agents, 1);
                obj.constraint_exprs{end+1} = obj.P_sym(:, 1) - initial_pos_vec;
                obj.constraint_bounds_lbg = [obj.constraint_bounds_lbg; zeros(2*obj.N_agents, 1)-eps];
                obj.constraint_bounds_ubg = [obj.constraint_bounds_ubg; zeros(2*obj.N_agents, 1)+eps];
                fprintf('  Initial constraints: %d\n', 2*obj.N_agents);
            end
            
            % --- Final Constraints ---
            if obj.config.enable_final_constraints
                final_pos_vec = reshape(cat(2, obj.agents.goal), 2*obj.N_agents, 1);
                obj.constraint_exprs{end+1} = obj.P_sym(:, obj.T+1) - final_pos_vec;
                obj.constraint_bounds_lbg = [obj.constraint_bounds_lbg; zeros(2*obj.N_agents, 1)-eps];
                obj.constraint_bounds_ubg = [obj.constraint_bounds_ubg; zeros(2*obj.N_agents, 1)+eps];
                fprintf('  Final constraints: %d\n', 2*obj.N_agents);
            end
            
            bench = false;

            % --- Collision Constraints ---
            if obj.config.enable_collision_constraints && obj.N_agents > 1
                obj.addCollisionConstraintsUnified(bench);
            end
            
            % --- Formation Constraints ---
            if obj.config.enable_formation_constraints && obj.sim_params.formation_enabled && obj.N_agents > 1
                obj.addFormationConstraintsUnified(bench);
            end
            
            % --- Obstacle Constraints ---
            if obj.config.enable_obstacle_constraints && ~isempty(obj.env_params.obstacles)
                obj.addObstacleConstraintsUnified(bench);
            end
            
            fprintf('  Total bounds: %d\n', length(obj.constraint_bounds_lbg));
            
            % Verify constraint-bounds alignment
            if length(obj.constraint_bounds_lbg) ~= length(obj.constraint_bounds_ubg)
                error('Internal error: lbg and ubg have different lengths!');
            end
        end

        function buildBenchmarkingConstraintsAndBounds(obj)
            % Build ALL constraints and bounds in a single unified pass
            % This ensures perfect synchronization between constraints and bounds
            
            fprintf('Building constraints and bounds together (debug info):\n');
            
            % --- Control Constraints ---
            control_count = length(obj.symbolic_control_constraints_template);
            
            obj.constraints_training = obj.control_constraints_training;
            obj.constraints_testing = obj.control_constraints_testing;

            % --- Initial Constraints ---
            if obj.config.enable_initial_constraints
                initial_pos_vec = reshape(cat(2, obj.agents.position), 2*obj.N_agents, 1);
                obj.constraints_training{end+1} = obj.P0(:, 1) - initial_pos_vec;
                obj.constraints_testing{end+1} = obj.P0(:, 1) - initial_pos_vec;
                fprintf('  Initial constraints: %d\n', 2*obj.N_agents);
            end
            
            % --- Final Constraints ---
            if obj.config.enable_final_constraints
                final_pos_vec = reshape(cat(2, obj.agents.goal), 2*obj.N_agents, 1);
                obj.constraints_training{end+1} = obj.P0(:, obj.T+1) - final_pos_vec;
                obj.constraints_testing{end+1} = obj.P0(:, obj.T+1) - final_pos_vec;
                fprintf('  Final constraints: %d\n', 2*obj.N_agents);
            end
            
            bench = true;

            % --- Collision Constraints ---
            if obj.config.enable_collision_constraints && obj.N_agents > 1
                obj.addCollisionConstraintsUnified(bench);
            end
            
            % --- Formation Constraints ---
            if obj.config.enable_formation_constraints && obj.sim_params.formation_enabled && obj.N_agents > 1
                obj.addFormationConstraintsUnified(bench);
            end
            
            % --- Obstacle Constraints ---
            if obj.config.enable_obstacle_constraints && ~isempty(obj.env_params.obstacles)
                obj.addObstacleConstraintsUnified(bench);
            end
            
            
            % --- Benchmarking Constraints ---
            max_control_disp_sq = (obj.agent_params.max_speed * obj.dt)^2;
            control_count_training = length(obj.control_constraints_training);
            control_count_testing = length(obj.control_constraints_testing);
            
            
            % Convert constraints_bench from cell array to a matrix
            obj.flattenConstraints();

            obj.lbg_training = [zeros(control_count_training, 1); obj.constraint_bounds_lbg(control_count+1:end)];
            obj.ubg_training = [max_control_disp_sq * ones(control_count_training, 1); obj.constraint_bounds_ubg(control_count+1:end)];
            obj.lbg_testing = [zeros(control_count_testing, 1); obj.constraint_bounds_lbg(control_count+1:end)];
            obj.ubg_testing = [max_control_disp_sq * ones(control_count_testing, 1); obj.constraint_bounds_ubg(control_count+1:end)];
        end


        function flattenConstraints(obj)
            flattened_constraints = [];
            for i = 1:length(obj.constraints_training)
                if iscell(obj.constraints_training{i})
                    if size(obj.constraints_training{i}, 1) > 1
                        % If the cell contains a vector, extract and concatenate each element
                        for j = 1:size(obj.constraints_training{i}, 1)
                            flattened_constraints = [flattened_constraints; obj.constraints_training{i}(j)];
                        end
                    else
                        % If already 1x1, just add it to the flattened array
                        flattened_constraints = [flattened_constraints; obj.constraints_training{i}];
                        end
                end
            end
            obj.constraints_training = flattened_constraints;

            flattened_constraints = [];
            for i = 1:length(obj.constraints_testing)
                if iscell(obj.constraints_testing{i})
                    if size(obj.constraints_testing{i}, 1) > 1
                        % If the cell contains a vector, extract and concatenate each element
                        for j = 1:size(obj.constraints_testing{i}, 1)
                            flattened_constraints = [flattened_constraints; obj.constraints_testing{i}(j)];
                        end
                    else
                        % If already 1x1, just add it to the flattened array
                        flattened_constraints = [flattened_constraints; obj.constraints_testing{i}];
                    end
                else
                    % If it's already a numeric value, add it directly
                    flattened_constraints = [flattened_constraints; obj.constraints_testing{i}];
                end
            end
            obj.constraints_testing = flattened_constraints;
        end
        
        function addCollisionConstraintsUnified(obj, constraint_exprs, lbg, ubg, bench)
            % Add collision constraints and bounds in unified manner
            min_dist_agent_sq = (2 * obj.agent_params.radius + obj.safety_margin)^2;
            
            switch obj.config.collision_method
                case 'minimum_distance'
                    obj.addMinDistanceCollisionConstraintsUnified(constraint_exprs, lbg, ubg, min_dist_agent_sq, bench);
                case 'pairwise'
                    obj.addPairwiseCollisionConstraintsUnified(constraint_exprs, lbg, ubg, min_dist_agent_sq, bench);
                otherwise
                    obj.addMinDistanceCollisionConstraintsUnified(constraint_exprs, lbg, ubg, min_dist_agent_sq, bench);
            end
        end
        
        function addMinDistanceCollisionConstraintsUnified(obj, min_dist_agent_sq, bench)
            % Add minimum distance collision constraints with bounds
            constraint_count = 0;
            for k = 1:obj.T
                Pos_k = reshape(obj.P_sym(:, k+1), 2, obj.N_agents);
                P0_k = reshape(obj.P0(:, k+1), 2, obj.N_agents);
                for i = 1:obj.N_agents-1
                    other_indices = setdiff(i:obj.N_agents, i);
                    if bench
                        distances_sq = sum((Pos_k(:,i) - Pos_k(:,other_indices)).^2, 1);
                        min_dist_sq = min(distances_sq);
                        obj.constraint_exprs{end+1} = min_dist_sq;
                        obj.constraint_bounds_lbg = [obj.constraint_bounds_lbg; min_dist_agent_sq];
                        obj.constraint_bounds_ubg = [obj.constraint_bounds_ubg; inf];
                    else
                        distances_sq_P0 = sum((P0_k(:,i) - P0_k(:,other_indices)).^2, 1);
                        min_dist_sq_P0 = min(distances_sq_P0);
                        obj.constraints_training{end+1} = min_dist_sq_P0;
                        obj.constraints_testing{end+1} = min_dist_sq_P0;
                    end
                    constraint_count = constraint_count + 1;
                end
            end
            fprintf('  Collision (min_distance) constraints: %d\n', constraint_count);
        end
        
        function addPairwiseCollisionConstraintsUnified(obj, min_dist_agent_sq, bench)
            % Add pairwise collision constraints with bounds
            [i_idx, j_idx] = find(triu(ones(obj.N_agents), 1));
            n_pairs = length(i_idx);
            constraint_count = 0;
            
            for k = 1:obj.T
                P_k_plus_1 = obj.P_sym(:, k+1);
                Pos_matrix = reshape(P_k_plus_1, 2, obj.N_agents);

                if bench
                    pos_i = Pos_matrix(:, i_idx);
                    pos_j = Pos_matrix(:, j_idx);
                    dist_sq_vec = sum((pos_i - pos_j).^2, 1);
                    obj.constraint_exprs{end+1} = dist_sq_vec';
                    obj.constraint_bounds_lbg = [obj.constraint_bounds_lbg; min_dist_agent_sq * ones(n_pairs, 1)];
                    obj.constraint_bounds_ubg = [obj.constraint_bounds_ubg; inf * ones(n_pairs, 1)];
                else
                    P0_k_plus_1 = obj.P0(:, k+1);
                    P0_matrix = reshape(P0_k_plus_1, 2, obj.N_agents);
                    dist_sq_vec_P0 = sum((P0_matrix(:, i_idx) - P0_matrix(:, j_idx)).^2, 1);
                    obj.constraints_training{end+1} = dist_sq_vec_P0';
                    obj.constraints_testing{end+1} = dist_sq_vec_P0';
                end

                constraint_count = constraint_count + n_pairs;
            end
            fprintf('  Collision (pairwise) constraints: %d\n', constraint_count);
        end
        
        function addFormationConstraintsUnified(obj, bench)
            % Add formation constraints with bounds
            formation_tolerance_sq_sum = (obj.agent_params.formation_tolerance^2) * obj.N_agents;
            Target_relative = obj.agent_params.formation_relative_positions;
            constraint_count = 0;
            
            for k = 1:obj.T
                P_k_plus_1 = obj.P_sym(:, k+1);
                Pos_k_plus_1_matrix = reshape(P_k_plus_1, 2, obj.N_agents);
                
                if bench
                    Centroid_k = sum2(Pos_k_plus_1_matrix) / obj.N_agents;
                    Relative_k = Pos_k_plus_1_matrix - Centroid_k;
                    dev_sq = reshape(Relative_k - Target_relative, 2*obj.N_agents, 1);
                    
                    obj.constraint_exprs{end+1} = dev_sq;
                    obj.constraint_bounds_lbg = [obj.constraint_bounds_lbg; 0*ones(2*obj.N_agents,1)];
                    obj.constraint_bounds_ubg = [obj.constraint_bounds_ubg; formation_tolerance_sq_sum*ones(2*obj.N_agents,1)];
                else
                    P0_k_plus_1 = obj.P0(:, k+1);
                    P0_k_plus_1_matrix = reshape(P0_k_plus_1, 2, obj.N_agents);
                    Centroid_k_P0 = sum(P0_k_plus_1_matrix,2) / obj.N_agents;
                    Relative_k_P0 = P0_k_plus_1_matrix - Centroid_k_P0;
                    dev_sq_P0 = reshape(Relative_k_P0 - Target_relative, 2*obj.N_agents, 1);
                    obj.constraints_training{end+1} = dev_sq_P0;
                    obj.constraints_testing{end+1} = dev_sq_P0;
                end

                constraint_count = constraint_count + 2*obj.N_agents;
            end
            fprintf('  Formation constraints: %d\n', constraint_count);
        end
        
        function addObstacleConstraintsUnified(obj, bench)
            % Add obstacle constraints with bounds
            obs_centers_matrix = cat(2, obj.env_params.obstacles.center);
            obs_radii = cat(2, obj.env_params.obstacles.radius);
            min_dist_sq_all_obs = (obj.agent_params.radius + obs_radii + obj.safety_margin).^2;
            constraint_count = 0;

            for k = 1:obj.T
                P_k_plus_1 = obj.P_sym(:, k+1);
                Pos_k_plus_1_matrix = reshape(P_k_plus_1, 2, obj.N_agents);
                P0_k_plus_1 = obj.P0(:, k+1);
                P0_k_plus_1_matrix = reshape(P0_k_plus_1, 2, obj.N_agents);

                for i = 1:obj.N_agents
                    if bench
                        Agent_pos_ik = Pos_k_plus_1_matrix(:, i);
                        dist_sq_all_obs = sumsqr(Agent_pos_ik - obs_centers_matrix);
                        obj.constraint_exprs{end+1} = dist_sq_all_obs;
                        obj.constraint_bounds_lbg = [obj.constraint_bounds_lbg; min_dist_sq_all_obs'];
                        obj.constraint_bounds_ubg = [obj.constraint_bounds_ubg; inf*ones(length(obs_radii),1)];
                    else
                        Agent_pos_ik_P0 = P0_k_plus_1_matrix(:, i); 
                        dist_sq_all_obs_P0 = sumsqr(Agent_pos_ik_P0 - obs_centers_matrix);
                        obj.constraints_training{end+1} = dist_sq_all_obs_P0;
                        obj.constraints_testing{end+1} = dist_sq_all_obs_P0;
                    end
                    constraint_count = constraint_count + length(obs_radii);
                end
            end
            fprintf('  Obstacle constraints: %d\n', constraint_count);
        end
        
        
        function [lbg, ubg] = getParameterizedConstraintBounds(obj)
            % Get constraint bounds for parameterized NLP 
            % These were computed when getParameterizedNLP() was called
            
            if isempty(obj.constraint_bounds_lbg) || isempty(obj.constraint_bounds_ubg)
                error('Constraint bounds not available. Call getParameterizedNLP() first.');
            end
            
            lbg = obj.constraint_bounds_lbg;
            ubg = obj.constraint_bounds_ubg;
            
            fprintf('Returning cached constraint bounds: %d constraints\n', length(lbg));
        end
        
        function w0 = getInitialGuess(obj)
            % Return flattened P0 as initial guess
            w0 = obj.P0(:);
        end

        function metrics = getBenchmarkingMetrics(obj)
            % Get benchmarking metrics
            metrics = struct();
            metrics.training_energy = full(obj.energy_training);
            metrics.testing_energy = full(obj.energy_testing);

            % Count control constraint violations
            control_count_training = length(obj.control_constraints_training);
            % Handle CasADi cell array comparison with vector bounds
            violations_lower = 0; 
            violations_upper = 0;

            for i = 1:control_count_training
                constraint_value = full(obj.control_constraints_training{i});
                if constraint_value < obj.lbg_training(i) - 1e-6
                    violations_lower = violations_lower + 1;
                end
                if constraint_value > obj.ubg_training(i) + 1e-6
                    violations_upper = violations_upper + 1;
                end
            end
            metrics.training_control_constraint_violations = violations_lower + violations_upper;

            % Count control testing constraint violations
            control_count_testing = length(obj.control_constraints_testing);
            violations_lower = 0;
            violations_upper = 0;
            for i = 1:control_count_testing
                constraint_value = full(obj.control_constraints_testing{i});
                if constraint_value < obj.lbg_testing(i) - 1e-6
                    violations_lower = violations_lower + 1;
                end
                if constraint_value > obj.ubg_testing(i) + 1e-6
                    violations_upper = violations_upper + 1;
                end
            end
            metrics.testing_control_constraint_violations = violations_lower + violations_upper;
            
            % Count other constraint violations
            violations_lower = 0;
            violations_upper = 0;
            for i = control_count_training+1:length(obj.constraints_training)
                if full(obj.constraints_training{i}) < obj.lbg_training(i) - 1e-6
                    violations_lower = violations_lower + 1;
                end
                if full(obj.constraints_training{i}) > obj.ubg_training(i) + 1e-6
                    violations_upper = violations_upper + 1;
                end
            end
            metrics.training_constraint_violations = violations_lower + violations_upper;    
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
            config.use_linear_approximation = false;
            config.use_stochastic_sampling = false;
        end
    end
end 