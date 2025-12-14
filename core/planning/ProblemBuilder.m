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
        
        % Pre-allocation tracking indices (for constraint building)
        current_cell_idx
        current_bound_idx
        current_training_idx
        current_testing_idx
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
            
            % Initialize tracking indices
            obj.current_cell_idx = 0;
            obj.current_bound_idx = 0;
            obj.current_training_idx = 0;
            obj.current_testing_idx = 0;
            
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
            [obj.ocean_current_func, obj.ocean_gradient_func] = create_symbolic_ocean_func(obj.current_params, false);
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
                    
                    n_ens = obj.current_params.num_ensemble_members;
                    all_C0 = horzcat(Currents_at_P0_sym{1:n_ens});
                    all_Js = horzcat(Js_at_P0_sym{1:n_ens});
                    pd = repmat(pose_diff, 1, n_ens);
                    
                    a = all_Js(1,1:2:end); b = all_Js(1,2:2:end);
                    c = all_Js(2,1:2:end); d = all_Js(2,2:2:end);
                    
                    all_Currents = all_C0 + [a.*pd(1,:)+b.*pd(2,:); c.*pd(1,:)+d.*pd(2,:)];
                    
                    Currents_k_matrix(1:n_ens) = mat2cell(all_Currents, 2, repmat(obj.N_agents, 1, n_ens));
                else
                    % Exact Symbolic ocean evaluations
                    [Currents_k_matrix{:}] = obj.ocean_current_func(Pos_k_matrix, k*obj.dt);
                end

                for i = 1:obj.current_params.num_ensemble_members
                    % Calculate currents using linear approximation template
                    % if obj.config.use_linear_approximation
                    %     % Template Per-agent linear approximation
                    %     Currents_k_matrix{i} = MX.zeros(2, obj.N_agents);
                    %     for j = 1:obj.N_agents
                    %         Currents_k_matrix{i}(:, j) = Currents_at_P0_sym{i}(:, j) + Js_at_P0_sym{i}(:, 2*(j-1)+1:2*j) * pose_diff(:, j);
                    %     end
                    % end
                    
                    % Build template objective contribution
                    Currents_k_vec = reshape(Currents_k_matrix{i}, 2*obj.N_agents, 1);
                    disp_control = disp_ground - Currents_k_vec * obj.dt;
                    disp_control_matrix = reshape(disp_control, 2, obj.N_agents);
                    
                    % Accumulate objective in symbolic template
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
            obj.energy_training = obj.energy_training/obj.dt;
            obj.energy_testing = obj.energy_testing/obj.dt;
            % fprintf('Average Control effort per timestep (training): %.4f\n', obj.energy_training/obj.T);
            % fprintf('Average Control effort per timestep (testing): %.4f\n', obj.energy_testing/obj.T);
            % fprintf('Timesteps: %d, Time resolution: %.4f, Total Time: %.4f\n', obj.T, obj.dt, obj.T*obj.dt);
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
            % Uses pre-allocation for performance
            
            fprintf('Building constraints and bounds together (debug info):\n');
            
            % === STEP 1: Count total constraints upfront ===
            total_constraint_cells = 0;
            total_bounds = 0;
            
            % Control constraints
            control_count = 0;
            if obj.config.enable_control_constraints && ~isempty(obj.symbolic_control_constraints_template)
                control_count = length(obj.symbolic_control_constraints_template);
                total_constraint_cells = total_constraint_cells + control_count;
                total_bounds = total_bounds + control_count;
            end
            
            % Initial constraints (1 cell entry, 2*N_agents bounds)
            if obj.config.enable_initial_constraints
                total_constraint_cells = total_constraint_cells + 1;
                total_bounds = total_bounds + 2*obj.N_agents;
            end
            
            % Final constraints (1 cell entry, 2*N_agents bounds)
            if obj.config.enable_final_constraints
                total_constraint_cells = total_constraint_cells + 1;
                total_bounds = total_bounds + 2*obj.N_agents;
            end
            
            % Collision constraints
            if obj.config.enable_collision_constraints && obj.N_agents > 1
                if strcmp(obj.config.collision_method, 'pairwise')
                    n_pairs = obj.N_agents * (obj.N_agents - 1) / 2;
                    total_constraint_cells = total_constraint_cells + obj.T;  % 1 per timestep
                    total_bounds = total_bounds + obj.T * n_pairs;
                else  % minimum_distance
                    total_constraint_cells = total_constraint_cells + obj.T * (obj.N_agents - 1);
                    total_bounds = total_bounds + obj.T * (obj.N_agents - 1);
                end
            end
            
            % Formation constraints
            if obj.config.enable_formation_constraints && obj.sim_params.formation_enabled && obj.N_agents > 1
                total_constraint_cells = total_constraint_cells + obj.T;  % 1 per timestep
                total_bounds = total_bounds + obj.T * 2 * obj.N_agents;
            end
            
            % Obstacle constraints
            if obj.config.enable_obstacle_constraints && ~isempty(obj.env_params.obstacles)
                n_obs = length(obj.env_params.obstacles);
                total_constraint_cells = total_constraint_cells + obj.T * obj.N_agents;
                total_bounds = total_bounds + obj.T * obj.N_agents * n_obs;
            end
            
            % === STEP 2: Pre-allocate arrays ===
            obj.constraint_exprs = cell(1, total_constraint_cells);
            obj.constraint_bounds_lbg = zeros(total_bounds, 1);
            obj.constraint_bounds_ubg = zeros(total_bounds, 1);
            
            % Reset tracking indices
            obj.current_cell_idx = 0;
            obj.current_bound_idx = 0;
            
            % === STEP 3: Fill arrays using index-based assignment ===
            
            % --- Control Constraints ---
            if obj.config.enable_control_constraints && ~isempty(obj.symbolic_control_constraints_template)
                obj.constraint_exprs(1:control_count) = obj.symbolic_control_constraints_template;
                obj.current_cell_idx = control_count;
                
                if obj.config.use_linear_approximation
                    max_control_disp_sq = ((1-3*obj.current_params.noise_level)*obj.agent_params.max_speed * obj.dt)^2;
                else
                    max_control_disp_sq = (obj.agent_params.max_speed * obj.dt)^2;
                end
                obj.constraint_bounds_lbg(1:control_count) = 0;
                obj.constraint_bounds_ubg(1:control_count) = max_control_disp_sq;
                obj.current_bound_idx = control_count;
                
                fprintf('  Control constraints: %d\n', control_count);
            end

            % --- Initial Constraints ---
            if obj.config.enable_initial_constraints
                initial_pos_vec = reshape(cat(2, obj.agents.position), 2*obj.N_agents, 1);
                obj.current_cell_idx = obj.current_cell_idx + 1;
                obj.constraint_exprs{obj.current_cell_idx} = obj.P_sym(:, 1) - initial_pos_vec;
                
                n_init = 2*obj.N_agents;
                obj.constraint_bounds_lbg(obj.current_bound_idx+1:obj.current_bound_idx+n_init) = -eps;
                obj.constraint_bounds_ubg(obj.current_bound_idx+1:obj.current_bound_idx+n_init) = eps;
                obj.current_bound_idx = obj.current_bound_idx + n_init;
                fprintf('  Initial constraints: %d\n', n_init);
            end
            
            % --- Final Constraints ---
            if obj.config.enable_final_constraints
                final_pos_vec = reshape(cat(2, obj.agents.goal), 2*obj.N_agents, 1);
                obj.current_cell_idx = obj.current_cell_idx + 1;
                obj.constraint_exprs{obj.current_cell_idx} = obj.P_sym(:, obj.T+1) - final_pos_vec;
                
                n_final = 2*obj.N_agents;
                obj.constraint_bounds_lbg(obj.current_bound_idx+1:obj.current_bound_idx+n_final) = -eps;
                obj.constraint_bounds_ubg(obj.current_bound_idx+1:obj.current_bound_idx+n_final) = eps;
                obj.current_bound_idx = obj.current_bound_idx + n_final;
                fprintf('  Final constraints: %d\n', n_final);
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
            % Uses pre-allocation for performance
            
            fprintf('Building constraints and bounds together (debug info):\n');
            
            % --- Control Constraints ---
            control_count = length(obj.symbolic_control_constraints_template);
            
            % Calculate total additional constraints for training/testing
            n_additional = 0;
            if obj.config.enable_initial_constraints
                n_additional = n_additional + 1;
            end
            if obj.config.enable_final_constraints
                n_additional = n_additional + 1;
            end
            if obj.config.enable_collision_constraints && obj.N_agents > 1
                if strcmp(obj.config.collision_method, 'pairwise')
                    n_additional = n_additional + obj.T;
                else
                    n_additional = n_additional + obj.T * (obj.N_agents - 1);
                end
            end
            if obj.config.enable_formation_constraints && obj.sim_params.formation_enabled && obj.N_agents > 1
                n_additional = n_additional + obj.T;
            end
            if obj.config.enable_obstacle_constraints && ~isempty(obj.env_params.obstacles)
                n_additional = n_additional + obj.T * obj.N_agents;
            end
            
            % Pre-allocate constraints_training and constraints_testing
            total_training = length(obj.control_constraints_training) + n_additional;
            total_testing = length(obj.control_constraints_testing) + n_additional;
            
            obj.constraints_training = cell(1, total_training);
            obj.constraints_testing = cell(1, total_testing);
            
            % Copy control constraints
            obj.constraints_training(1:length(obj.control_constraints_training)) = obj.control_constraints_training;
            obj.constraints_testing(1:length(obj.control_constraints_testing)) = obj.control_constraints_testing;
            
            % Initialize tracking indices
            obj.current_training_idx = length(obj.control_constraints_training);
            obj.current_testing_idx = length(obj.control_constraints_testing);

            % --- Initial Constraints ---
            if obj.config.enable_initial_constraints
                initial_pos_vec = reshape(cat(2, obj.agents.position), 2*obj.N_agents, 1);
                obj.current_training_idx = obj.current_training_idx + 1;
                obj.current_testing_idx = obj.current_testing_idx + 1;
                obj.constraints_training{obj.current_training_idx} = obj.P0(:, 1) - initial_pos_vec;
                obj.constraints_testing{obj.current_testing_idx} = obj.P0(:, 1) - initial_pos_vec;
                fprintf('  Initial constraints: %d\n', 2*obj.N_agents);
            end
            
            % --- Final Constraints ---
            if obj.config.enable_final_constraints
                final_pos_vec = reshape(cat(2, obj.agents.goal), 2*obj.N_agents, 1);
                obj.current_training_idx = obj.current_training_idx + 1;
                obj.current_testing_idx = obj.current_testing_idx + 1;
                obj.constraints_training{obj.current_training_idx} = obj.P0(:, obj.T+1) - final_pos_vec;
                obj.constraints_testing{obj.current_testing_idx} = obj.P0(:, obj.T+1) - final_pos_vec;
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
            
            % Convert constraints from cell array to matrix using vertcat
            obj.flattenConstraints();

            obj.lbg_training = [zeros(control_count_training, 1); obj.constraint_bounds_lbg(control_count+1:end)];
            obj.ubg_training = [max_control_disp_sq * ones(control_count_training, 1); obj.constraint_bounds_ubg(control_count+1:end)];
            obj.lbg_testing = [zeros(control_count_testing, 1); obj.constraint_bounds_lbg(control_count+1:end)];
            obj.ubg_testing = [max_control_disp_sq * ones(control_count_testing, 1); obj.constraint_bounds_ubg(control_count+1:end)];
        end


        function flattenConstraints(obj)
            % Efficiently flatten cell arrays using vertcat instead of iterative concatenation
            
            % For training constraints - filter non-empty cells and vertcat
            if ~isempty(obj.constraints_training) && iscell(obj.constraints_training)
                valid_mask = ~cellfun(@isempty, obj.constraints_training);
                if any(valid_mask)
                    obj.constraints_training = vertcat(obj.constraints_training{valid_mask});
                else
                    obj.constraints_training = [];
                end
            end
            
            % For testing constraints - filter non-empty cells and vertcat
            if ~isempty(obj.constraints_testing) && iscell(obj.constraints_testing)
                valid_mask = ~cellfun(@isempty, obj.constraints_testing);
                if any(valid_mask)
                    obj.constraints_testing = vertcat(obj.constraints_testing{valid_mask});
                else
                    obj.constraints_testing = [];
                end
            end
        end
        
        function addCollisionConstraintsUnified(obj, bench)
            % Add collision constraints and bounds in unified manner
            min_dist_agent_sq = (2 * obj.agent_params.radius + obj.safety_margin)^2;
            
            switch obj.config.collision_method
                case 'minimum_distance'
                    obj.addMinDistanceCollisionConstraintsUnified(min_dist_agent_sq, bench);
                case 'pairwise'
                    obj.addPairwiseCollisionConstraintsUnified(min_dist_agent_sq, bench);
                otherwise
                    obj.addMinDistanceCollisionConstraintsUnified(min_dist_agent_sq, bench);
            end
        end
        
        function addMinDistanceCollisionConstraintsUnified(obj, min_dist_agent_sq, bench)
            % Add minimum distance collision constraints with bounds
            % Uses pre-allocated arrays with index tracking
            constraint_count = 0;
            for k = 1:obj.T
                Pos_k = reshape(obj.P_sym(:, k+1), 2, obj.N_agents);
                P0_k = reshape(obj.P0(:, k+1), 2, obj.N_agents);
                for i = 1:obj.N_agents-1
                    other_indices = setdiff(i:obj.N_agents, i);
                    if ~bench
                        distances_sq = sum((Pos_k(:,i) - Pos_k(:,other_indices)).^2, 1);
                        min_dist_sq = min(distances_sq);
                        
                        obj.current_cell_idx = obj.current_cell_idx + 1;
                        obj.constraint_exprs{obj.current_cell_idx} = min_dist_sq;
                        
                        obj.current_bound_idx = obj.current_bound_idx + 1;
                        obj.constraint_bounds_lbg(obj.current_bound_idx) = min_dist_agent_sq;
                        obj.constraint_bounds_ubg(obj.current_bound_idx) = inf;
                    else
                        distances_sq_P0 = sum((P0_k(:,i) - P0_k(:,other_indices)).^2, 1);
                        min_dist_sq_P0 = min(distances_sq_P0);
                        
                        obj.current_training_idx = obj.current_training_idx + 1;
                        obj.current_testing_idx = obj.current_testing_idx + 1;
                        obj.constraints_training{obj.current_training_idx} = min_dist_sq_P0;
                        obj.constraints_testing{obj.current_testing_idx} = min_dist_sq_P0;
                    end
                    constraint_count = constraint_count + 1;
                end
            end
            fprintf('  Collision (min_distance) constraints: %d\n', constraint_count);
        end
        
        function addPairwiseCollisionConstraintsUnified(obj, min_dist_agent_sq, bench)
            % Add pairwise collision constraints with bounds
            % Uses pre-allocated arrays with index tracking
            [i_idx, j_idx] = find(triu(ones(obj.N_agents), 1));
            n_pairs = length(i_idx);
            constraint_count = 0;
            
            for k = 1:obj.T
                P_k_plus_1 = obj.P_sym(:, k+1);
                Pos_matrix = reshape(P_k_plus_1, 2, obj.N_agents);

                if ~bench
                    pos_i = Pos_matrix(:, i_idx);
                    pos_j = Pos_matrix(:, j_idx);
                    dist_sq_vec = sum((pos_i - pos_j).^2, 1);
                    
                    obj.current_cell_idx = obj.current_cell_idx + 1;
                    obj.constraint_exprs{obj.current_cell_idx} = dist_sq_vec';
                    
                    obj.constraint_bounds_lbg(obj.current_bound_idx+1:obj.current_bound_idx+n_pairs) = min_dist_agent_sq;
                    obj.constraint_bounds_ubg(obj.current_bound_idx+1:obj.current_bound_idx+n_pairs) = inf;
                    obj.current_bound_idx = obj.current_bound_idx + n_pairs;
                else
                    P0_k_plus_1 = obj.P0(:, k+1);
                    P0_matrix = reshape(P0_k_plus_1, 2, obj.N_agents);
                    dist_sq_vec_P0 = sum((P0_matrix(:, i_idx) - P0_matrix(:, j_idx)).^2, 1);
                    
                    obj.current_training_idx = obj.current_training_idx + 1;
                    obj.current_testing_idx = obj.current_testing_idx + 1;
                    obj.constraints_training{obj.current_training_idx} = dist_sq_vec_P0';
                    obj.constraints_testing{obj.current_testing_idx} = dist_sq_vec_P0';
                end

                constraint_count = constraint_count + n_pairs;
            end
            fprintf('  Collision (pairwise) constraints: %d\n', constraint_count);
        end
        
        function addFormationConstraintsUnified(obj, bench)
            % Add formation constraints with bounds
            % Uses pre-allocated arrays with index tracking
            formation_tolerance_sq_sum = (obj.agent_params.formation_tolerance^2) * obj.N_agents;
            Target_relative = obj.agent_params.formation_relative_positions;
            constraint_count = 0;
            
            for k = 1:obj.T
                P_k_plus_1 = obj.P_sym(:, k+1);
                Pos_k_plus_1_matrix = reshape(P_k_plus_1, 2, obj.N_agents);
                
                if ~bench
                    Centroid_k = sum2(Pos_k_plus_1_matrix) / obj.N_agents;
                    Relative_k = Pos_k_plus_1_matrix - Centroid_k;
                    dev_sq = reshape(Relative_k - Target_relative, 2*obj.N_agents, 1);
                    
                    obj.current_cell_idx = obj.current_cell_idx + 1;
                    obj.constraint_exprs{obj.current_cell_idx} = dev_sq;
                    
                    n_bounds = 2*obj.N_agents;
                    obj.constraint_bounds_lbg(obj.current_bound_idx+1:obj.current_bound_idx+n_bounds) = 0;
                    obj.constraint_bounds_ubg(obj.current_bound_idx+1:obj.current_bound_idx+n_bounds) = formation_tolerance_sq_sum;
                    obj.current_bound_idx = obj.current_bound_idx + n_bounds;
                else
                    P0_k_plus_1 = obj.P0(:, k+1);
                    P0_k_plus_1_matrix = reshape(P0_k_plus_1, 2, obj.N_agents);
                    Centroid_k_P0 = sum(P0_k_plus_1_matrix,2) / obj.N_agents;
                    Relative_k_P0 = P0_k_plus_1_matrix - Centroid_k_P0;
                    dev_sq_P0 = reshape(Relative_k_P0 - Target_relative, 2*obj.N_agents, 1);
                    
                    obj.current_training_idx = obj.current_training_idx + 1;
                    obj.current_testing_idx = obj.current_testing_idx + 1;
                    obj.constraints_training{obj.current_training_idx} = dev_sq_P0;
                    obj.constraints_testing{obj.current_testing_idx} = dev_sq_P0;
                end

                constraint_count = constraint_count + 2*obj.N_agents;
            end
            fprintf('  Formation constraints: %d\n', constraint_count);
        end
        
        function addObstacleConstraintsUnified(obj, bench)
            % Add obstacle constraints with bounds
            % Uses pre-allocated arrays with index tracking
            obs_centers_matrix = cat(2, obj.env_params.obstacles.center);
            obs_radii = cat(2, obj.env_params.obstacles.radius);
            min_dist_sq_all_obs = (obj.agent_params.radius + obs_radii + obj.safety_margin).^2;
            n_obs = length(obs_radii);
            constraint_count = 0;

            for k = 1:obj.T
                P_k_plus_1 = obj.P_sym(:, k+1);
                Pos_k_plus_1_matrix = reshape(P_k_plus_1, 2, obj.N_agents);
                P0_k_plus_1 = obj.P0(:, k+1);
                P0_k_plus_1_matrix = reshape(P0_k_plus_1, 2, obj.N_agents);

                for i = 1:obj.N_agents
                    if ~bench
                        Agent_pos_ik = Pos_k_plus_1_matrix(:, i);
                        dist_sq_all_obs = sumsqr(Agent_pos_ik - obs_centers_matrix);
                        
                        obj.current_cell_idx = obj.current_cell_idx + 1;
                        obj.constraint_exprs{obj.current_cell_idx} = dist_sq_all_obs;
                        
                        obj.constraint_bounds_lbg(obj.current_bound_idx+1:obj.current_bound_idx+n_obs) = min_dist_sq_all_obs';
                        obj.constraint_bounds_ubg(obj.current_bound_idx+1:obj.current_bound_idx+n_obs) = inf;
                        obj.current_bound_idx = obj.current_bound_idx + n_obs;
                    else
                        Agent_pos_ik_P0 = P0_k_plus_1_matrix(:, i); 
                        dist_sq_all_obs_P0 = sumsqr(Agent_pos_ik_P0 - obs_centers_matrix);
                        
                        obj.current_training_idx = obj.current_training_idx + 1;
                        obj.current_testing_idx = obj.current_testing_idx + 1;
                        obj.constraints_training{obj.current_training_idx} = dist_sq_all_obs_P0;
                        obj.constraints_testing{obj.current_testing_idx} = dist_sq_all_obs_P0;
                    end
                    constraint_count = constraint_count + n_obs;
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
                if full(obj.constraints_training(i)) < obj.lbg_training(i) - 1e-6
                    violations_lower = violations_lower + 1;
                end
                if full(obj.constraints_training(i)) > obj.ubg_training(i) + 1e-6
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