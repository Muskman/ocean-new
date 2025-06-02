function config = default_config()
%DEFAULT_CONFIG Create default configuration for ocean simulation
%   Returns a structure containing all simulation parameters with proper
%   documentation and validation.

    %% Simulation Parameters
    config.simulation = struct();
    config.simulation.dt = 0.5;                    % Time step (s)
    config.simulation.total_time = 500;            % Total simulation time (s)
    config.simulation.enable_visualization = true; % Enable/disable real-time visualization
    config.simulation.visualization_interval = 20; % Update visualization every N steps
    config.simulation.vector_scale = 1.0;          % Scaling factor for velocity vectors
    config.simulation.enable_formation = true;     % Enable formation control
    config.simulation.enable_gradient_estimation = true; % Enable current gradient estimation
    config.simulation.verbose = true;              % Enable verbose output
    config.simulation.save_results = true;         % Save simulation results
    
    %% Planning Parameters
    config.planning = struct();
    config.planning.horizon_time = 500;           % Planning horizon (s) - will be converted to steps
    config.planning.replan_interval_time = 500;   % Replanning interval (s) - will be converted to steps
    config.planning.safety_margin = 0.2;          % Safety margin for obstacles/collisions (m)
    config.planning.solver_tolerance = 1e-10;     % CasADi solver tolerance
    config.planning.max_iterations = 2000;        % Maximum solver iterations
    config.planning.solver_verbosity = 3;         % Solver output level (0=quiet, 3=normal, 5=verbose)
    config.planning.enable_warm_start = false;    % Enable solver warm starting
    config.planning.enforce_final_positions = false; % Force agents to reach exact goals
    config.planning.enable_collision_avoidance = true; % Enable inter-agent collision avoidance
    config.planning.boundary_margin = 0.0;        % Soft boundary margin (0 = use hard bounds)
    
    %% Environment Parameters
    config.environment = struct();
    config.environment.x_bounds = [-50, 50];      % Environment X boundaries (m)
    config.environment.y_bounds = [-50, 50];      % Environment Y boundaries (m)
    config.environment.obstacles = [];             % Will be populated by create_obstacles()
    
    %% Ocean Current Parameters
    config.ocean = struct();
    config.ocean.type = 'static';                 % 'static' or 'time_varying'
    config.ocean.noise_std = 0.0;                 % Current estimation noise std dev (m/s)
    config.ocean.gradient_noise_std = 0.0;        % Current gradient estimation noise std dev (1/s)
    config.ocean.vortices = create_default_vortices(); % Vortex definitions
    
    %% Agent Parameters
    config.agents = struct();
    config.agents.count = 4;                      % Number of agents
    config.agents.radius = 1.5;                   % Agent physical radius (m)
    config.agents.max_speed = 5.0;                % Maximum control speed (m/s)
    config.agents.initial_positions = [];         % Will be auto-generated if empty
    config.agents.goal_positions = [];            % Will be auto-generated if empty
    
    %% Formation Parameters
    config.formation = struct();
    config.formation.inter_agent_distance = 8.0;  % Desired distance between agents (m)
    config.formation.tolerance = 1e-2;            % Formation tolerance (m)
    config.formation.weight = 0.5;                % Formation cost weight
    config.formation.relative_positions = [];     % Will be auto-generated based on agent count
    config.formation.use_hard_constraints = false; % Use constraints instead of soft cost
    
    %% Visualization Parameters
    config.visualization = struct();
    config.visualization.figure_size = [800, 700]; % Figure size [width, height]
    config.visualization.figure_position = [100, 100]; % Figure position [x, y]
    config.visualization.grid_resolution_contour = 50; % Grid points for current contour
    config.visualization.grid_resolution_quiver = 20;  % Grid points for current vectors
    config.visualization.colormap = 'parula';     % Colormap for current visualization
    config.visualization.show_paths = true;       % Show agent trajectory paths
    config.visualization.show_plans = true;       % Show planned trajectories
    config.visualization.show_formation_links = true; % Show formation connections
    config.visualization.show_current_vectors = true; % Show current vector field
    config.visualization.show_agent_vectors = true;   % Show agent control/current vectors
    
    %% Derived Parameters (computed from base parameters)
    config = compute_derived_parameters(config);
    
    %% Validation
    config = validate_configuration(config);
end

function vortices = create_default_vortices()
%CREATE_DEFAULT_VORTICES Create default ocean vortex configuration
    vortices = struct('center', {}, 'strength', {}, 'core_radius', {});
    vortices(1) = struct('center', [10; 10], 'strength', 100, 'core_radius', 20);
    vortices(2) = struct('center', [-15; -15], 'strength', -80, 'core_radius', 30);
end

function config = compute_derived_parameters(config)
%COMPUTE_DERIVED_PARAMETERS Calculate dependent parameters
    
    % Convert time-based parameters to steps
    config.simulation.total_steps = floor(config.simulation.total_time / config.simulation.dt);
    config.planning.horizon_steps = max(1, floor(config.planning.horizon_time / config.simulation.dt));
    config.planning.replan_interval_steps = max(1, floor(config.planning.replan_interval_time / config.simulation.dt));
    
    % Generate formation positions if not specified
    if isempty(config.formation.relative_positions)
        config.formation.relative_positions = generate_formation_positions(...
            config.agents.count, config.formation.inter_agent_distance);
    end
    
    % Generate agent colors
    config.agents.colors = lines(config.agents.count);
    
    % Generate initial and goal positions if not specified
    if isempty(config.agents.initial_positions)
        config.agents.initial_positions = generate_initial_positions(config);
    end
    
    if isempty(config.agents.goal_positions)
        config.agents.goal_positions = generate_goal_positions(config);
    end
end

function formation_pos = generate_formation_positions(num_agents, spacing)
%GENERATE_FORMATION_POSITIONS Create formation geometry based on agent count
    
    switch num_agents
        case 3
            % Equilateral triangle
            h = spacing * sqrt(3)/2;
            R = h * 2/3;
            formation_pos = [0, R; -spacing/2, -h/3; spacing/2, -h/3]';
            
        case 4
            % Square formation
            formation_pos = [-spacing/2, spacing/2; spacing/2, spacing/2; 
                           -spacing/2, -spacing/2; spacing/2, -spacing/2]';
            
        otherwise
            % Linear formation
            formation_pos = zeros(2, num_agents);
            for i = 1:num_agents
                formation_pos(1, i) = (i - (num_agents+1)/2) * spacing;
            end
    end
    
    % Center the formation
    formation_pos = formation_pos - mean(formation_pos, 2);
end

function initial_pos = generate_initial_positions(config)
%GENERATE_INITIAL_POSITIONS Create default initial agent positions in formation
    
    num_agents = config.agents.count;
    
    % Get formation positions based on agent count
    formation_pos = generate_formation_positions(num_agents, config.formation.inter_agent_distance);
    
    % Calculate safe margins to ensure all agents stay within bounds
    x_margin = max(abs(formation_pos(1,:))) + config.agents.radius + 5;
    y_margin = max(abs(formation_pos(2,:))) + config.agents.radius + 5;
    
    % Generate random centroid position within safe bounds
    x_min = config.environment.x_bounds(1) + x_margin;
    x_max = config.environment.x_bounds(1) + (config.environment.x_bounds(2) - config.environment.x_bounds(1))/3;
    y_min = config.environment.y_bounds(1) + y_margin;
    y_max = config.environment.y_bounds(2) - y_margin;
    
    centroid = [
        x_min + rand() * (x_max - x_min);
        y_min + rand() * (y_max - y_min)
    ];
    
    % Apply formation positions to the centroid
    initial_pos = formation_pos + centroid;
end

function goal_pos = generate_goal_positions(config)
%GENERATE_GOAL_POSITIONS Create default goal positions in formation
    
    num_agents = config.agents.count;
    
    % Get formation positions based on agent count
    formation_pos = generate_formation_positions(num_agents, config.formation.inter_agent_distance);
    
    % Calculate safe margins to ensure all agents stay within bounds
    x_margin = max(abs(formation_pos(1,:))) + config.agents.radius + 5;
    y_margin = max(abs(formation_pos(2,:))) + config.agents.radius + 5;
    
    % Generate random centroid position within safe bounds
    x_min = config.environment.x_bounds(1) + (config.environment.x_bounds(2) - config.environment.x_bounds(1))*2/3;
    x_max = config.environment.x_bounds(2) - x_margin;
    y_min = config.environment.y_bounds(1) + y_margin;
    y_max = config.environment.y_bounds(2) - y_margin;
    
    centroid = [
        x_min + rand() * (x_max - x_min);
        y_min + rand() * (y_max - y_min)
    ];
    
    % Apply formation positions to the centroid
    goal_pos = formation_pos + centroid;
end

function config = validate_configuration(config)
%VALIDATE_CONFIGURATION Ensure configuration parameters are valid
    
    % Validate simulation parameters
    assert(config.simulation.dt > 0, 'Time step must be positive');
    assert(config.simulation.total_time > 0, 'Total time must be positive');
    assert(config.simulation.total_steps > 0, 'Must have at least one time step');
    
    % Validate environment bounds
    assert(config.environment.x_bounds(2) > config.environment.x_bounds(1), ...
        'X upper bound must be greater than lower bound');
    assert(config.environment.y_bounds(2) > config.environment.y_bounds(1), ...
        'Y upper bound must be greater than lower bound');
    
    % Validate agent parameters
    assert(config.agents.count > 0, 'Must have at least one agent');
    assert(config.agents.radius > 0, 'Agent radius must be positive');
    assert(config.agents.max_speed > 0, 'Maximum speed must be positive');
    
    % Validate formation parameters
    if config.simulation.enable_formation && config.agents.count > 1
        assert(config.formation.inter_agent_distance > 2 * config.agents.radius, ...
            'Formation distance must be greater than twice agent radius');
        assert(config.formation.tolerance > 0, 'Formation tolerance must be positive');
    end
    
    % Validate planning parameters
    assert(config.planning.horizon_steps > 0, 'Planning horizon must be at least one step');
    assert(config.planning.replan_interval_steps > 0, 'Replan interval must be at least one step');
    assert(config.planning.safety_margin >= 0, 'Safety margin must be non-negative');
    
    fprintf('Configuration validation passed.\n');
end 