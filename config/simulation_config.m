function [sim_params, env_params, current_params, agent_params, video_params] = simulation_config(num_agents, num_ensemble_members, noise_level, formation_enabled)
    % SIMULATION_CONFIG Returns all simulation configuration parameters
    %
    % This function replaces the hardcoded parameter definitions in main_simulation.m
    % All parameters are defined here in one place for easy modification.
    %
    % Outputs:
    %   sim_params    - Simulation parameters structure
    %   env_params    - Environment parameters structure  
    %   current_params - Ocean current parameters structure
    %   agent_params  - Agent parameters structure
    %   video_params  - Video export parameters structure
    
    fprintf('Setting up simulation parameters...\n');
    
    % --- Simulation Parameters ---
    sim_params.dt = 1;                       % Simulation time step (s)
    sim_params.T_final = 200;                 % Total simulation time (s)
    sim_params.time_steps = floor(sim_params.T_final / sim_params.dt);
    sim_params.visualization = true;         % Enable/disable visualization
    sim_params.vis_interval = 5;             % Update visualization every N steps
    sim_params.vis_vector_scale = 1;         % Scaling factor for velocity vector visualization
    sim_params.formation_enabled = formation_enabled;     % toggle formation control
    sim_params.planning_horizon = sim_params.T_final/sim_params.dt;       % Number of steps planner looks ahead
    sim_params.replan_interval = sim_params.T_final/sim_params.dt;        % Replan interval 
    
    % --- algorithm related parameters ---
    sim_params.algo = {'astar', 'fullOpt', 'ssca', 'dssca', 'stomp', 'eesto'}; % {'fullOpt','ssca','dssca'};               % Planning algorithm: 'fullOpt', 'sca', 'ssca'
    sim_params.initial_guess = 'aStar'; % 'straightline' or 'aStar'
    if any([strcmp(sim_params.algo, 'ssca'), strcmp(sim_params.algo, 'dssca')]) 
        sim_params.max_outer_iterations = 50;
        sim_params.mu = 1e-6;
        sim_params.k_bar = 1;
        sim_params.w = 1;
        sim_params.c = 1;

        sim_params.learning_rate = sim_params.k_bar / (sim_params.w)^(1/3);
        sim_params.gradient_tracking_weight = sim_params.c * sim_params.learning_rate^2;
    end
    sim_params.gradient_required_by_planner = false;

    % --- Environment Parameters ---
    env_params.x_limits = [-50, 50];         % Environment boundaries (m)
    env_params.y_limits = [-50, 50];         % Environment boundaries (m)
    env_params.obstacles = struct('center', {}, 'radius', {}); % Obstacles structure
    
    if false
        % Example obstacles
        env_params.obstacles(1) = struct('center', [17; 17], 'radius', 15);
        env_params.obstacles(2) = struct('center', [-30; 15], 'radius', 7);
        env_params.obstacles(3) = struct('center', [-25; -25], 'radius', 10);
    else
        % Generate a specified number of random obstacles within the environment bounds
        num_obstacles = 3; % You can change this number as desired
        env_params.obstacles = generate_random_obstacles(num_obstacles, env_params.x_limits, env_params.y_limits);
    end

    % --- Ocean Current Parameters ---
    current_params.type = 'static';           % 'static' or 'time_varying'
    current_params.vortices = struct('center', {}, 'strength', {}, 'core_radius', {});
    current_params.vortices(1) = struct('center', [10; 10], 'strength', 25*4, 'core_radius', 20);
    current_params.vortices(2) = struct('center', [-15; -15], 'strength', -20*4, 'core_radius', 30);
    if strcmp(current_params.type, 'time_varying')
        current_params.vortices_end(1) = struct('center', [-10; 10], 'strength', 25*4, 'core_radius', 20);
        current_params.vortices_end(2) = struct('center', [15; -15], 'strength', -20*4, 'core_radius', 30);
        current_params.T_final = sim_params.T_final;
    end
    
    current_params.num_ensemble_members = num_ensemble_members;
    current_params.num_ensemble_members_test = floor(num_ensemble_members*0.25);
    current_params.noise_level = noise_level;         % Standard deviation of noise added to current estimate
    current_params.gradient_noise_level = 0; 

    % --- Agent Parameters ---
    agent_params.num_agents = num_agents;
    agent_params.radius = 1.5; 
    agent_params.max_speed = 5;
    agent_params.safety_margin = 0.1;
    agent_params.collision_weight = 0*1e2;
    agent_params.color = lines(num_agents); % Assign distinct colors

    % --- Formation Parameters ---
    agent_params.formation_inter_agent_distance = 15.0;
    agent_params.formation_tolerance = 1e-6;
    agent_params.formation_weight = 0.5*1e-2;
    radius = agent_params.formation_inter_agent_distance / 2;
    % Arrange agents on a circle of diameter d at equal angular distances
    angles = linspace(0, 2*pi, num_agents+1);
    angles(end) = []; % remove duplicated endpoint
    agent_params.formation_relative_positions = [radius * cos(angles); radius * sin(angles)];
    % Normalize relative positions so the mean is [0;0] if not already centered
    agent_params.formation_relative_positions = agent_params.formation_relative_positions - mean(agent_params.formation_relative_positions, 2);

    % --- Video Export Parameters ---
    video_params.enabled = false;                    % Enable/disable video export
    video_params.save_figure = false;                 % Enable/disable figure saving
    video_params.format = 'Motion JPEG AVI';        % Video format (fallback: auto-detect best available)
    video_params.quality = 95;                      % Video quality (0-100)
    video_params.framerate = 30;                    % Output video framerate

end



% ---- Helper function to generate random obstacles within environment ----
function obstacles = generate_random_obstacles(num, xlim, ylim)
    min_radius = 5;  % Minimum possible obstacle radius
    max_radius = 10; % Maximum possible obstacle radius

    obstacles = struct('center', {}, 'radius', {});
    for k = 1:num
        % Sample radius first
        radius = (max_radius - min_radius) * rand() + min_radius;
        % Now restrict the range for center so entire obstacle fits
        x_min_allowed = xlim(1) + radius;
        x_max_allowed = xlim(2) - radius;
        y_min_allowed = ylim(1) + radius;
        y_max_allowed = ylim(2) - radius;

        % In case environment is smaller than obstacle, clamp
        if x_min_allowed > x_max_allowed
            x_min_allowed = (xlim(1) + xlim(2))/2;
            x_max_allowed = x_min_allowed;
        end
        if y_min_allowed > y_max_allowed
            y_min_allowed = (ylim(1) + ylim(2))/2;
            y_max_allowed = y_min_allowed;
        end

        rand_x = (x_max_allowed - x_min_allowed) * rand() + x_min_allowed;
        rand_y = (y_max_allowed - y_min_allowed) * rand() + y_min_allowed;
        obstacles(k) = struct('center', [rand_x; rand_y], 'radius', radius);
    end
end