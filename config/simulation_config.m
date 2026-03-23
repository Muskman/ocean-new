function [sim_params, env_params, current_params, agent_params, video_params] = simulation_config(opts)
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

    env_type = opts.env_type;
    current_type = opts.current_type;
    num_agents = opts.num_agents;
    num_obstacles = opts.num_obstacles;
    num_ensemble_members = opts.num_ensemble_members;
    noise_level = opts.noise_level;
    formation_enabled = opts.formation_enabled;
    formation_type = opts.formation_type;
    algorithms = opts.algorithms;
    initial_guess = opts.initial_guess;
    num_mc_simulations = opts.num_mc_simulations;
    random_seed = opts.random_seed;
    
    fprintf('Setting up simulation parameters...\n');
    
    % --- Simulation Parameters ---
    if strcmp(env_type, 'sim')
        sim_params.dt = 1;                         % Simulation time step (s)
        sim_params.T_final = 200;                  % Total simulation time (s)
    elseif strcmp(env_type, 'real')
        sim_params.dt = 3*1e3;                     % Simulation time step (s)
        sim_params.T_final = 6*1e5;                % Total simulation time (s)
    else
        error('Invalid environment type: %s | Valid options: sim, real', env_type);
    end
    sim_params.time_steps = floor(sim_params.T_final / sim_params.dt);
    sim_params.visualization = true;         % Enable/disable visualization
    sim_params.vis_interval = 2;             % Update visualization every N steps
    sim_params.vis_vector_scale = 1;         % Scaling factor for velocity vector visualization
    sim_params.formation_enabled = formation_enabled;     % toggle formation control
    sim_params.planning_horizon = sim_params.T_final/sim_params.dt;       % Number of steps planner looks ahead
    sim_params.replan_interval = sim_params.T_final/sim_params.dt;        % Replan interval 
    sim_params.random_seed = random_seed;

    % --- algorithm related parameters ---
    sim_params.algo = algorithms; % Planning algorithm: 'fullOpt', 'sca', 'ssca'
    sim_params.num_mc_simulations = num_mc_simulations;
    sim_params.initial_guess = initial_guess; % 'straightline' or 'astar'
    if any([strcmp(sim_params.algo, 'ssca'), strcmp(sim_params.algo, 'dssca')]) 
        sim_params.max_outer_iterations = 10;
        sim_params.mu = 1e-6;
        if strcmp(env_type, 'real')
            sim_params.k_bar = 1e2;
        else
            sim_params.k_bar = 1;
        end
        sim_params.w = sim_params.k_bar^3;
        sim_params.c = 1;

        sim_params.learning_rate = sim_params.k_bar / (sim_params.w)^(1/3);
        sim_params.gradient_tracking_weight = sim_params.c * sim_params.learning_rate^2;
    end
    sim_params.gradient_required_by_planner = false;

    % --- Environment Parameters ---
    if strcmp(env_type, 'sim')
        env_params.type = 'sim';
        current_params.env_type = 'sim';
        current_params.type = current_type;          % 'static' or 'time_varying'
        env_params.x_limits = [-50, 50];         % Environment boundaries (m)
        env_params.y_limits = [-50, 50];         % Environment boundaries (m)
        env_params.obstacles = generate_random_obstacles(num_obstacles, env_params.x_limits, env_params.y_limits, random_seed);
    elseif strcmp(env_type, 'real')
        env_params.type = 'real';
        current_params.env_type = 'real';
        current_params.type = 'static';
        x_min = 1; y_min = 1;
        x_max = 101; y_max = 101;
        env_params.x_limits_idx = [x_min, x_max];         % Environment boundaries (m)
        env_params.y_limits_idx = [y_min, y_max];         % Environment boundaries (m)
        env_params.obstacles = generate_random_obstacles(num_obstacles, env_params.x_limits_idx, env_params.y_limits_idx, random_seed);
    end
    
    if false
        % Example manually added obstacles
        env_params.obstacles(1) = struct('center', [17; 17], 'radius', 15);
        env_params.obstacles(2) = struct('center', [-30; 15], 'radius', 7);
        env_params.obstacles(3) = struct('center', [-25; -25], 'radius', 10);
    end

    % --- Ocean Current Parameters ---
    if strcmp(env_type, 'sim')
        current_params.vortices = struct('center', {}, 'strength', {}, 'core_radius', {});
        current_params.vortices(1) = struct('center', [10; 10], 'strength', 25*4, 'core_radius', 20);
        current_params.vortices(2) = struct('center', [-15; -15], 'strength', -20*4, 'core_radius', 30);
        if strcmp(current_params.type, 'time_varying')
            current_params.vortices_end(1) = struct('center', [-10; 10], 'strength', 25*4, 'core_radius', 20);
            current_params.vortices_end(2) = struct('center', [15; -15], 'strength', -20*4, 'core_radius', 30);
            current_params.T_final = sim_params.T_final;
        end
        current_params.dc = 1; dc_scale = 1;
    else
        [~,~,~,~,q_x_m,q_y_m,u,v,mag] = real_ocean_environment();
    
        current_params.X_lim = q_x_m(y_min:y_max,x_min:x_max); current_params.Y_lim = q_y_m(y_min:y_max,x_min:x_max);
        current_params.U = u(y_min:y_max,x_min:x_max); current_params.V = v(y_min:y_max,x_min:x_max); 
        current_params.mag = mag(y_min:y_max,x_min:x_max);

        [Uc, Vc, xc, yc, meta] = learn_ocean_currents_real(current_params);
        current_params.Uc = Uc;
        current_params.Vc = Vc;
        current_params.xc = xc;
        current_params.yc = yc;
        current_params.meta = meta;
        current_params.dc = sqrt((xc(2) - xc(1))^2 + (yc(2) - yc(1))^2);

        for i = 1:num_obstacles
            env_params.obstacles(i).center_idx = env_params.obstacles(i).center;
            env_params.obstacles(i).radius_idx = env_params.obstacles(i).radius;
            c = env_params.obstacles(i).center;
            env_params.obstacles(i).center = [xc(c(1)); yc(c(2))];
            env_params.obstacles(i).radius = env_params.obstacles(i).radius * current_params.dc;
        end

        env_params.x_limits = [xc(x_min), xc(x_max)];         % Environment boundaries (m)
        env_params.y_limits = [yc(y_min), yc(y_max)];         % Environment boundaries (m)
        dc_scale = current_params.dc*0.75;
    end
    
    current_params.num_ensemble_members = num_ensemble_members;
    current_params.num_ensemble_members_test = floor(num_ensemble_members*0.25);
    current_params.noise_level = noise_level;         % Standard deviation of noise added to current estimate
    current_params.gradient_noise_level = 0; 

    % --- Agent Parameters ---
    agent_params.num_agents = num_agents;
    agent_params.radius = 1.5*dc_scale; 
    agent_params.safety_margin = 0.1*current_params.dc;
    agent_params.max_speed = 1;
    agent_params.collision_weight = 0*1e2;
    agent_params.color = lines(num_agents); % Assign distinct colors

    % --- Formation Parameters ---
    if strcmp(formation_type, 'circular')
        agent_params.formation_inter_agent_distance = 20.0*dc_scale;
        agent_params.formation_tolerance = 1e-6;
        agent_params.formation_weight = 0.5*1e-2/current_params.dc^2;
        radius = agent_params.formation_inter_agent_distance / 2;
        % Arrange agents on a circle of diameter d at equal angular distances
        angles = linspace(0, 2*pi, num_agents+1);
        angles(end) = []; % remove duplicated endpoint
        agent_params.formation_relative_positions = [radius * cos(angles); radius * sin(angles)];
        % Normalize relative positions so the mean is [0;0] if not already centered
        agent_params.formation_relative_positions = agent_params.formation_relative_positions - mean(agent_params.formation_relative_positions, 2);
    elseif strcmp(formation_type, 'group')
        % custom group wise circular formation
        % two groups of agents, the groups exchange their positions
    elseif strcmp(formation_type, 'ral')
        % ral formation
    end

    % --- Video Export Parameters ---
    video_params.enabled = false;                    % Enable/disable video export
    video_params.save_figure = false;                 % Enable/disable figure saving
    video_params.format = 'Motion JPEG AVI';        % Video format (fallback: auto-detect best available)
    video_params.quality = 95;                      % Video quality (0-100)
    video_params.framerate = 30;                    % Output video framerate

end



% ---- Helper function to generate random obstacles within environment ----
function obstacles = generate_random_obstacles(num, xlim, ylim, random_seed)
    min_radius = 5;  % Minimum possible obstacle radius
    max_radius = 10; % Maximum possible obstacle radius

    rng(random_seed, "philox");

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