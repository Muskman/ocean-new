function [sim_params, env_params, current_params, agent_params, num_agents, video_params] = simulation_config()
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
    %   num_agents    - Number of agents
    %   video_params  - Video export parameters structure
    
    fprintf('Setting up simulation parameters...\n');
    
    % --- Simulation Parameters ---
    sim_params.dt = 0.5;                       % Simulation time step (s)
    sim_params.T_final = 200;                 % Total simulation time (s)
    sim_params.time_steps = floor(sim_params.T_final / sim_params.dt);
    sim_params.visualization = true;         % Enable/disable visualization
    sim_params.vis_interval = 5;             % Update visualization every N steps (adjust as needed)
    sim_params.vis_vector_scale = 1;         % Scaling factor for velocity vector visualization
    sim_params.formation_enabled = true;     % toggle formation control
    sim_params.planning_horizon = sim_params.T_final/sim_params.dt;       % Number of steps planner looks ahead
    sim_params.replan_interval = sim_params.T_final/sim_params.dt;        % Replan interval (adjust as needed)
    
    % --- algorithm related parameters ---
    sim_params.algo = {'fullOpt', 'ssca'};               % Planning algorithm: 'fullOpt', 'sca', 'ssca'
    if any(strcmp(sim_params.algo, 'ssca')) 
        sim_params.max_outer_iterations = 10;
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
    
    % Example obstacles
    % env_params.obstacles(1) = struc  t('center', [15; 15], 'radius', 15);
    % env_params.obstacles(2) = struct('center', [-20; 15], 'radius', 7);
    % env_params.obstacles(1) = struct('center', [15; 30], 'radius', 10);
 
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
    
    total_num_ensemble_members = 1;
    current_params.num_ensemble_members = ceil(total_num_ensemble_members*0.8);
    current_params.num_ensemble_members_test = floor(total_num_ensemble_members*0.2);
    current_params.noise_level = 0.2;         % Standard deviation of noise added to current estimate
    current_params.gradient_noise_level = 0; 

    % --- Agent Parameters ---
    num_agents = 1;
    agent_params.radius = 1.5; 
    agent_params.max_speed = 5.0;
    agent_params.safety_margin = 0.2;
    agent_params.color = lines(num_agents); % Assign distinct colors

    % --- Formation Parameters ---
    agent_params.formation_inter_agent_distance = 10.0;
    agent_params.formation_tolerance = 1e-6;
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
        for i = 1:num_agents; positions(2, i) = (i - (num_agents+1)/2) * d; end
        agent_params.formation_relative_positions = positions;
    end
    % Normalize relative positions so the mean is [0;0] if not already centered
    agent_params.formation_relative_positions = agent_params.formation_relative_positions - mean(agent_params.formation_relative_positions, 2);

    % --- Video Export Parameters ---
    video_params.enabled = false;                    % Enable/disable video export
    video_params.format = 'Motion JPEG AVI';        % Video format (fallback: auto-detect best available)
    video_params.quality = 95;                      % Video quality (0-100)
    video_params.framerate = 30;                    % Output video framerate

end
