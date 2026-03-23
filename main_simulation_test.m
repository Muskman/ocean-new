% main_simulation_compare.m  —  single-run interactive entry point
clear; clc; close all;

opts.env_type             = 'real'; % 'sim' or 'real'
opts.current_type         = 'static'; % 'static' or 'time_varying'
opts.num_agents           = 4;
opts.num_obstacles        = 0;
opts.num_ensemble_members = 100;
opts.noise_level          = 0.2;
opts.formation_enabled    = true;
opts.formation_type       = 'circular'; % 'circular' or 'group' or 'ral'
opts.algorithms           = {'astar', 'ssca', 'dssca', 'fullOpt'}; % {'astar','fullOpt','ssca','dssca','stomp','eesto'};
opts.initial_guess        = 'straightline'; % 'straightline' or 'astar'
opts.num_mc_simulations   = 1;
opts.random_seed          = randi(1000); % 752, 518, 953

date                 = datestr(now, 'yyyy-mm-dd');
timestamp            = datestr(now, 'HH-MM-SS');

[sim_params, env_params, current_params, agent_params, video_params] = ...
    simulation_config(opts);
video_params.run_date       = date;
video_params.run_timestamp  = timestamp;
video_params.enabled        = false;
video_params.save_figure    = false;


all_metrics = run_simulation_case(sim_params, env_params, current_params, agent_params, video_params);
