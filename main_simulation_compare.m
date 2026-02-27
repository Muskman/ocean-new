% main_simulation_compare.m  —  single-run interactive entry point
clear; clc; close all;

num_agents           = 1;
num_ensemble_members = 1;
noise_level          = 0.2;
formation_enabled    = true;

date                 = datestr(now, 'yyyy-mm-dd');
timestamp            = datestr(now, 'HH-MM-SS');

[sim_params, env_params, current_params, agent_params, video_params] = ...
    simulation_config(num_agents, num_ensemble_members, noise_level, formation_enabled);
video_params.run_date       = date;
video_params.run_timestamp  = timestamp;
video_params.enabled        = true;
video_params.save_figure    = true;


all_metrics = run_simulation_case(sim_params, env_params, current_params, agent_params, video_params);
