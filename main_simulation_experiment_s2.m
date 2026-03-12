% main_simulation_experiment_s2.m
%
% Batch experiment sweeping:
%   - num_agents  : 4
%   - noise_level : 0.2
%   - formation_enabled : true
%
% All videos, figures, and results are saved under a shared time-stamped
% folder so the entire batch is grouped together on disk.
%
% Results are saved to a .mat file for later LaTeX table generation.

clear; clc; close all;

% seeds for recreating exoeriment results in text
recreate_experiment_results = false;
random_seed_all = [74, 176, 623, 359];

% --- Sweep definitions ---
agent_sweep          = 3;       % [3]
noise_level          = 0.25;
num_ensemble_members_sweep = [25, 50, 100, 200];           % fixed for this experiment
formation_enabled    = true;
algorithms           = {'astar', 'fullOpt', 'ssca', 'dssca'};
num_mc_simulations   = 5;

% Single timestamp shared across the whole batch
run_timestamp = datestr(now, 'HH-MM-SS');
run_date      = datestr(now, 'yyyy-mm-dd');

fprintf('Batch experiment started: %s %s\n', run_date, run_timestamp);
fprintf('Sweeping %d agent counts x %d ensemble member levels = %d total cases\n\n', ...
    length(agent_sweep), length(num_ensemble_members_sweep), ...
    length(agent_sweep) * length(num_ensemble_members_sweep));

% results_table{ai, ni} holds all_metrics for that (num_agents, noise_level) case
results_table = cell(length(agent_sweep), length(num_ensemble_members_sweep));

% --- Outer batch loop ---
for ai = 1:length(agent_sweep)
    if recreate_experiment_results
        random_seed = random_seed_all(ai);
    else
        random_seed = randi(1000);
    end

    for ni = 1:length(num_ensemble_members_sweep)
        n_agents    = agent_sweep(ai);
        num_ensemble_members = num_ensemble_members_sweep(ni);

        fprintf('\n%s\n', repmat('#', 1, 70));
        fprintf('CASE [%d/%d]: num_agents = %d  |  num_ensemble_members = %d\n', ...
            (ai-1)*length(num_ensemble_members_sweep) + ni, ...
            length(agent_sweep)*length(num_ensemble_members_sweep), ...
            n_agents, num_ensemble_members);
        fprintf('%s\n', repmat('#', 1, 70));

        % Load config for this case
        [sim_params, env_params, current_params, agent_params, video_params] = ...
            simulation_config(n_agents, num_ensemble_members, noise_level, formation_enabled, algorithms, num_mc_simulations, random_seed);

        % Enable video and inject shared timestamp so all files land in the
        % same date/time folder on disk
        video_params.enabled        = true;
        video_params.save_figure    = true;
        video_params.run_date       = run_date;
        video_params.run_timestamp  = run_timestamp;
        
        % Run simulation and collect metrics
        results_table{ai, ni} = run_simulation_case( ...
            sim_params, env_params, current_params, agent_params, video_params);
    end
end

% --- Save results for later analysis / LaTeX table generation ---
out_dir = fullfile('results', run_date, run_timestamp);
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

algorithms = sim_params.algo;
save_path = fullfile(out_dir, 'batch_results.mat');
save(save_path, 'results_table', 'agent_sweep', 'noise_level', ...
     'num_ensemble_members_sweep', 'run_timestamp', 'run_date', 'formation_enabled', 'algorithms');

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('All %d cases complete.\n', length(agent_sweep) * length(num_ensemble_members_sweep));
fprintf('Results saved to: %s\n', save_path);
fprintf('%s\n', repmat('=', 1, 70));

% --- Print aggregate summary across all cases ---
% print_batch_summary(results_table, agent_sweep, noise_sweep, algorithms);
