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

% --- Sweep definitions ---
agent_sweep          = [3,4,5,6];       % [2, 4, 6, 8, 10]
noise_sweep          = [0.2, 0.4];
num_ensemble_members = 50;           % fixed for this experiment
formation_enabled    = true;

% Single timestamp shared across the whole batch
run_timestamp = datestr(now, 'HH-MM-SS');
run_date      = datestr(now, 'yyyy-mm-dd');

fprintf('Batch experiment started: %s %s\n', run_date, run_timestamp);
fprintf('Sweeping %d agent counts x %d noise levels = %d total cases\n\n', ...
    length(agent_sweep), length(noise_sweep), ...
    length(agent_sweep) * length(noise_sweep));

% results_table{ai, ni} holds all_metrics for that (num_agents, noise_level) case
results_table = cell(length(agent_sweep), length(noise_sweep));

% --- Outer batch loop ---
for ai = 1:length(agent_sweep)
    for ni = 1:length(noise_sweep)
        n_agents    = agent_sweep(ai);
        noise_level = noise_sweep(ni);

        fprintf('\n%s\n', repmat('#', 1, 70));
        fprintf('CASE [%d/%d]: num_agents = %d  |  noise_level = %.2f\n', ...
            (ai-1)*length(noise_sweep) + ni, ...
            length(agent_sweep)*length(noise_sweep), ...
            n_agents, noise_level);
        fprintf('%s\n', repmat('#', 1, 70));

        % Load config for this case
        [sim_params, env_params, current_params, agent_params, video_params] = ...
            simulation_config(n_agents, num_ensemble_members, noise_level, formation_enabled);

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
save(save_path, 'results_table', 'agent_sweep', 'noise_sweep', ...
     'num_ensemble_members', 'run_timestamp', 'run_date', 'formation_enabled', 'algorithms');

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('All %d cases complete.\n', length(agent_sweep) * length(noise_sweep));
fprintf('Results saved to: %s\n', save_path);
fprintf('%s\n', repmat('=', 1, 70));

% --- Print aggregate summary across all cases ---
print_batch_summary(results_table, agent_sweep, noise_sweep, sim_params.algo);
