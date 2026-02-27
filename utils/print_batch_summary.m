function print_batch_summary(results_table, agent_sweep, noise_sweep, algorithms_to_run)
    % PRINT_BATCH_SUMMARY  Aggregate statistics across all batch cases.
    %
    % For each algorithm reports:
    %   - % of successful runs
    %   - % of runs with zero violations (train + test control + constraint)
    %   - Average testing energy
    %   - Average training time per agent
    %
    % Inputs:
    %   results_table      - cell(n_agents, n_noise), each cell = all_metrics struct
    %   agent_sweep        - vector of num_agents values used in the sweep
    %   noise_sweep        - vector of noise_level values used in the sweep
    %   algorithms_to_run  - cell array of algorithm name strings

    if ~iscell(algorithms_to_run)
        algorithms_to_run = {algorithms_to_run};
    end

    n_algos      = length(algorithms_to_run);
    total_cases  = numel(results_table);

    % --- Collect per-algorithm statistics ---
    n_success        = zeros(1, n_algos);
    n_zero_viol      = zeros(1, n_algos);  % zero on all violation counts
    energy_sum       = zeros(1, n_algos);
    time_per_agent_sum = zeros(1, n_algos);
    n_valid          = zeros(1, n_algos);  % successful cases contributing to averages

    for ai = 1:length(agent_sweep)
        for ni = 1:length(noise_sweep)
            m        = results_table{ai, ni};
            n_agents = agent_sweep(ai);

            for k = 1:n_algos
                algo = algorithms_to_run{k};
                if ~isfield(m, algo); continue; end
                am = m.(algo);

                if ~am.success; continue; end

                n_success(k) = n_success(k) + 1;
                n_valid(k)   = n_valid(k) + 1;

                if am.training_control_constraint_violations == 0 && ...
                   am.testing_control_constraint_violations  == 0 && ...
                   am.training_constraint_violations         == 0
                    n_zero_viol(k) = n_zero_viol(k) + 1;
                end

                energy_sum(k)         = energy_sum(k)         + am.testing_energy;
                time_per_agent_sum(k) = time_per_agent_sum(k) + am.training_time / n_agents;
            end
        end
    end

    % --- Print ---
    col_w      = 22;
    label_w    = 32;
    total_width = label_w + 2 + col_w * n_algos + n_algos + 1;

    banner = 'BATCH SUMMARY METRICS';
    fprintf('\n%s\n', repmat('=', 1, total_width));
    pad = floor((total_width - length(banner)) / 2);
    fprintf('%s%s\n', repmat(' ', 1, pad), banner);
    fprintf('%s\n', repmat('=', 1, total_width));
    fprintf('Total cases: %d  |  Agents: [%s]  |  Noise: [%s]\n', ...
        total_cases, num2str(agent_sweep), num2str(noise_sweep, '%.1f '));
    fprintf('%s\n', repmat('-', 1, total_width));

    % Header row
    fprintf('| %-*s ', label_w, 'Metric');
    for k = 1:n_algos
        fprintf('| %-*s ', col_w, algorithms_to_run{k});
    end
    fprintf('|\n');
    fprintf('%s\n', repmat('-', 1, total_width));

    % Success rate
    fprintf('| %-*s ', label_w, 'Success Rate (%)');
    for k = 1:n_algos
        fprintf('| %*.1f ', col_w, 100 * n_success(k) / total_cases);
    end
    fprintf('|\n');

    % Zero violations rate  (% of successful runs)
    fprintf('| %-*s ', label_w, 'Zero Violations Rate (%)');
    for k = 1:n_algos
        if n_valid(k) > 0
            fprintf('| %*.1f ', col_w, 100 * n_zero_viol(k) / n_valid(k));
        else
            fprintf('| %*s ', col_w, 'N/A');
        end
    end
    fprintf('|\n');

    fprintf('%s\n', repmat('-', 1, total_width));

    % Average testing energy
    fprintf('| %-*s ', label_w, 'Avg Testing Energy');
    for k = 1:n_algos
        if n_valid(k) > 0
            fprintf('| %*.4f ', col_w, energy_sum(k) / n_valid(k));
        else
            fprintf('| %*s ', col_w, 'N/A');
        end
    end
    fprintf('|\n');

    % Average training time per agent
    fprintf('| %-*s ', label_w, 'Avg Train Time / Agent (s)');
    for k = 1:n_algos
        if n_valid(k) > 0
            fprintf('| %*.4f ', col_w, time_per_agent_sum(k) / n_valid(k));
        else
            fprintf('| %*s ', col_w, 'N/A');
        end
    end
    fprintf('|\n');

    fprintf('%s\n\n', repmat('=', 1, total_width));
end
