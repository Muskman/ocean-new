function print_metrics(all_metrics, algorithms_to_run)
    % --- Print Comparison Metrics ---
    total_width = 29 + (21 * length(algorithms_to_run));
    banner_text = 'ALGORITHM COMPARISON METRICS';
    banner_len = length(banner_text);

    fprintf('\n%s\n', repmat('=', 1, total_width));
    padding = floor((total_width - banner_len) / 2);
    fprintf('%s%s%s\n', repmat(' ', 1, padding), banner_text, repmat(' ', 1, total_width - padding - banner_len));
    fprintf('%s\n', repmat('=', 1, total_width));

    % Create header with algorithm names
    header_format = '| %-25s ';
    for i = 1:length(algorithms_to_run)
        header_format = [header_format, '| %-18s '];
    end
    header_format = [header_format, '|\n'];

    fprintf(header_format, 'Metric', algorithms_to_run{:});
    fprintf('|%s%s%s|\n', repmat('-', 1, 27), repmat('+', 1, 1), repmat('-', 1, 21*length(algorithms_to_run)-1));

    % Success (%) row across Monte Carlo runs
    fprintf('| %-25s ', 'Success (%)');
    for i = 1:length(algorithms_to_run)
        algo = algorithms_to_run{i};
        runs = all_metrics.(algo);
        if isempty(runs)
            fprintf('| %18s ', 'N/A');
            continue;
        end

        [success_rate, ok] = mc_success_rate(runs);
        if isempty(ok)
            fprintf('| %18s ', 'N/A');
            continue;
        end

        fprintf('| %18.1f ', success_rate);
    end
    fprintf('|\n');

    fprintf('%s%s\n', repmat('-', 1, 29), repmat('-', 1, 21*length(algorithms_to_run)));

    print_row_mc(all_metrics, algorithms_to_run, 'Energy (Training)',    'training_energy',                        'float');
    print_row_mc(all_metrics, algorithms_to_run, 'Energy (Testing)',     'testing_energy',                         'float');
    print_row_mc(all_metrics, algorithms_to_run, 'Control Viol (Train)', 'training_control_constraint_violations', 'int');
    print_row_mc(all_metrics, algorithms_to_run, 'Control Viol (Test)',  'testing_control_constraint_violations',  'int');
    print_row_mc(all_metrics, algorithms_to_run, 'Formation Violations', 'formation_constraint_violations',        'float');

    fprintf('%s%s\n', repmat('-', 1, 29), repmat('-', 1, 21*length(algorithms_to_run)));

    print_row_mc(all_metrics, algorithms_to_run, 'Constraint Violations', 'training_constraint_violations',         'int');
    print_row_mc(all_metrics, algorithms_to_run, 'Formulation Time (s)',  'formulation_time',                       'float');
    print_row_mc(all_metrics, algorithms_to_run, 'Training Time (s)',     'training_time',                          'float');

    fprintf('%s%s\n', repmat('=', 1, 29), repmat('=', 1, 21*length(algorithms_to_run)));
end

% -------------------------------------------------------------------------
function print_row_mc(all_metrics, algorithms_to_run, label, field, fmt)
    fprintf('| %-25s ', label);
    for i = 1:length(algorithms_to_run)
        algo = algorithms_to_run{i};
        runs = all_metrics.(algo);
        if isempty(runs)
            fprintf('| %18s ', 'N/A');
            continue;
        end

        [~, ok] = mc_success_rate(runs);
        if isempty(ok) || ~any(ok)
            fprintf('| %18s ', 'N/A');
            continue;
        end

        vals = mc_extract_vals(runs, ok, field);

        if isempty(vals)
            fprintf('| %18s ', 'N/A');
            continue;
        end

        mu = mean(vals);
        sd = std(vals, 0);

        cell_str = sprintf('%.2f+/-%.2f', mu, sd);
        
        fprintf('| %18s ', cell_str);
    end
    fprintf('|\n');
end

% -------------------------------------------------------------------------
function [success_rate, ok] = mc_success_rate(runs)
    % Supports either:
    %   - cell array of structs (preferred in this project)
    %   - struct array
    %
    % Returns:
    %   success_rate: percentage in [0,100]
    %   ok: logical vector, true for successful runs

    if iscell(runs)
        n = numel(runs);
        ok = false(1, n);
        for j = 1:n
            r = runs{j};
            if isstruct(r) && isfield(r, 'success')
                ok(j) = logical(r.success);
            end
        end
    elseif isstruct(runs)
        if ~isfield(runs, 'success')
            ok = [];
            success_rate = NaN;
            return;
        end
        ok = logical([runs.success]);
    else
        ok = [];
        success_rate = NaN;
        return;
    end

    if isempty(ok)
        success_rate = NaN;
    else
        success_rate = 100 * mean(ok);
    end
end

% -------------------------------------------------------------------------
function vals = mc_extract_vals(runs, ok, field)
    % Extract numeric field values from successful MC runs.
    % Works for runs stored as cell-of-struct or struct array.

    vals = [];
    if iscell(runs)
        for j = 1:numel(runs)
            if ~ok(j); continue; end
            r = runs{j};
            if isstruct(r) && isfield(r, field)
                v = r.(field);
                if isnumeric(v) && isscalar(v)
                    vals(end+1) = v; %#ok<AGROW>
                end
            end
        end
    elseif isstruct(runs)
        for j = 1:numel(runs)
            if ~ok(j); continue; end
            if isfield(runs(j), field)
                v = runs(j).(field);
                if isnumeric(v) && isscalar(v)
                    vals(end+1) = v; %#ok<AGROW>
                end
            end
        end
    end
end
