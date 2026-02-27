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

    % Status row (always first — makes failures immediately visible)
    fprintf('| %-25s ', 'Status');
    for i = 1:length(algorithms_to_run)
        algo = algorithms_to_run{i};
        if all_metrics.(algo).success
            fprintf('| %18s ', 'OK');
        else
            fprintf('| %18s ', 'FAILED');
        end
    end
    fprintf('|\n');

    fprintf('%s%s\n', repmat('-', 1, 29), repmat('-', 1, 21*length(algorithms_to_run)));

    print_row(all_metrics, algorithms_to_run, 'Energy (Training)',    'training_energy',                         'float');
    print_row(all_metrics, algorithms_to_run, 'Energy (Testing)',     'testing_energy',                          'float');
    print_row(all_metrics, algorithms_to_run, 'Control Viol (Train)', 'training_control_constraint_violations',  'int');
    print_row(all_metrics, algorithms_to_run, 'Control Viol (Test)',  'testing_control_constraint_violations',   'int');
    print_row(all_metrics, algorithms_to_run, 'Formation Violations', 'formation_constraint_violations',         'float');

    fprintf('%s%s\n', repmat('-', 1, 29), repmat('-', 1, 21*length(algorithms_to_run)));

    print_row(all_metrics, algorithms_to_run, 'Constraint Violations','training_constraint_violations',          'int');
    print_row(all_metrics, algorithms_to_run, 'Formulation Time (s)', 'formulation_time',                        'float');
    print_row(all_metrics, algorithms_to_run, 'Training Time (s)',    'training_time',                           'float');

    fprintf('%s%s\n', repmat('=', 1, 29), repmat('=', 1, 21*length(algorithms_to_run)));
end

% -------------------------------------------------------------------------
function print_row(all_metrics, algorithms_to_run, label, field, fmt)
    fprintf('| %-25s ', label);
    for i = 1:length(algorithms_to_run)
        algo = algorithms_to_run{i};
        if all_metrics.(algo).success
            if strcmp(fmt, 'int')
                fprintf('| %18d ', all_metrics.(algo).(field));
            else
                fprintf('| %18.4f ', all_metrics.(algo).(field));
            end
        else
            fprintf('| %18s ', 'N/A');
        end
    end
    fprintf('|\n');
end
