function [planned_trajectories, metrics] = astar_planner(agents, env_params, current_params, sim_params, agent_params)

    sim_params.initial_guess = 'astar';

    builder = ProblemBuilder(agents, env_params, current_params, sim_params, agent_params, ProblemBuilder.getDefaultConfig());
    
    planned_trajectories = cell(length(agents), 1);
    for i = 1:length(agents)
        P0 = builder.P0(2*i-1:2*i,:); % P0(:, 0.5*T+1) = [];
        planned_trajectories{i} = struct('planned_positions', P0);
    end
    builder.buildSymbolicTemplates();
    builder.buildAllConstraintsAndBounds();
    builder.buildBenchmarkingExpressions();
    metrics = builder.getBenchmarkingMetrics();
    metrics.training_time = builder.init_time;
    metrics.formulation_time = 0;
end
