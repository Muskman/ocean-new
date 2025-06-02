% update_agent_state.m
function agent = update_agent_state(agent, current_time, dt, current_params, agent_params) % Added agent_params
    % Get true ocean current at the agent's position
    ocean_current = calculate_ocean_current(agent.position, current_time, current_params);

    % Calculate agent's velocity relative to ground (inertial frame)
    % Ground Velocity = Commanded Velocity relative to Water + Water Velocity (Current)
    ground_velocity = agent.control_velocity + ocean_current;

    % Update position using simple Euler integration
    agent.position = agent.position + ground_velocity * dt;

    % Optional: Add boundary checks or wrap-around logic if needed
    % agent.position(1) = max(env_params.x_limits(1), min(env_params.x_limits(2), agent.position(1))); % Needs env_params
    % agent.position(2) = max(env_params.y_limits(1), min(env_params.y_limits(2), agent.position(2))); % Needs env_params
    % Or pass limits through agent_params if preferred
end 