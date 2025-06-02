% calculate_ocean_current.m
function current = calculate_ocean_current(position, time, current_params)
    % Calculates the ocean current at a given position (x,y) and time.
    % Uses superposition of Lamb vortices.

    x = position(1);
    y = position(2);
    current = [0; 0]; % Initialize velocity U = [u; v]

    for i = 1:length(current_params.vortices)
        vortex = current_params.vortices(i);
        x0 = vortex.center(1);
        y0 = vortex.center(2);
        Gamma = vortex.strength;
        R = vortex.core_radius;

        % TODO: Add time-varying behavior if needed (e.g., move centers, change strength)
        if strcmp(current_params.type, 'time_varying')
             % Example: Oscillating vortex center
             x0 = x0 - 5 * sin(0.1 * time);
             y0 = y0 + 3 * cos(0.7 * time); % Keep y constant or make it vary too
             Gamma = Gamma * (1 + 0.1*cos(0.5*time)); % Varying strength
        end

        r_sq = (x - x0)^2 + (y - y0)^2;

        if r_sq == 0 % Avoid division by zero at the exact center
            v_theta = 0;
        else
            % Lamb-Oseen vortex velocity profile
            v_theta = (Gamma / (2 * pi * sqrt(r_sq))) * (1 - exp(-r_sq / R^2));
        end

        % Convert polar velocity (v_theta) to Cartesian components (u, v)
        if r_sq > eps % Avoid issues at the center
            angle = atan2(y - y0, x - x0);
            u_vortex = -v_theta * sin(angle);
            v_vortex = v_theta * cos(angle);
            current = current + [u_vortex; v_vortex];
        end
    end
end 