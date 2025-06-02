% plot_environment.m
function plot_environment(ax, env_params, agents)
    hold(ax, 'on');

    obstacle_color = [0.73 0.57 0.32]; % Light gray for obstacles
    obstacle_edge_color = [0 0 0]; % Make edge black
    obstacle_line_width = 1.5;    % Make edge thicker

    % Plot obstacles using fill
    for i = 1:length(env_params.obstacles)
        center = env_params.obstacles(i).center;
        radius = env_params.obstacles(i).radius;
        [x_verts, y_verts] = create_circle_vertices(center, radius);
        fill(ax, x_verts, y_verts, obstacle_color, 'EdgeColor', obstacle_edge_color, 'LineWidth', obstacle_line_width);
%         plot(ax, center(1), center(2), 'rx'); % Optional: Mark center
    end

    % Plot agent start positions and goals (Keep as markers for clarity)
    start_marker_color = [0 0 1]; % Blue
    goal_marker_color = [0 0.7 0]; % Darker Green
    for i = 1:length(agents)
        plot(ax, agents(i).position(1), agents(i).position(2), '+', 'MarkerEdgeColor', start_marker_color, 'MarkerSize', 10, 'LineWidth', 2);
        plot(ax, agents(i).goal(1), agents(i).goal(2), 'x', 'Color', goal_marker_color, 'MarkerSize', 10, 'LineWidth', 2);
%         text(ax, agents(i).position(1)+1, agents(i).position(2), sprintf('A%d Start', i), 'Color', start_marker_color);
%         text(ax, agents(i).goal(1)+1, agents(i).goal(2), sprintf('A%d Goal', i), 'Color', goal_marker_color);
    end

    % hold(ax, 'off');
end 