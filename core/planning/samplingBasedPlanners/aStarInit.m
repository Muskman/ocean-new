function [initPath, map] = aStarInit(agents, env_params, current_params, sim_params, agent_params)
    % finds initial guess using astar grid search in an occupancy map for environment 
    % agents, env_params, sim_params, agent_params, current_params

    nx_contour = 500; ny_contour = 500;
    x_vec_contour = linspace(env_params.x_limits(1), env_params.x_limits(2), nx_contour);
    y_vec_contour = linspace(env_params.y_limits(1), env_params.y_limits(2), ny_contour);
    [X_grid_contour, Y_grid_contour] = meshgrid(x_vec_contour, y_vec_contour);

    contour_positions = [X_grid_contour(:)'; Y_grid_contour(:)']; % Create 2xN matrix
    [U_grid_contour_flat, V_grid_contour_flat] = calculate_ocean_current_vectorized(contour_positions, 0, current_params);
    Current_Mag_flat = sqrt(U_grid_contour_flat.^2 + V_grid_contour_flat.^2);
    % Reshape back to grid format
    Current_Mag = reshape(Current_Mag_flat, size(X_grid_contour));
    U = reshape(U_grid_contour_flat, size(X_grid_contour));
    V = reshape(V_grid_contour_flat, size(Y_grid_contour));

    dt = sim_params.dt;

    opts.U = U;
    opts.V = V;
    opts.X_lim = X_grid_contour;
    opts.Y_lim = Y_grid_contour;
    
    inflate = (agent_params.safety_margin + agent_params.radius)*4/current_params.dc;

    mag = Current_Mag;
    % X_lim = env_params.x_limits; Y_lim = env_params.y_limits; 
    x_obs = cat(2,env_params.obstacles.center);
    r_obs = cat(2,env_params.obstacles.radius); r_a = agent_params.radius;
    n_obs = length(env_params.obstacles); n_agents = length(agents);
    map = zeros(size(mag));
    [y_lim_idx, x_lim_idx] = size(map);

    for i = 1:x_lim_idx
        for j = 1:y_lim_idx
            for k = 1:n_obs
                if norm([opts.X_lim(1,i);opts.Y_lim(j,1)]-x_obs(:,k)) <= r_obs(k)+r_a
                    map(j,i) = 1;
                end
            end
        end
    end

%     for i = 1:opts.n_agents
%         x_start = opts.x_start(2*i-1:2*i); x_goal = opts.x_goal(2*i-1:2*i);

scenario = robotScenario(UpdateRate=1,StopTime=10);    

addMesh(scenario,"Plane", Size=[5 5], Position = [0 0 0], Color=[0.7 0.7 0.7]);
r_obs_c = r_obs/(opts.X_lim(1,2)-opts.X_lim(1,1));
for k = 1:n_obs
    x_obs_c = [find(abs(opts.X_lim(1,:)-x_obs(1,k))==min(abs(opts.X_lim(1,:)-x_obs(1,k)))); ...
               find(abs(opts.Y_lim(:,1)-x_obs(2,k))==min(abs(opts.Y_lim(:,1)-x_obs(2,k))))];
    addMesh(scenario,"Cylinder",Size = [r_obs_c(k) 1],Position=[x_obs_c' 0],IsBinaryOccupied=true);
end
% show3D(scenario);


%occupancyMap = binaryOccupancyMap(scenario,MapHeightLimits=[-0.1 0.1], ...
%                                    GridOriginInLocal=[X_lim(1,1) Y_lim(1,1)],MapSize=[X_lim(1,end)-X_lim(1,1),Y_lim(end,1)-Y_lim(1,1)],MapResolution=1);

occupancyMap = binaryOccupancyMap(scenario,MapHeightLimits=[-0.1 0.1], ...
                                    GridOriginInLocal=[0 0],MapSize=[size(opts.X_lim,2)-1,size(opts.Y_lim,1)-1],MapResolution=1);



%% AStar
figure(20)
show(occupancyMap)
hold on
a = [size(opts.X_lim,2),size(opts.Y_lim,1)];
temp_x_lim = meshgrid(1:4:a(1),1:4:a(2)); temp_y_lim = meshgrid(1:4:a(2),1:4:a(1))';
% quiver(opts.X_lim(1:4:end,1:4:end),opts.Y_lim(1:4:end,1:4:end),opts.U(1:4:end,1:4:end),opts.V(1:4:end,1:4:end),'k','LineWidth',0.8)
quiver(temp_x_lim,temp_y_lim,U(1:4:end,1:4:end),V(1:4:end,1:4:end),'k','LineWidth',0.8)

if ~sim_params.formation_enabled
    mPath = cell(n_agents,1);
    occupancyMap.setOccupancy(flipud(occupancyMap.getOccupancy))
    occupancyMap.inflate(inflate)
    for i = 1:n_agents
        planner = plannerAStarGrid(occupancyMap);
        planner.GCostFcn = @(pose1,pose2)oceanMovementCost(pose1,pose2,dt,opts);
        planner.HCostFcn = @(pose1,pose2) 0;

        % fprintf('For real environments, start and goal is not set to grid points. Will throw error if run.')

        x_start_c = find(abs(opts.X_lim(1,:) - agents(i).start(1))==min(abs(opts.X_lim(1,:) - agents(i).start(1))));
        y_start_c = find(abs(opts.Y_lim(:,1) - agents(i).start(2))==min(abs(opts.Y_lim(:,1) - agents(i).start(2))));

        x_goal_c = find(abs(opts.X_lim(1,:) - agents(i).goal(1))==min(abs(opts.X_lim(1,:) - agents(i).goal(1))));
        y_goal_c = find(abs(opts.Y_lim(:,1) - agents(i).goal(2))==min(abs(opts.Y_lim(:,1) - agents(i).goal(2))));

        start = fliplr([x_start_c y_start_c]); 
        goal = fliplr([x_goal_c y_goal_c]);  
        path = plan(planner, start, goal);

        l = size(path,1); 
        T = sim_params.planning_horizon;
        if l > T+1
            idx = (l-2)/(T-1);
            idx = [2, idx:idx:l-1];
            idx = idx(1:T-1); % Keep T-1 rows in between
            idx = round([1, idx, l]);
            mPath{i} = path(idx, [2 1]);
            mPath{i} = [opts.X_lim(1, mPath{i}(:,1))' opts.Y_lim(mPath{i}(:,2), 1)]';
        else
            % If the path is too short, generate a straight line with 201 waypoints between start and goal in continuous space.
            % Obtain agent's real start and goal positions
            xy_start = agents(i).start(:);
            xy_goal = agents(i).goal(:);
            num_waypoints = T+1;
            line_path = [linspace(xy_start(1), xy_goal(1), num_waypoints); ...
                        linspace(xy_start(2), xy_goal(2), num_waypoints)];
            mPath{i} = line_path;
        end
        plot(path(:,2), path(:,1),'r','LineWidth',2,'LineStyle','-.')

        plot(start(2),start(1),'g.','MarkerSize',20)
        plot(goal(2),goal(1),'r.','MarkerSize',20)
    end

    hold off

    initPath = cat(1,mPath{:});
    initPath(:,1) = cat(1,agents.start);
    initPath(:,end) = cat(1,agents.goal);
    fprintf('Generated initial path for %d agents with %d waypoints\n', n_agents, size(initPath,2))
else
    start_centroid = mean(cat(2,agents.start), 2);
    goal_centroid = mean(cat(2,agents.goal), 2);
    
    occupancyMap.setOccupancy(flipud(occupancyMap.getOccupancy))
    occupancyMap.inflate(inflate+agent_params.formation_inter_agent_distance*3/current_params.dc)

    planner = plannerAStarGrid(occupancyMap);
    planner.GCostFcn = @(pose1,pose2)oceanMovementCost(pose1,pose2,dt,opts);
    planner.HCostFcn = @(pose1,pose2) 0;

    % fprintf('For real environments, start and goal is not set to grid points. Will throw error if run.')

    x_start_c = find(abs(opts.X_lim(1,:) - start_centroid(1))==min(abs(opts.X_lim(1,:) - start_centroid(1))));
    y_start_c = find(abs(opts.Y_lim(:,1) - start_centroid(2))==min(abs(opts.Y_lim(:,1) - start_centroid(2))));

    x_goal_c = find(abs(opts.X_lim(1,:) - goal_centroid(1))==min(abs(opts.X_lim(1,:) - goal_centroid(1))));
    y_goal_c = find(abs(opts.Y_lim(:,1) - goal_centroid(2))==min(abs(opts.Y_lim(:,1) - goal_centroid(2))));

    start = fliplr([x_start_c y_start_c]); 
    goal = fliplr([x_goal_c y_goal_c]);  
    path = plan(planner, start, goal);

    l = size(path,1); 
    T = sim_params.planning_horizon;
    if l > T+1
        idx = (l-2)/(T-1);
        idx = [2, idx:idx:l-1];
        idx = idx(1:T-1); % Keep T-1 rows in between
        idx = round([1, idx, l]);
        mPath = path(idx, [2 1]);
        mPath = [opts.X_lim(1, mPath(:,1))' opts.Y_lim(mPath(:,2), 1)]';
    else
        % If the path is too short, generate a straight line with 201 waypoints between start and goal in continuous space.
        % Obtain agent's real start and goal positions
        xy_start = start_centroid;
        xy_goal = goal_centroid;
        num_waypoints = T+1;
        line_path = [linspace(xy_start(1), xy_goal(1), num_waypoints); ...
                    linspace(xy_start(2), xy_goal(2), num_waypoints)];
        mPath = line_path;
    end
    plot(path(:,2), path(:,1),'r','LineWidth',2,'LineStyle','-.')

    plot(start(2),start(1),'g.','MarkerSize',20)
    plot(goal(2),goal(1),'r.','MarkerSize',20)

    hold off

    mPath_cell = cell(n_agents,1);
    for i = 1:n_agents
        mPath_cell{i} = mPath + agent_params.formation_relative_positions(:,i);
    end

    initPath = cat(1,mPath_cell{:});
    initPath(:,1) = cat(1,agents.start);
    initPath(:,end) = cat(1,agents.goal);
    fprintf('Generated initial path for %d agents in formation with %d waypoints\n', n_agents, size(initPath,2))
end

% figure
% show(planner)
% hold on;
% quiver(opts.X_lim(1:4:end,1:4:end),opts.Y_lim(1:4:end,1:4:end),opts.U(1:4:end,1:4:end),opts.V(1:4:end,1:4:end),'k','LineWidth',0.8)

%{
%% RRT Star
figure
show(occupancyMap)
hold on

bounds = [occupancyMap.XWorldLimits; occupancyMap.YWorldLimits; [-pi pi]];
ss = stateSpaceDubins(bounds);
ss.MinTurningRadius = 0.01;

stateValidator = validatorOccupancyMap(ss); 
stateValidator.Map = occupancyMap;
stateValidator.ValidationDistance = 1;


for i = 1:2
    planner = plannerRRTStar(ss,stateValidator);
    % planner = plannerHybridAStar(ss,stateValidator);
    planner.MaxConnectionDistance = 10;
    planner.MaxIterations = 30000;
    planner.GoalReachedFcn = @exampleHelperCheckIfGoal;

    % planner = plannerAStarGrid(occupancyMap);
    % planner.GCostFcn = @(pose1,pose2)norm(pose1-pose2);
    % start = [x_start(2*i-1) x_start(2*i)]; goal = [x_goal(2*i-1) x_goal(2*i)];
    % plan(planner,fliplr(start),fliplr(goal));

    start = [x_start(2*i-1) x_start(2*i) 0]; goal = [x_goal(2*i-1),x_goal(2*i) 0];  
    [pthObj, solnInfo] = plan(planner,start,goal);
    

    % Plot entire search tree.
    plot(solnInfo.TreeData(:,1),solnInfo.TreeData(:,2),'.-');

    % Interpolate and plot path.
    interpolate(pthObj,300)
    plot(pthObj.States(:,1),pthObj.States(:,2),'r-','LineWidth',2)

    % Show start and goal in grid map.
    plot(start(1),start(2),'ro')
    plot(goal(1),goal(2),'mo')

    % show(planner)
end


%% Hybrid AStar
figure
show(occupancyMap)
hold on

bounds = [occupancyMap.XWorldLimits; occupancyMap.YWorldLimits; [-pi pi]];
ss = stateSpaceSE2;
ss.StateBounds = bounds;
% ss.MinTurningRadius = 0.01;

stateValidator = validatorOccupancyMap(ss); 
stateValidator.Map = occupancyMap;
% stateValidator.ValidationDistance = 1;


% for i = 1:1
    planner = plannerHybridAStar(stateValidator,MinTurningRadius=2, MotionPrimitiveLength=2);
    start = [x_start(2*i-1) x_start(2*i) 0]; goal = [x_goal(2*i-1), x_goal(2*i) 0];  
    
    refpath = plan(planner,start,goal);     

    show(planner); hold on

    % Show start and goal in grid map.
    plot(start(1),start(2),'g.','MarkerSize',20)
    plot(goal(1),goal(2),'r.','MarkerSize',20)
% end

%}
figure(20)
close

end