function [initPath, map] = findOMap(opts)
    % finds occupancy map for environment 

mag = opts.m ag;
X_lim = opts.X_lim; Y_lim = opts.Y_lim; 
x_obs = opts.x_obs;
r_obs = opts.r_obs; r_a = opts.r_a;
n_obs = opts.n_obs;
map = zeros(size(mag));
[y_lim_idx, x_lim_idx] = size(map);

for i = 1:x_lim_idx
    for j = 1:y_lim_idx
        for k = 1:n_obs
            if norm([X_lim(1,i);Y_lim(j,1)]-x_obs(:,k)) <= r_obs(k)+r_a
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
    x_obs_c = [find(abs(opts.X_lim(1,:)-opts.x_obs(1,k))==min(abs(opts.X_lim(1,:)-opts.x_obs(1,k)))); ...
               find(abs(opts.Y_lim(:,1)-opts.x_obs(2,k))==min(abs(opts.Y_lim(:,1)-opts.x_obs(2,k))))];
    addMesh(scenario,"Cylinder",Size = [r_obs_c(k) 1],Position=[x_obs_c' 0],IsBinaryOccupied=true);
end
% show3D(scenario);


%occupancyMap = binaryOccupancyMap(scenario,MapHeightLimits=[-0.1 0.1], ...
%                                    GridOriginInLocal=[X_lim(1,1) Y_lim(1,1)],MapSize=[X_lim(1,end)-X_lim(1,1),Y_lim(end,1)-Y_lim(1,1)],MapResolution=1);

occupancyMap = binaryOccupancyMap(scenario,MapHeightLimits=[-0.1 0.1], ...
                                    GridOriginInLocal=[0 0],MapSize=[size(X_lim,2)-1,size(Y_lim,1)-1],MapResolution=1);



%% AStar
figure(20)
show(occupancyMap)
hold on
a = [size(opts.X_lim,2),size(opts.Y_lim,1)];
temp_x_lim = meshgrid(1:4:a(1),1:4:a(2)); temp_y_lim = meshgrid(1:4:a(2),1:4:a(1))';
% quiver(opts.X_lim(1:4:end,1:4:end),opts.Y_lim(1:4:end,1:4:end),opts.U(1:4:end,1:4:end),opts.V(1:4:end,1:4:end),'k','LineWidth',0.8)
quiver(temp_x_lim,temp_y_lim,opts.U(1:4:end,1:4:end),opts.V(1:4:end,1:4:end),'k','LineWidth',0.8)

mPath = [];

for i = 1:opts.n_agents
    if i==1
        occupancyMap.setOccupancy(flipud(occupancyMap.getOccupancy))
        occupancyMap.inflate(opts.inflate)
    end
    planner = plannerAStarGrid(occupancyMap);
    planner.GCostFcn = @(pose1,pose2)oceanMovementCost(pose1,pose2,opts.dt(i),opts);
    planner.HCostFcn = @(pose1,pose2) 0;

    % fprintf('For real environments, start and goal is not set to grid points. Will throw error if run.')

    x_start_c = find(abs(opts.X_lim(1,:) - opts.x_start(2*i-1))<sqrt(eps));
    y_start_c = find(abs(opts.Y_lim(:,1) - opts.x_start(2*i))<sqrt(eps));

    x_goal_c = find(abs(opts.X_lim(1,:) - opts.x_goal(2*i-1))<sqrt(eps));
    y_goal_c = find(abs(opts.Y_lim(:,1) - opts.x_goal(2*i))<sqrt(eps));

    start = fliplr([x_start_c y_start_c]); goal = fliplr([x_goal_c y_goal_c]);  
    path = plan(planner,start,goal);

    l = size(path,1); T = opts.T;
    % if l>T
    %     idx = [2:round(l/T):l-1];
    %     if length(idx) > T-2
    %         idx(randi(length(idx),length(idx)-opts.T+2,1)) = [];
    %     end
    %     idx = [1 idx l];
    % end
    if l > T
        idx = (l-2)/(T-2);
        idx = [2,idx:idx:l-1];
        idx = idx(1:T-2); % Keep T-2 rows in between
        idx = round([1,idx,l]);
    end

    try
        mPath = [mPath; path(idx,[2 1])];
    catch
        keyboard
    end
    
    plot(path(:,2), path(:,1),'r','LineWidth',2,'LineStyle','-.')

    plot(start(2),start(1),'g.','MarkerSize',20)
    plot(goal(2),goal(1),'r.','MarkerSize',20)
end

hold off

mPath = [X_lim(1,mPath(:,1))' Y_lim(mPath(:,2),1)];

initPath = reshape(mPath',opts.n_agents*2*T,1);
fprintf('Generated initial path with %d waypoints\n', size(mPath,1)/2)

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