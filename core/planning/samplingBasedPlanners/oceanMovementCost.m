function cost = oceanMovementCost(pose1, pose2, dt, opts)
    % movement cost considers the direction of ocean currents
    
    U = (opts.U); V = (opts.V);
    dt = dt*0.5;

    try
        u = U(pose1(1), pose1(2)); v = V(pose1(1), pose1(2));
    catch
        keyboard
        u = 0; v = 0;
    end
    try
        cost = norm(([diff(opts.Y_lim(1:2,1)) diff(opts.X_lim(1,1:2))].*(pose2(1:2)-pose1(1:2))-dt*1*[v u])/dt)^2;% - 10*(pose2(1:2)-pose1(1:2))*[v u]'/dt;
    catch
        keyboard
    end

    % if sqrt(cost)>opts.v
    %     cost = cost+1e6;
    % end

    % cost = (pose2(1:2)-pose1(1:2))*[v u]';









end