function [Uc, Vc, xc, yc, meta] = learn_ocean_currents_real(current_params)
    %LEARN_OCEAN_CURRENTS_NEW Build NaN-safe current maps as Uc(p), Vc(p).
    %
    %   [Uc, Vc, meta] = learn_ocean_currents_new(current_params)
    %   [Uc, Vc, meta, xc, yc] = learn_ocean_currents_new(current_params)
    %
    %   Learns a static current field from gridded U/V samples and exposes it as
    %   function handles of positions p (similar to calculate_ocean_current_vectorized):
    %
    %     - p is either 2xN ([x; y]) or Nx2 ([x y])
    %     - Uc(p), Vc(p) return 1xN row vectors
    %     - xc(i) maps continuous i in [1, nx] to X in [X_lim(1,1), X_lim(1,end)]
    %     - yc(i) maps continuous i in [1, ny] to Y in [Y_lim(1,1), Y_lim(end,1)]
    %
    %   Required fields in current_params:
    %     - X_lim (ny x nx), Y_lim (ny x nx), U (ny x nx), V (ny x nx)
    %
    %   Behavior:
    %     - NaNs in U/V are ignored during fitting with nearest fallback.
    %     - Queries outside the rectangular domain return 0.
    
    Xg = current_params.X_lim;
    Yg = current_params.Y_lim;
    Ug = double(current_params.U);
    Vg = double(current_params.V);
    
    if ~isequal(size(Xg), size(Yg), size(Ug), size(Vg))
        error('learn_ocean_currents_new:SizeMismatch', ...
            'X_lim, Y_lim, U, V must be the same size.');
    end
    
    % Domain (supports increasing or decreasing axes).
    x1 = Xg(1,1); x2 = Xg(1,end);
    y1 = Yg(1,1); y2 = Yg(end,1);
    xMin = min(x1, x2); xMax = max(x1, x2);
    yMin = min(y1, y2); yMax = max(y1, y2);
    
    ny = size(Xg, 1);
    nx = size(Xg, 2);
    
    % Continuous index -> coordinate maps (preserve direction in X_lim/Y_lim endpoints)
    xc = @(i) local_linmap(i, 1, nx, x1, x2);
    yc = @(i) local_linmap(i, 1, ny, y1, y2);
    
    X = Xg(:);
    Y = Yg(:);
    
    uMask = isfinite(X) & isfinite(Y) & isfinite(Ug(:));
    vMask = isfinite(X) & isfinite(Y) & isfinite(Vg(:));
    
    meta = struct();
    meta.xRange = [xMin, xMax];
    meta.yRange = [yMin, yMax];
    meta.numGridPoints = numel(Xg);
    meta.numUPoints = nnz(uMask);
    meta.numVPoints = nnz(vMask);
    meta.fracUNaN = 1 - meta.numUPoints / max(1, meta.numGridPoints);
    meta.fracVNaN = 1 - meta.numVPoints / max(1, meta.numGridPoints);
    meta.interface = 'Uc(p), Vc(p) with p in R^{2xN} or R^{Nx2}; outputs 1xN';
    meta.interpMethod = 'scatteredInterpolant(natural) + nearest fallback; 0 outside domain';
    meta.xc = xc;
    meta.yc = yc;
    meta.nx = nx;
    meta.ny = ny;
    
    if meta.numUPoints < 3
        F_u_lin = [];
        F_u_nn = [];
    else
        F_u_lin = scatteredInterpolant(X(uMask), Y(uMask), Ug(uMask), 'natural', 'none');
        F_u_nn  = scatteredInterpolant(X(uMask), Y(uMask), Ug(uMask), 'nearest', 'nearest');
    end
    
    if meta.numVPoints < 3
        F_v_lin = [];
        F_v_nn = [];
    else
        F_v_lin = scatteredInterpolant(X(vMask), Y(vMask), Vg(vMask), 'natural', 'none');
        F_v_nn  = scatteredInterpolant(X(vMask), Y(vMask), Vg(vMask), 'nearest', 'nearest');
    end
    
    Uc = @(p) local_eval_p(F_u_lin, F_u_nn, p, xMin, xMax, yMin, yMax);
    Vc = @(p) local_eval_p(F_v_lin, F_v_nn, p, xMin, xMax, yMin, yMax);

    % --- Provide rectilinear, NaN-free grids for CasADi LUT construction ---
    % create_symbolic_ocean_func_real needs numeric LUTs (CasADi cannot use MATLAB
    % function handles symbolically). If the input grid is rectilinear, we can
    % export a filled version of U/V on monotonic (increasing) x/y vectors.
    [is_rect, xgrid_rect, ygrid_rect] = local_rectilinear_axes(Xg, Yg);
    meta.is_rectilinear = is_rect;
    if is_rect
        % Ensure increasing axes (match CasADi interpolant requirements)
        if xgrid_rect(1) > xgrid_rect(end); xgrid_rect = fliplr(xgrid_rect); end
        if ygrid_rect(1) > ygrid_rect(end); ygrid_rect = fliplr(ygrid_rect); end

        [Xr, Yr] = meshgrid(xgrid_rect, ygrid_rect);
        meta.xgrid_rect = xgrid_rect;
        meta.ygrid_rect = ygrid_rect;

        meta.U_filled_rect = local_eval_xy(F_u_lin, F_u_nn, Xr, Yr, xMin, xMax, yMin, yMax);
        meta.V_filled_rect = local_eval_xy(F_v_lin, F_v_nn, Xr, Yr, xMin, xMax, yMin, yMax);
    end
end
    
function z = local_eval_p(F_lin, F_nn, p, xMin, xMax, yMin, yMax)
    if isempty(p)
        z = zeros(1, 0);
        return;
    end
    
    if isvector(p) && numel(p) == 2
        % Single point as [x;y] or [x y]
        p = p(:);
    end
    
    if size(p, 1) == 2
        x = p(1, :);
        y = p(2, :);
    elseif size(p, 2) == 2
        x = p(:, 1).';
        y = p(:, 2).';
    else
        error('learn_ocean_currents_new:BadPositionShape', ...
            'p must be 2xN ([x;y]) or Nx2 ([x y]).');
    end
    
    z = local_eval_xy(F_lin, F_nn, x, y, xMin, xMax, yMin, yMax);
    z = reshape(z, 1, []);
end
    
function z = local_eval_xy(F_lin, F_nn, x, y, xMin, xMax, yMin, yMax)
    % Vectorized evaluation for row-vectors x,y
    xq = x(:);
    yq = y(:);
    
    zq = zeros(size(xq));
    inside = isfinite(xq) & isfinite(yq) & (xq >= xMin) & (xq <= xMax) & (yq >= yMin) & (yq <= yMax);
    if any(inside)
        if isempty(F_lin)
            zInside = zeros(nnz(inside),1);
        else
            zInside = F_lin(xq(inside), yq(inside));
            bad = isnan(zInside);
            if any(bad)
                if isempty(F_nn)
                    zInside(bad) = 0;
                else
                    idxInside = find(inside);
                    idxBad = idxInside(bad);
                    zInside(bad) = F_nn(xq(idxBad), yq(idxBad));
                    zInside(isnan(zInside)) = 0;
                end
            end
        end
        zq(inside) = zInside;
    end
    
    z = reshape(zq, size(x));
end
    
function out = local_linmap(i, i1, i2, v1, v2)
    % Linear map i in [i1,i2] -> v in [v1,v2]
    if i2 == i1
        out = v1 .* ones(size(i));
    else
        out = v1 + (i - i1) .* (v2 - v1) ./ (i2 - i1);
    end
end

function [is_rect, xgrid, ygrid] = local_rectilinear_axes(Xg, Yg)
    % Check if Xg/Yg represent a rectilinear meshgrid.
    % If so, return the axis vectors (not forced monotonic).
    xgrid = Xg(1, :);
    ygrid = Yg(:, 1).';
    is_rect = true;
    try
        % X should be constant across rows, Y constant across columns
        is_rect = is_rect && all(all(abs(Xg - repmat(xgrid, size(Xg, 1), 1)) < 1e-9 | (~isfinite(Xg) & ~isfinite(repmat(xgrid, size(Xg, 1), 1)))));
        is_rect = is_rect && all(all(abs(Yg - repmat(ygrid.', 1, size(Yg, 2))) < 1e-9 | (~isfinite(Yg) & ~isfinite(repmat(ygrid.', 1, size(Yg, 2))))));
    catch
        is_rect = false;
    end
    if ~is_rect
        xgrid = [];
        ygrid = [];
    end
end
    
    