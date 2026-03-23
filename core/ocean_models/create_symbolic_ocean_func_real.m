% create_symbolic_ocean_func_real.m
function [ocean_func, ocean_gradient_func] = create_symbolic_ocean_func_real(current_params, use_mx)
    %CREATE_SYMBOLIC_OCEAN_FUNC_REAL CasADi current + gradient from gridded U/V.
    %
    % IMPORTANT:
    % - You cannot call MATLAB function handles (e.g., Uc/Vc) on CasADi symbolic
    %   variables directly. CasADi requires symbolic expressions.
    % - Therefore we build numeric lookup tables (LUTs) and then differentiate
    %   the LUT expression symbolically.
    %
    % Inputs (required):
    %   - current_params.X_lim (ny x nx)
    %   - current_params.Y_lim (ny x nx)
    %   - current_params.U     (ny x nx) (may contain NaN)
    %   - current_params.V     (ny x nx) (may contain NaN)
    %
    % Optional (if available, used to be consistent with learn_ocean_currents_real):
    %   - current_params.meta.U_filled_rect, meta.V_filled_rect, meta.xgrid_rect, meta.ygrid_rect
    %   - current_params.Uc, current_params.Vc (MATLAB handles; used only at build time)

    import casadi.*

    if nargin < 2
        use_mx = true;
    end

    if use_mx
        Vsym = MX;
    else
        Vsym = SX;
    end

    % --- Validate inputs ---
    req = {'X_lim', 'Y_lim', 'U', 'V'};
    for k = 1:numel(req)
        if ~isfield(current_params, req{k})
            error('create_symbolic_ocean_func_real:MissingField', ...
                'current_params.%s is required.', req{k});
        end
    end

    X_lim = current_params.X_lim;
    Y_lim = current_params.Y_lim;
    U = double(current_params.U);
    V = double(current_params.V);

    if ~isequal(size(X_lim), size(Y_lim), size(U), size(V))
        error('create_symbolic_ocean_func_real:SizeMismatch', ...
            'X_lim, Y_lim, U, V must be the same size.');
    end

    if ~isfield(current_params, 'noise_level'); current_params.noise_level = 0; end
    if ~isfield(current_params, 'num_ensemble_members'); current_params.num_ensemble_members = 0; end
    if ~isfield(current_params, 'num_ensemble_members_test'); current_params.num_ensemble_members_test = 0; end

    noise_level = current_params.noise_level;
    M = current_params.num_ensemble_members;
    T = current_params.num_ensemble_members_test;

    % --- Build rectilinear grids (x along columns, y along rows) ---
    xgrid = X_lim(1, :);
    ygrid = Y_lim(:, 1)';

    if numel(xgrid) < 2 || numel(ygrid) < 2
        error('create_symbolic_ocean_func_real:GridTooSmall', ...
            'X_lim/Y_lim must define at least a 2x2 grid.');
    end

    % Ensure monotonic increasing grids for CasADi interpolant
    if xgrid(1) > xgrid(end)
        xgrid = fliplr(xgrid);
        U = fliplr(U);
        V = fliplr(V);
    end
    if ygrid(1) > ygrid(end)
        ygrid = fliplr(ygrid);
        U = flipud(U);
        V = flipud(V);
    end

    if any(diff(xgrid) <= 0) || any(diff(ygrid) <= 0)
        error('create_symbolic_ocean_func_real:NonMonotonicGrid', ...
            'X_lim(1,:) and Y_lim(:,1) must be strictly monotonic (after flipping).');
    end

    % --- Build a fully-defined numeric field for CasADi LUTs ---
    % Priority:
    %   1) meta.{U_filled_rect,V_filled_rect,xgrid_rect,ygrid_rect} if provided
    %   2) evaluate prelearned Uc/Vc on the rectilinear grid (build-time only)
    %   3) fill NaNs in U/V with scatteredInterpolant fallback
    [Xg, Yg] = meshgrid(xgrid, ygrid); % (ny x nx)

    used_meta_rect = false;
    if isfield(current_params, 'meta') && isstruct(current_params.meta)
        meta = current_params.meta;
        if isfield(meta, 'U_filled_rect') && isfield(meta, 'V_filled_rect') && ...
           isfield(meta, 'xgrid_rect') && isfield(meta, 'ygrid_rect')
            try
                if isequal(size(meta.U_filled_rect), size(Xg)) && isequal(size(meta.V_filled_rect), size(Xg)) && ...
                   isequal(meta.xgrid_rect, xgrid) && isequal(meta.ygrid_rect, ygrid)
                    U_filled = double(meta.U_filled_rect);
                    V_filled = double(meta.V_filled_rect);
                    used_meta_rect = true;
                end
            catch
                used_meta_rect = false;
            end
        end
    end

    if ~used_meta_rect
        use_prelearned = isfield(current_params, 'Uc') && isfield(current_params, 'Vc') && ...
            isa(current_params.Uc, 'function_handle') && isa(current_params.Vc, 'function_handle');

        if use_prelearned
            try
                pts = [Xg(:)'; Yg(:)']; % 2xN
                Uq = current_params.Uc(pts);
                Vq = current_params.Vc(pts);
                U_filled = reshape(double(Uq), size(Xg));
                V_filled = reshape(double(Vq), size(Xg));

                if any(~isfinite(U_filled(:))) || any(~isfinite(V_filled(:)))
                    U_filled = local_fill_nans(Xg, Yg, U);
                    V_filled = local_fill_nans(Xg, Yg, V);
                end
            catch
                U_filled = local_fill_nans(Xg, Yg, U);
                V_filled = local_fill_nans(Xg, Yg, V);
            end
        else
            U_filled = local_fill_nans(Xg, Yg, U);
            V_filled = local_fill_nans(Xg, Yg, V);
        end
    end

    % CasADi expects values as vector with dimensions matching {xgrid, ygrid}.
    % Our U_filled is (ny x nx) = (y,x), so transpose to (x,y) before vectorizing.
    U_vals = U_filled';
    V_vals = V_filled';

    U_lut = interpolant('U_lut', 'bspline', {xgrid, ygrid}, U_vals(:));
    V_lut = interpolant('V_lut', 'bspline', {xgrid, ygrid}, V_vals(:));

    % --- Symbolic variables ---
    P = Vsym.sym('P', 2, 1); % position
    t = Vsym.sym('t', 1, 1); %#ok<NASGU> time (unused; kept for interface parity)

    % Clamp outside domain to 0 (consistent with learn_ocean_currents_real)
    xMin = xgrid(1); xMax = xgrid(end);
    yMin = ygrid(1); yMax = ygrid(end);
    inside = (P(1) >= xMin) & (P(1) <= xMax) & (P(2) >= yMin) & (P(2) <= yMax);

    u0 = if_else(inside, U_lut(P), Vsym.zeros(1, 1));
    v0 = if_else(inside, V_lut(P), Vsym.zeros(1, 1));
    base_current = [u0; v0];

    % --- Assemble ensemble outputs (same naming style as create_symbolic_ocean_func.m) ---
    nOut = M + 1 + T;
    current_vec = cell(1, nOut);
    J = cell(1, nOut);
    outputCurrentNames = cell(1, nOut);
    outputGradientNames = cell(1, nOut);

    % Average slot
    avgIdx = M + 1;
    current_vec{avgIdx} = Vsym.zeros(2, 1);
    J{avgIdx} = Vsym.zeros(2, 2);
    outputCurrentNames{avgIdx} = 'current_out_avg';
    outputGradientNames{avgIdx} = 'gradient_out_avg';

    % Ensemble members (contribute to avg)
    for j = 1:M
        A = (eye(2) + diag(randn(2, 1) * noise_level));
        current_vec{j} = A * base_current;
        J{j} = jacobian(current_vec{j}, P);
        outputCurrentNames{j} = ['current_out_', num2str(j)];
        outputGradientNames{j} = ['gradient_out_', num2str(j)];

        current_vec{avgIdx} = current_vec{avgIdx} + current_vec{j} / max(1, M);
        J{avgIdx} = J{avgIdx} + J{j} / max(1, M);
    end

    % If no ensemble members, define avg as the base field (no noise)
    if M == 0
        current_vec{avgIdx} = base_current;
        J{avgIdx} = jacobian(current_vec{avgIdx}, P);
    end

    % Test members (do not contribute to avg)
    for k = 1:T
        idx = avgIdx + k;
        A = (eye(2) + diag(randn(2, 1) * noise_level));
        current_vec{idx} = A * base_current;
        J{idx} = jacobian(current_vec{idx}, P);
        outputCurrentNames{idx} = ['current_test_', num2str(k)];
        outputGradientNames{idx} = ['gradient_test_', num2str(k)];
    end

    ocean_func_single = Function('ocean_current_single', ...
        {P, t}, ...
        current_vec, ...
        {'pos_in', 't_in'}, ...
        outputCurrentNames);

    ocean_gradient_func = Function('ocean_gradient_single', ...
        {P, t}, ...
        J, ...
        {'pos_in', 't_in'}, ...
        outputGradientNames);

    ocean_func = ocean_func_single.map(1, 'serial');
    ocean_gradient_func = ocean_gradient_func.map(1, 'serial');
end

function Z_filled = local_fill_nans(Xg, Yg, Z)
    Z_filled = Z;
    nanMask = ~isfinite(Z_filled);
    if ~any(nanMask(:))
        return;
    end

    X = Xg(:);
    Y = Yg(:);
    Zvec = Z_filled(:);
    ok = isfinite(X) & isfinite(Y) & isfinite(Zvec);

    if nnz(ok) < 3
        % Too few samples -> zero field everywhere (consistent with learn_ocean_currents_real)
        Z_filled(:) = 0;
        return;
    end

    F_nat = scatteredInterpolant(X(ok), Y(ok), Zvec(ok), 'natural', 'none');
    F_nn = scatteredInterpolant(X(ok), Y(ok), Zvec(ok), 'nearest', 'nearest');

    Zi = F_nat(Xg(nanMask), Yg(nanMask));
    bad = isnan(Zi);
    if any(bad)
        tmpX = Xg(nanMask);
        tmpY = Yg(nanMask);
        Zi(bad) = F_nn(tmpX(bad), tmpY(bad));
        Zi(isnan(Zi)) = 0;
    end

    Z_filled(nanMask) = Zi;
end