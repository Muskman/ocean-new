function [lat,lon,q_x,q_y,q_x_m,q_y_m,u,v,mag] = real_ocean_environment() 

    filename =  'ca_subSCB_das_2017121621.nc'; % 
    conv_factor = 1;
              
    param_id.lat = 2;
    param_id.lon = 3;
    param_id.U = 6;
    param_id.V = 7;
    
    % opening .nc file
    ncid = netcdf.open(filename);

    % reading the latitudes and longitudes
    varname = netcdf.inqVar(ncid,param_id.lat);
    varid = netcdf.inqVarID(ncid,varname);
    lat = netcdf.getVar(ncid,varid);

    varname = netcdf.inqVar(ncid,param_id.lon);
    varid = netcdf.inqVarID(ncid,varname);
    lon = netcdf.getVar(ncid,varid);


    % creating the whole ocean space interms of lat-long 
    q_x = zeros(length(lat),length(lon));
    q_y = zeros(length(lat),length(lon));
    for i = 1:length(lon)
        q_y(:,i) = lat;
    end

    for i = 1:length(lat)
        q_x(i,:) = lon;
    end

    % interms of euclidean cordinates
    q_x_m = zeros(length(lat),length(lon));
    q_y_m = zeros(length(lat),length(lon));
    
    for i = 1:length(lat)
        q_y_m(i,:) = 1000 * lldistkm([lat(1),lon(1)],[lat(i),lon(1)])*conv_factor  ;
    end

    for i = 1:length(lon)
        q_x_m(:,i) = 1000 * lldistkm([lat(1),lon(1)],[lat(1),lon(i)])*conv_factor ;
    end

    % reading the ocean currents
    varname = netcdf.inqVar(ncid,param_id.U);
    varid = netcdf.inqVarID(ncid,varname);
    data = netcdf.getVar(ncid,varid);
    u = data(:,:,2);
    u = u.';
    
    varname = netcdf.inqVar(ncid,param_id.V);
    varid = netcdf.inqVarID(ncid,varname);
    data = netcdf.getVar(ncid,varid);
    v = data(:,:,2);
    v = v.';
    
    % finding the magnitude of ocean currents
    [n,m] = size(v);
    mag = zeros(n,m);
    for i = 1:n
        for j = 1:m
            flag = false;

            if u(i,j) == -9999 || u(i,j) == -32768
                u(i,j) = NaN;
                flag = true;
            end

            if v(i,j) == -9999 || v(i,j) == -32768
                v(i,j) = NaN;
                flag = true;
            end

            if flag == false
                try
                    mag(i,j) = sqrt(u(i,j)^2 + v(i,j)^2);
                catch
                    keyboard
                end
            else
                mag(i,j) = NaN;
            end
        end
    end

    u(75,71) = 0;
    v(75,71) = 0;
    mag(75,71) = 0;

    u(74,54:56) = 0;
    v(74,54:56) = 0;
    mag(74,54:56) = 0;
    
end
    