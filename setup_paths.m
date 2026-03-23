if ismac
    casadi_path = '/Users/lavish/Dev/software/casadi-3.7.2-osx_arm64-matlab2018b'; 
else
    casadi_path = '/media/lavish/OS/Users/ROG2021/Documents/projects/casadi-3.7.0-linux64-matlab2018b';
end
addpath(genpath(casadi_path));
disp('Added casadi path');

addpath(genpath(pwd));
disp('Added current directory path');

disp('Setup paths complete');

