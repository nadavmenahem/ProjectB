clc; clear all; close all;


addpath(genpath("C:\Program Files\MATLAB\R2024a\toolbox\matpower8.0")); % Replace with location of matPower

savepath;

% Load the IEEE 39-bus system
mpc = loadcase('case39'); 

% For IEEE 118-bus
% mpc = loadcase('case118');

% Number of branches in the system
num_lines = size(mpc.branch, 1);

% Number of line outage cases to simulate
num_outages = 20;

% Data storage
phasor_angles = []; % Will hold phasor angle data for all cases
labels = [];        % 1 for outage, 0 for non-outage

% Loop over outage cases
for case_idx = 1:num_outages
    % Randomly select a line to remove
    outage_line = randi(num_lines);
    
    % Create a copy of the original case for this simulation
    mpc_outage = mpc;
    
    % Remove the selected line (set it as "out of service")
    mpc_outage.branch(outage_line, :) = [];
    
    % Solve power flow
    results = runpf(mpc_outage, mpoption('verbose', 0));
    
    % Store phasor angles
    if results.success
        % Collect bus voltage angles (phasor angles)
        phasor_angles = [phasor_angles, results.bus(:, 9)]; % Voltage angles (in degrees)
        labels = [labels, 1]; % Mark this as an outage case
    else
        warning('Power flow did not converge for case %d.', case_idx);
    end
end

% Collect data for normal cases (no outage)
for case_idx = 1:(num_outages)
    results = runpf(mpc, mpoption('verbose', 0));
    if results.success
        phasor_angles = [phasor_angles, results.bus(:, 9)]; % Voltage angles (in degrees)
        labels = [labels, 0]; % Mark this as a non-outage case
    else
        warning('Power flow did not converge for normal case %d.', case_idx);
    end
end


for t = 1:(200 * 8) % Simulate 200s at 1/8s interval
    % Perturb system state or keep it constant during simulation
    results = rundcpf(mpc_outage); % DC power flow
    phasor_angles = [phasor_angles, results.bus(:, 9)]; % Voltage angles (in degrees)
end

% Save data for training
data.X = phasor_angles; % Spatial-temporal data
data.y = labels;        % Labels
save('line_outage_data.mat', 'data');
