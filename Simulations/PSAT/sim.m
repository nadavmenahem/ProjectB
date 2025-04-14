clc; clear all; close all;


addpath(genpath("C:\Program Files\MATLAB\R2024a\toolbox\psat")); % Replace with location of PSAT

savepath;

%fid = fopen('testfile.txt', 'w');
%fprintf(fid, 'Write test successful.');
%fclose(fid);


loadcase('C:\Program Files\MATLAB\R2024a\toolbox\psat\tests\d_006.mdl'); % Load the IEEE 39-bus system data

%runpsat('td');  

%addpath(genpath("C:\Program Files\MATLAB\R2024a\toolbox\matpower8.0")); % Replace with location of matPower

%runpsat('pf'); % Solve power flow
%runpsat('d_003', 'C:\Program Files\MATLAB\R2024a\toolbox\psat\data'); % For IEEE 39-bus