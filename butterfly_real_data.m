% Script in accordance with Manuscript
% Data collected by Robotikum AB
% Last update: 2024-08-25
% Artem Angelchev Shiryaev


clearvars;
close all;


% Define the datasets
datasets = {'52 grams', '120.5 grams', '154 grams'};

% Add your own path here   
file_path = 'C:\Users\Artem\Desktop\Matlab_files\';

% Initialize results storage
Omega_opt_eq100_results = zeros(length(datasets), 1);

% Constants
XX = 30; % Last XX seconds of steady-state behavior
new_step_size = 5; % Step size for resampled data




% Loop over each dataset
for ds = 1:length(datasets)
    
    %% Step 1: Initialize Parameters and Read Data File Names

    % Select the data set
    data_set = datasets{ds};

    % Define scales, input frequencies, and file path based on the selected data set
    switch data_set
        case '52 grams'
            scales = [0.008, 0.008, 0.006, 0.005, 0.003, 0.003, 0.003, 0.004, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1];
            input_freq = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8, 10, 12, 14, 16];
            path_to_data = [file_path, '52gmat\'];
        case '120.5 grams'
            scales = [0.015, 0.01, 0.01, 0.01, 0.01, 0.008, 0.004, 0.003, 0.006, 0.01, 0.02, 0.04, 0.07, 0.1];
            input_freq = [3, 3.5, 4, 4.5, 4.75, 5, 5.5, 6, 6.5, 7, 9, 11, 13, 15];
            path_to_data = [file_path, '120_5gmat\'];
        case '154 grams'
            scales = [0.015, 0.015, 0.015, 0.008, 0.004, 0.003, 0.006, 0.01, 0.02, 0.03, 0.06, 0.08, 0.1];
            input_freq = [3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 10, 12, 14];
            path_to_data = [file_path, '154gmat\'];
        otherwise
            error('Invalid data set selected.');
    end

    % Read names of data files
    names_of_data_files_as_a_cell = read_file_names(path_to_data);
    number_of_recorded_experiments = length(names_of_data_files_as_a_cell{1});

    % Initialize storage for results
    As = zeros(number_of_recorded_experiments, 1);
    Oms = zeros(number_of_recorded_experiments, 1);
    Order_stat_all = cell(number_of_recorded_experiments, 1);
    raw_data_amp_omega = cell(number_of_recorded_experiments, 1);

    %% Analysis part
    for NN = 1:number_of_recorded_experiments
        
        %% Step 2: Load real experiment data 
        currently_loaded_file = fullfile(path_to_data, names_of_data_files_as_a_cell{1}{NN});
        load(currently_loaded_file, 'D');

        % Extract data
        t = D(:, 1);
        u = D(:, 2);
        q = D(:, 3);
        dq = D(:, 4); % data are: time "t", input "u", angle "q", angular velocity "dq"  



        %% Step 4: Analyze sampling irregularity
        dt = diff(t);
        dt_mean = mean(dt);     % <- the average of the sampling steps

        %% Step 5: Resample the recorded values of the angle with the new sampling period 
        q_original = timeseries(q, t);
        uniform_step_size = (t(end) - t(1)) / (length(t) - 1);
        t_with_equally_distributed_sampling = (t(1):uniform_step_size:t(end))';
        q_temp = resample(q_original, t_with_equally_distributed_sampling); 
        q_with_equally_distributed_sampling = q_temp.Data;

        %new_step_size = 5; % This variable defines new step size for resampled data
        q_resampled = q_with_equally_distributed_sampling(1:new_step_size:end);
        t_resampled = t_with_equally_distributed_sampling(1:new_step_size:end);

        %% Step 6: Consider only the last XX seconds of steady-state behavior 
        %XX = 30;   
        dt_r = t_resampled(2) - t_resampled(1); 

        q_steady_state_r = q_resampled(max(1, length(q_resampled) - floor(XX/dt_r) + 1):end);
        t_steady_state_r = t_resampled(max(1, length(t_resampled) - floor(XX/dt_r) + 1):end);

        %% Step 7: Analyze the time response of the (resampled) angle values by taking its DFT 
        y = fft(q_steady_state_r'); 
        y(1) = []; 
        n = length(y);
        power = abs(y(1:floor(n/2))).^2; % Power of first half of transform data
        maxfreq = 1/dt_r*(1/2); % Maximum frequency
        freq = (1:n/2)/(n/2)*maxfreq*2*pi; % Range of frequencies measured in rad/s

        %% Step 8: Compute dominant frequency and estimate amplitude
        [max_of_power_spectrum, ind] = max(power); 
        Om = freq(ind); % The dominant frequency of the response

        min_value = min(q_steady_state_r); 
        max_value = max(q_steady_state_r);
        q_ss_mean = (max_value + min_value) / 2; 
        q_ss_centred = q_steady_state_r - q_ss_mean; 

        q2_ss_centered = abs(q_ss_centred).^2;
        A2 = sqrt(2) * sqrt(sum(q2_ss_centered)) * sqrt(uniform_step_size / XX);

        Oms(NN) = Om;
        As(NN) = 20*log10(A2 / scales(NN));
        
        data_amp_omega = [Oms, As];
        raw_data_amp_omega{NN} = data_amp_omega;
        Order_stat = sortrows(data_amp_omega, -2);
        Order_stat_all{NN} = Order_stat;
    end

    %% Step 9: Compute Omega Opt based on eq 100
    sqrd_omega_gK = (Order_stat_all{number_of_recorded_experiments}(1,1)).^2;
    sqrd_omega_gK1 = (Order_stat_all{number_of_recorded_experiments}(2,1)).^2;
    sqrd_g_K = (Order_stat_all{number_of_recorded_experiments}(1,2)).^2;
    sqrd_g_K1 = (Order_stat_all{number_of_recorded_experiments}(2,2)).^2;

    Omega_opt_eq100 = sqrt((sqrd_omega_gK * sqrt((sqrd_g_K1 / sqrd_g_K) + 1) + sqrd_omega_gK1 * sqrt(2)) ...
        / (sqrt((sqrd_g_K1 / sqrd_g_K) + 1) + sqrt(2)));

    % Store the result for the current dataset
    Omega_opt_eq100_results(ds) = Omega_opt_eq100;

    % Display results for the current dataset
    fprintf('Optimal Omega for %s: %.4f\n', data_set, Omega_opt_eq100);

end

%% Display All Results
disp('Optimal Omega for all datasets:');
for ds = 1:length(datasets)
    fprintf('%s: %.4f\n', datasets{ds}, Omega_opt_eq100_results(ds));
end

%% Constrained Optimization for p1,p2,p3

% Prepare the data
data = [0.052, 0.1205,0.154; % mass in kg
        0.104, 0.104,0.104;     % length in cm
        4.1889, 5.7606,6.281;]';% optimal omega based on eq 100


% Constants approximate from experience
M = 0.2; 
L = 0.104; 
Jc = 0.003; 
g = 9.81;

% Initial guess for p1, p2, and p3
initial_guess = [(M*L^2 + Jc)*0.5, 0.001*0.5, (M*L*g)*0.5]; % Adjusted for p1, p2, and p3

% Define lower and upper bounds for p1, p2, and p3
lb = [0, 0, 0]; % Lower bounds for p1, p2, and p3
ub = [(M*L^2 + Jc)*20, 0.001*20, (M*L*g)*20]; % Upper bounds for p1, p2, and p3

% Objective function to minimize
objective_function = @(params) compute_f(params, data, g);

% Perform optimization using fmincon with bounds
options = optimoptions('fmincon', 'Display', 'iter'); % Show iteration information
estimated_params = fmincon(objective_function, initial_guess, [], [], [], [], lb, ub, [], options);

% Display the estimated parameters
disp('Estimated p1, p2, and p3:');
disp(estimated_params);



%% Function to compute f(p1, p2, p3)
function f_value = compute_f(params, data, g)
    p1 = params(1);
    p2 = params(2);
    p3 = params(3);
    
    % Initialize the sum
    f_value = 0;
    
    % Loop over the data points
    for i = 1:size(data, 1)
        m_a_k = data(i, 1);
        l_a_n = data(i, 2);
        omega_opt = data(i, 3); % Use the Omega from eq 100
        
        % Calculate the left side of the equation (Omega opt squared)
        omega_opt_sq = omega_opt^2;
        
        % Calculate the right side of the equation
        term1 = (p3 + m_a_k * g * l_a_n) / (p1 + m_a_k * l_a_n^2);
        term2 = (1/2) * (p2^2) / (p1 + m_a_k * l_a_n^2)^2;
        right_side = term1 - term2;
        
        % Sum of squared differences
        f_value = f_value + abs((omega_opt_sq - right_side))^2;
    end
end

%% Reading data function
function A = read_file_names(path_to_data)
    % Returns the list of names of files stored in the directory
    listed_files_names = fullfile(path_to_data, 'file_names.txt');
    fileID = fopen(listed_files_names, 'r');
    if fileID == -1
        error('File not found: %s', listed_files_names);
    end
    formatSpec = '%s';
    A = textscan(fileID, formatSpec);
    fclose(fileID);
end
