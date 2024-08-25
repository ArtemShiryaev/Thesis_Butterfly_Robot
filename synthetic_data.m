clc, close all, clear all

%% Step 0: Define the number of iterations and initialize storage
N = 3; % Define the number of iterations
Order_stat_all = cell(N,1); % Cell array to store order statistic from iterations
Order_stat_eq77 = cell(N,1);
raw_data_amp_omega = cell(N,1);


% initial conditions
th0 = 0.01; Dth0 = 0;

% true model parameters
M = 0.1; L = 0.05; Jc = 0.003; B = 10; g = 9.81; Kf = 0.001; p3 = B;
p1 = M*L^2 + Jc; p2 = M*L*g;
% Define initial values for p1 and p2
p1_initial = M*L^2 + Jc;
p2_initial = M*L*g;



% Mass and Length change each iteration
Mass_alpha = 0.05; % 
Length_alpha = 0.005; 


% Store values of Mass and Length alpha, p1,p2
Mass_alpha_values = zeros(N,1);
Length_alpha_values = zeros(N,1);
p1_values = zeros(N, 1);
p2_values = zeros(N, 1);

for iter = 1:N
    %% Step 1: Update p1 and p2
    % I vårt ex. p1 = p1, p2 = Kf, p3 = p2, p3 = p4
    p1 = p1_initial + (iter-1)*(Mass_alpha*(Length_alpha.^2) );
    p2 = p2_initial + (iter-1)*(Length_alpha*g*Mass_alpha);

    % Store values
    p1_values(iter) = p1;
    p2_values(iter) = p2;
    Mass_alpha_values(iter) = (iter-1)*Mass_alpha;
    Length_alpha_values(iter) = (iter-1)*Length_alpha;
    
    % Update the system transfer function with new p1 and p2
    sys = tf(p3,[p1 Kf p2]); 
    
    %% Rest of your code remains the same
    % parameters of input sinusoid "A*sin( Om*t )"
    input_freq2 = 1:0.5:7; % rad/sec
    scales2 = [ones(length(input_freq2),1)]*1e-5;
    
    % generic parameters for running simulation
    step_size = 0.01; % step size for the solver
    T = 70;   % simulation time
    NNN = 1+(iter-1); % Internal paramater for eq 77 in simulink
    simIn2 = Simulink.SimulationInput("Simulink_2"); 
    number_of_recorded_experiments = length(scales2);
    
    % Set up
    number_of_iterations2 = 1;
    Oms2 = zeros(number_of_recorded_experiments, number_of_iterations2);
    As2 = zeros(number_of_recorded_experiments, number_of_iterations2);
    seed_rand = 222;
    for NN = 1:number_of_recorded_experiments
        A_in2 = scales2(NN); 
        Om_in2 = input_freq2(NN); 
        for iteration = 1:number_of_iterations2
            %% Step 2: run synthetic experiment NN for generating the  data 
            out = sim(simIn2);
            t2 = out.tout; 
            u2 = out.u.Data; 
        end
    end

    % Best function eq 77/79 (forbidden eq)
    data = [out.MnL.Data out.omega_opt.Data];
    data_unique = unique(data, 'rows', 'stable');
    Order_stat_eq77{iter} = data_unique;

    % (Optional) Display progress
    fprintf('Iteration %d completed with p1 = %.4f and p2 = %.4f\n', iter, p1, p2);
end


% Summary of eqn 77
temp_var = zeros(N,3);
for k=1:N
temp_var(k, :) = [Order_stat_eq77{k}(1,1) Order_stat_eq77{k}(1,2) Order_stat_eq77{k}(1,3)];
end
summary_table2 = array2table(temp_var,'VariableNames', {'Mass alpha', 'Length alpha', 'Optimal Omega Eq 79'});
disp(summary_table2)


%% same script with old values
clearvars -except summary_table2 N


%% Step 1: initialize the Simulink model 

% initial conditions
th0 = 0.01; Dth0 = 0;

% true model parameters
M = 0.1; L = 0.05; Jc = 0.003; B = 10; g = 9.81; Kf = 0.001;

% saving optimal omega


Num123 = N;
Omega_opt_eq100 = zeros(Num123,1);
Alpha_Values = zeros(Num123,2);
for kk=1:Num123
Mass_alpha = 0.05*(kk -1); % 
Length_alpha = 0.005*(kk -1); 
Alpha_Values(kk,:) = [Mass_alpha Length_alpha];

p1 = M*L^2 + Jc + Mass_alpha*Length_alpha^2 ; p2 = M*L*g + Mass_alpha*g*Length_alpha ; p3 = B;
%p1 = Jc + M*(L^2)
%p2 = Kf + Kemf*(Ki/R)
%p3 = M*g*L
%p4 = Ki/R

% I vårt ex. p1 = p1, p2 = Kf, p3 = p2, p3 = p4
sys = tf(p3,[p1 Kf p2]); 

% parameters of input sinusoid "A*sin( Om*t )"
input_freq = [1, 1.5,  2, 2.5,  3, 3.5,  4, 4.5,  5, 5.5,  6, 6.5,  7]; % rad/sec
scales = [1,   1,  1,   1,  1,   1,  1,   1,  1,   1,  1,   1,  1]*1e-5;



% generic parameters for running simulation
step_size = 0.01; % step size for the solver
T = 70;   % simulation time
%simIn = Simulink.SimulationInput("Model_of_pendulum_dynamics_for_SI");
simIn = Simulink.SimulationInput("Simulink_1"); 
number_of_recorded_experiments = length(scales);

% Set up
number_of_iterations = 1;
Oms = zeros(number_of_recorded_experiments, number_of_iterations);
As = zeros(number_of_recorded_experiments, number_of_iterations);


seed_rand = 222;

for NN = 1:number_of_recorded_experiments
    A_in = scales(NN); 
    Om_in = input_freq(NN); 
    for iteration = 1:number_of_iterations
        %% Step 2: run synthetic experiment NN for generating the  data 
        out = sim(simIn);
        t = out.tout; 
        u = out.u.Data; 
        q = out.q.Data; % data are: time "t", input "u", angle "q"  

        %% Add noise to the input signal
        seed_rand = 222*iteration;


        %% Step 3: visualize the input signal 
        u_max = max(u); 
        frequency_of_input_signal = input_freq(NN); 
        %figure(1), plot(t,u), grid on, xlabel('time (sec)'), ylabel('control input (dimensionless)');
        %ttl = strcat('Control input to the system is a sinusoid of amplitude ','=', num2str(u_max),' and frequency  ', '=', num2str(frequency_of_input_signal), ' rad/sec'); 
        %title(ttl); 
        %ylim([-u_max*1.1, u_max*1.1]); 

        %% Step 4: visualize the irregularity of sampling
        dt = zeros(size(t)); 
        dt(1:end-1) = t(2:end); 
        dt = dt - t; 
        dt(end)=[];
        dt_mean = mean(dt);     % <- the average of the sampling steps
        %figure(2), plot(dt(1:end),'o'), hold on, plot(ones(length(dt),1)*dt_mean,'r--','LineWidth',4), hold off, grid on, xlabel('samples (indexes)'), ylabel('dt'), 
        %title('Intervals between samples is varying; the mean value of sampling time is in red');

        %% Step 5: resample and visualize the recorded values of the angle with the new sampling period 
        q_original = timeseries(q,t);
        uniform_step_size = ( t(end) - t(1) )/( length(t)-1);
        t_with_equally_distributed_sampling = (t(1):uniform_step_size:t(end))';
        q_temp = resample(q_original, t_with_equally_distributed_sampling); 
        q_with_equally_distributed_sampling = q_temp.Data;

        new_step_size = 4;       % this variable defines new step size for resampled data
        q_resampled = q_with_equally_distributed_sampling(1:new_step_size:end);
        t_resampled = t_with_equally_distributed_sampling(1:new_step_size:end);

        %figure(3), plot(t,q,'b-'), hold on;
        %figure(3), plot(t_resampled,q_resampled,'r.'), hold off;
        %title('Comparison of the original and resampled signal');

        %% Step 6: ignore the data that correspond to transition to the steady-state behavior and consider only last XX seconds of the experiment 
        XX = 45;   
        dt_r = t_resampled(2) - t_resampled(1); 
        q_steady_state_r = q_resampled( length(q_resampled)-floor(XX/dt_r):end );
        t_steady_state_r = t_resampled( length(q_resampled)-floor(XX/dt_r):end );

        q_steady_state = q_with_equally_distributed_sampling( length(q_with_equally_distributed_sampling)-floor(XX/uniform_step_size):end );
        t_steady_state = t_with_equally_distributed_sampling( length(t_with_equally_distributed_sampling)-floor(XX/uniform_step_size):end );

        %figure(4), plot( t_steady_state_r, q_steady_state_r , 'ro' ),  hold on;
        %figure(4), plot( t_steady_state,   q_steady_state , 'b.' ),  grid on, hold off;
        %xlabel('time (sec)'), ylabel('q (rad)'), title('Steady state behavior of the angle (densed and resampled)');

        %% Step 7: analyze the time response of the (resampled) angle values by taking its DFT 
        y = fft(q_steady_state_r'); 
        y(1) = []; 
        n = length(y);
        power = abs( y(1:floor(n/2)) ).^2;     % power of first half of transform data
        maxfreq = 1/dt_r*(1/2);                 % maximum frequency
        freq = (1:n/2)/(n/2)*maxfreq*2*pi;  % range of frequencies measured in rad/s

        %figure(5), plot(freq, power,'r.-'), grid on, xlim([0.1 max(input_freq)+1.5]); %zoom in on max power input_freq
        %xlabel('frequency \omega (rad/sec)'), ylabel('power (dimensionless)'), 
        %title('Periodogram of the steady state response');

        %% Step 8: observe that the system response is approximately a sinusoid; and compute its frequency
        [max_of_power_spectrum,ind] = max(power); 
        Om = freq(ind);    % the dominant frequency of the response

        %% Step 9: computing an estimate for the amplitude of frequency response at steady-state 
        % compute and remove the mean for the original data for the last XX seconds 
        min_value = min(q_steady_state_r); 
        max_value = max(q_steady_state_r);
        q_ss_mean = (max_value + min_value)/2; 
        q_ss_centred = q_steady_state_r - q_ss_mean; 

        %figure(10), plot(t_steady_state_r, q_ss_centred,'o' ), grid on;

        q2_ss_centered = abs(q_steady_state  - q_ss_mean ).^2;
        %figure(11), plot(t_steady_state,  q2_ss_centered , '.b' ), grid on;

        % estimate 1: if the response would be indeed a sine of frequency "Om" then its magnitude can be approximate  
        A2 = sqrt(2)*sqrt( sum(q2_ss_centered) )*sqrt(uniform_step_size/XX); 
        new_sin = A2*sin( Om*t_steady_state);
        %figure(10), hold on, plot(t_steady_state,new_sin,'g-','LineWidth',2), hold off;

        Oms(NN, iteration) = Om;
        As(NN, iteration) = 20*log10(A2/scales(NN));
    end
end

% % Plot the results
% figure(20), bodemag(sys,{0.9, 9}), hold on;
% figure(20), plot(Oms,As,'ro','MarkerSize',4), grid on, xticks([0.5:0.5:8]), yticks([30:2:50]);
% xlabel('frequency \omega'), ylabel('Amplitude of transfer function (dB)'), 
% title('Amplitude of transfer function');
% figure(20), hold off;
% % 
% figure(30), plot(Oms,As,'ro','MarkerSize',4), grid on, 
% xlim([0.5 7.5]), xticks([0.5:0.5:8.5]), yticks([30:2:50]);
% xlabel('frequency \omega'), ylabel('Amplitude (dB)'), 
% title('Approximate for amplitude of transfer function');
% print('Approximate_for_Gq','-dpdf');


%% New section
temp_variable = As;
temp_variable_max = max(temp_variable);
temp_variable_max2 = max(temp_variable(temp_variable<max(temp_variable)));

temp_variable2 = [As, Oms];

[MM, Index] = max(temp_variable2);
Max__2 = [temp_variable2(Index(1),1), temp_variable2(Index(1),2)];
Max__2;

[MM2, Index2] = max(temp_variable(temp_variable<max(temp_variable)));
Max__3 = [temp_variable2(Index2(1),1), temp_variable2(Index2(1),2)];
Max__3;


Largest_Oms_As = [Max__2;Max__3];


% testing the new methods
% a_0 = p3/p1, a_1 = p2/p1
A__0 = p2/p1;
A__1 = Kf/p1;


Omg_out = sqrt(A__0 - (A__1.^2)/2);
% Omg_out = 3.8788

Omg_out2 = sqrt(M*g*L/(Jc + M*(L^2)) - ((((Kf + B)^2)/2*((Jc + M*(L^2))^2))));
% Omg_out2 = 3.8848

%% Omega Opt
g_K = Largest_Oms_As(1,1);
g_K1 = Largest_Oms_As(2,1);
omega_gK = Largest_Oms_As(1,2);
omega_gK1 = Largest_Oms_As(2,2);
Omega_opt_eq100(kk) = sqrt((omega_gK^2 * sqrt((g_K1^2 / g_K^2) + 1) + omega_gK1^2 * sqrt(2)) / (sqrt((g_K1^2 / g_K^2) + 1) + sqrt(2)));

fprintf('Iteration %d completed with p1 = %.4f and p2 = %.4f\n', kk, p1, p2);

end


summary_data = [Alpha_Values Omega_opt_eq100];
summary_table = array2table(summary_data);
summary_table.Properties.VariableNames = {'Mass alpha', 'Length alpha', 'Optimal Omega eq 100'};


clearvars -except summary_table summary_table2 Kf N

disp(summary_table)
disp(summary_table2)

%% Optimzation of paramaters

%% Compute p1,p3
data = [table2array(summary_table2) table2array(summary_table(:,3))];

data_table = array2table(data);
data_table.Properties.VariableNames = {'Mass alpha', 'Length alpha','Optimal Omega eq 79', 'Optimal Omega eq 100'};

disp(data_table)


% Constants
Kf = 0.001; 
p2 = Kf; 
M = 0.1; L = 0.05; Jc = 0.003; g = 9.81; Kf = 0.001; 

% Objective function to minimize
objective_function = @(params) compute_f(params, data, g, p2);

% Initial guess for p1 and p3
initial_guess = [(M*L^2 + Jc)*0.5, (M*L*g)*0.5];

% Define lower and upper bounds for p1 and p3
lb = [0, 0]; % Lower bounds
ub = [1,1];
%ub = [(M*L^2 + Jc)*5, (M*L*g)*5]; % Upper bounds

% Perform optimization using fmincon with bounds
options = optimoptions('fmincon','Display','iter'); % Show iteration information
estimated_params = fmincon(objective_function, initial_guess, [], [], [], [], lb, ub, [], options);

% Display the estimated parameters
disp('Estimated p1 and p3:');
disp(estimated_params);

real_params = [(M*L^2 + Jc) M*L*g];
disp('Real p1 and p3:');
disp(real_params)

temp_residuals = real_params - estimated_params;
disp('Estimated Residual p1 and p3:');
disp(temp_residuals)


%% Compute p1,p2,p3

% Initial guess for p1, p2, and p3
initial_guess2 = [(M*L^2 + Jc)*0.5, 0.001*0.5, (M*L*g)*0.5]; % Adjusted for p1, p2, and p3

% Define lower and upper bounds for p1, p2, and p3
lb2 = [0, 0, 0]; % Lower bounds for p1, p2, and p3
ub2 = [(M*L^2 + Jc)*10, 0.001*10, (M*L*g)*10]; % Upper bounds for p1, p2, and p3

% Objective function to minimize
objective_function2 = @(params) compute_ff(params, data, g);

% Perform optimization using fmincon with bounds
options2 = optimoptions('fmincon', 'Display', 'iter'); % Show iteration information
estimated_params2 = fmincon(objective_function2, initial_guess2, [], [], [], [], lb2, ub2, [], options2);

% Display the estimated parameters
disp('Estimated p1, p2, and p3:');
disp(estimated_params2);

real_params2 = [(M*L^2 + Jc), 0.001, M*L*g];
disp('Real p1, p2, and p3:');
disp(real_params2)

temp_residuals2 = real_params2 - estimated_params2;
disp('Residuals for p1, p2, and p3:');
disp(temp_residuals2)



%% Function to compute f(p1, p3)
function f_value = compute_f(params, data, g, p2)
    p1 = params(1);
    p3 = params(2);
    
    % Initialize the sum
    f_value = 0;
    
    % Loop over the data points
    for i = 1:size(data, 1)
        m_a_k = data(i, 1);
        l_a_n = data(i, 2);
        omega_opt = data(i, 4); % Use the Omega from eq 100
        
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

%% Function to compute f(p1, p2, p3)
function f_value2 = compute_ff(params, data, g)
    p1 = params(1);
    p2 = params(2);
    p3 = params(3);
    
    % Initialize the sum
    f_value2 = 0;
    
    % Loop over the data points
    for i = 1:size(data, 1)
        m_a_k = data(i, 1);
        l_a_n = data(i, 2);
        omega_opt = data(i, 4); % Use the Omega from eq 100
        
        % Calculate the left side of the equation (Omega opt squared)
        omega_opt_sq = omega_opt^2;
        
        % Calculate the right side of the equation
        term1 = (p3 + m_a_k * g * l_a_n) / (p1 + m_a_k * l_a_n^2);
        term2 = (1/2) * (p2^2) / (p1 + m_a_k * l_a_n^2)^2;
        right_side = term1 - term2;
        
        % Sum of squared differences
        f_value2 = f_value2 + abs((omega_opt_sq - right_side))^2;
    end
end
