% =========================================================================
% MATLAB Post-Processing: Optimized R1 Module (1M Steps)
% =========================================================================

clear; clc; close all;

% 1. Load the Exported Data
cum_reward = readmatrix('matlab_data/rewards.csv');
ep_length = readmatrix('matlab_data/episode_lengths.csv');
success_rate = readmatrix('matlab_data/success_rates.csv');

% Generate the X-axis (Episode Number)
episodes = 1:length(cum_reward);

% 2. Statistical Variance
var_reward = var(cum_reward);
var_length = var(ep_length);
var_success = var(success_rate);

% 3. Polynomial Line of Best Fit (LOBF) 
[p_r, S_r, mu_r] = polyfit(episodes, cum_reward', 2);
[p_len, S_len, mu_len] = polyfit(episodes, ep_length', 2);
[p_suc, S_suc, mu_suc] = polyfit(episodes, success_rate', 2);

[y_r, ~] = polyval(p_r, episodes, S_r, mu_r);
[y_len, ~] = polyval(p_len, episodes, S_len, mu_len);
[y_suc, ~] = polyval(p_suc, episodes, S_suc, mu_suc);

% 4. Moving Average Filters (Window = 1000)
filt_coeffs = ones(1, 1000) / 1000;
filt_reward = filter(filt_coeffs, 1, cum_reward);
filt_length = filter(filt_coeffs, 1, ep_length);
filt_success = filter(filt_coeffs, 1, success_rate);

% =========================================================================
% 5. Plotting the 3x3 Master Grid
% Row 1: Cumulative Reward | Row 2: Episode Length | Row 3: Success Rate
% =========================================================================

figure('Name', 'R1 Optimization: Training Performance', 'Position', [100, 100, 1200, 800]);

% --- ROW 1: Cumulative Rewards ---
subplot(3,3,1)
plot(episodes, cum_reward, '.', 'Color', [0.7 0.7 0.7])
xlabel("Episode Number")
ylabel("Reward")
title("Raw Rewards (R1)")

subplot(3,3,2)
plot(episodes, y_r, 'b', 'LineWidth', 2)
xlabel("Episode Number")
ylabel("Reward")
title("LOBF Rewards (R1)")

subplot(3,3,3)
plot(episodes, filt_reward, 'k', 'LineWidth', 1.5)
xlabel("Episode Number")
ylabel("Reward")
title("MA Rewards (R1)")

% --- ROW 2: Episode Length (Efficiency Proof) ---
subplot(3,3,4)
plot(episodes, ep_length, '.', 'Color', [0.7 0.7 0.7])
xlabel("Episode Number")
ylabel("Timesteps")
title("Raw Ep Length (R1)")

subplot(3,3,5)
plot(episodes, y_len, 'r', 'LineWidth', 2)
xlabel("Episode Number")
ylabel("Timesteps")
title("LOBF Ep Length (R1)")

subplot(3,3,6)
plot(episodes, filt_length, 'k', 'LineWidth', 1.5)
xlabel("Episode Number")
ylabel("Timesteps")
title("MA Ep Length (R1)")
yline(4.1, '--r', 'Convergence');

% --- ROW 3: Success Rate (Accuracy Proof) ---
subplot(3,3,7)
plot(episodes, success_rate, '.', 'Color', [0.7 0.7 0.7])
xlabel("Episode Number")
ylabel("Success (0-1)")
title("Raw Success Rate (R1)")

subplot(3,3,8)
plot(episodes, y_suc, 'g', 'LineWidth', 2)
xlabel("Episode Number")
ylabel("Success (0-1)")
title("LOBF Success Rate (R1)")

subplot(3,3,9)
plot(episodes, filt_success, 'k', 'LineWidth', 1.5)
xlabel("Episode Number")
ylabel("Success (0-1)")
title("MA Success Rate (R1)")
yline(1.0, '--g', '100%');