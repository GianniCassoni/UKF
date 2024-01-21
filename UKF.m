% (c) 2024 Gianni Cassoni <giannicassoni@polimi.it>
% This code is licensed under MIT license 
function [x] = UKF(x, d, u, t, FIR_L, FIR_lambda, FIR_dim_in, FIR_dim_out, x0, P0, Q_noise, R_noise)
% This function implements the Unscented Kalman Filter (UKF) algorithm for
% a discrete-time system with state vector x, output vector d, input vector u,
% and time step t. 
% Inputs:
% x: state vector (column vector of size FIR_dim_in)
% d: output vector (column vector of size FIR_dim_out)
% u: input vector (column vector of size FIR_dim_in)
% t: time step (scalar)
% FIR_L: scaling parameter for the sigma points (scalar)
% FIR_lambda: scaling parameter for the weights (scalar)
% FIR_dim_in: dimension of the input vector (scalar)
% FIR_dim_out: dimension of the output vector (scalar)
% x0: initial state vector (column vector of size FIR_dim_in)
% P0: initial state covariance matrix (square matrix of size FIR_dim_in)
% Q_noise: process noise covariance matrix (square matrix of size FIR_dim_in)
% R_noise: measurement noise covariance matrix (square matrix of size FIR_dim_out)
% Outputs:
% x: updated state vector (column vector of size FIR_dim_in)
persistent P lambda L chi yepsilon dim_in dim_out Q R Wm Wc % declare persistent variables to store previous values
% \\ The following is initialization, and is executed once
if (ischar(x) && strcmp(x,'initial')) % check if the input x is a string 'initial'
    dim_in =  FIR_dim_in; % set the dimension of the input vector
    dim_out = FIR_dim_out; % set the dimension of the output vector
    L = FIR_L; % set the scaling parameter for the sigma points
    lambda = FIR_lambda; % set the scaling parameter for the weights
    P = P0; % set the initial state covariance matrix
    Q = Q_noise; % set the process noise covariance matrix
    R = R_noise; % set the measurement noise covariance matrix
    x = x0'; % set the initial state vector and transpose it to a column vector
    chi = zeros(dim_in, 2*dim_in + 1); % initialize the sigma points matrix
    yepsilon = zeros(dim_out, 2*dim_in + 1); % initialize the predicted output matrix
    [Wm] = weight(L,lambda,dim_in,"m"); % compute the weights for the mean using the weight function
    [Wc] = weight(L,lambda,dim_in,"c"); % compute the weights for the covariance using the weight function
end
% \\ set the sigma point
chi(:, 1) = x; 
chi(:, 2:1:(dim_in+1))          = x + sqrt((L*lambda))*chol(P); % chol is for the sqrt of a matrix
chi(:, (dim_in+2):1:2*dim_in+1) = x - sqrt((L*lambda))*chol(P); % chol is for the sqrt of a matrix
% \\ loop over all sigma points to predict the state using the BDFT_Discrete function
for k = 1:2*dim_in+1 
    [f,~,~,~] = Function_estimation(chi(:, k),u,t); 
    chi(:, k) = f;      
    [~,~,h,~] = Function_estimation(f,u,t); 
    yepsilon(:, k) = h; 
end
% \\  predict the state and output mean by taking the weighted sum of the sigma points
x = sum(Wm .* chi, 2);      
y = sum(Wm .* yepsilon, 2);
% \\ Calculate predicted state and output covariance and cross-covariance
P   = 0; % initialize the predicted state covariance
Pyy = 0; % initialize the predicted output covariance
Pxy = 0; % initialize the predicted cross-covariance
for k = 1:2*dim_in+1 
    P   = P   + Wc(1,k).* (chi(:,k) - x) * (chi(:,k) - x)';            % Predicted state covariance
    Pyy = Pyy + Wc(1,k).*  (yepsilon(:,k) - y) * (yepsilon(:,k) - y)'; % Predicted output covariance
    Pxy = Pxy + Wc(1,k).* (chi(:,k) - x) * (yepsilon(:,k) - y)';       % Predicted cross-covariance
end
P   = P   + Q; % add the process noise covariance to the predicted state covariance
Pyy = Pyy + R; % add the measurement noise covariance to the predicted output covariance
% \\ compute the Kalman and update the state vector
K = Pxy/Pyy; 
x = x + K*(d - y);
% \\ update the state covariance matrix (P should be spd)
P = P - K*Pyy*K';
%% weight function
function [W] = weight(L,lambda,dim_in, case_str)
    % This function computes the weights for the mean or the covariance
    W = zeros(1,2*dim_in + 1); % initialize the weight vector
    alpha =  1; % set the alpha parameter
    beta  =  2; % set the beta parameter
    switch case_str % switch based on the case string
        case "m" % if the case is for the mean
            W(1) = lambda/(lambda + L); % set the first weight as lambda/(lambda + L)
        case "c" % if the case is for the covariance
            W(1) = lambda/(lambda + L) + 1 - alpha^2 + beta; % set the first weight as lambda/(lambda + L) + 1 - alpha^2 + beta
        otherwise % otherwise
            error("wrong string") % throw an error
    end
    W(2:end) = 1/(2*(lambda + L)); % set the remaining weights as 1/(2*(lambda + L))
end
end
