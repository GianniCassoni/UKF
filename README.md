# UKF
Unscented kalman filter example: 
```matlab
% Initialize the UKF with the first two measurements and the input
x = UKF('initial',[d(2) d(3)]',u(1), t(1), L, lambda, dim_in, dim_out, x0, P0, Q, R);

% Preallocate a matrix to store the state estimates
x_ukf = zeros(length(d),dim_in);

% Run the UKF for each measurement
for j = 2:length(y)
    x = UKF(x,[d(j+1) d(j+2)]',u(j), t);
    x_ukf(j, :) = x';
end
```
