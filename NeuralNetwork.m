
% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex3data1.mat'); % training data stored in arrays X, y
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;


function [h, display_array] = displayData(X, example_width)


% Set example_width automatically if not passed in
if ~exist('example_width', 'var') || isempty(example_width) 
	example_width = round(sqrt(size(X, 2)));
end

% Gray Image
colormap(gray);

% Compute rows, cols
[m n] = size(X);
example_height = (n / example_width);

% Compute number of items to display
display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);

% Between images padding
pad = 1;

% Setup blank display
display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));

% Copy each example into a patch on the display array
curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > m, 
			break; 
		end
		% Copy the patch
		
		% Get the max value of the patch
		max_val = max(abs(X(curr_ex, :)));
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, 
		break; 
	end
end

% Display Image
h = imagesc(display_array, [-1 1]);

% Do not show axis
axis image off

drawnow;

end

displayData(sel);


function g=sigmoid(z)
    
    g=1./(1+exp(-z));
    
endfunction
    
function [J,gd] = computecost(theta,x,y,lambda); 
              m=length(y);
              J=0;
              gd=zeros(size(theta));
              h=sigmoid(x * theta);
              cost = sum(-y.* log(h) -(1-y) .* log(1-h));
              grad = x' * (h-y);

              grad_reg = lambda * theta;
              grad_reg(1) = 0;

              grad = grad + grad_reg;

              J = cost / m + (lambda / (2.0 * m)) * sum(theta(2:size(theta)) .^ 2);
              gd = grad / m;

endfunction

fprintf('\nTesting lrCostFunction() with regularization');

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J , gd] = computecost(theta_t, X_t, y_t, lambda_t);

fprintf('\nCost: %f\n', J);
fprintf('Gradients:\n');
fprintf(' %f \n', gd);
     

function [all_theta] = oneVsAll(X, y, num_labels, lambda)

m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50);
for c = 1:num_labels
	[theta] = fminunc(@(t)(computecost(t, X, (y == c), lambda)), initial_theta, options);
	all_theta(c, :) = theta';
end



% =========================================================================


end     

fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


function pred = predictOneVsAll(all_theta, X)

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
pred = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];



[_, pred] = max(sigmoid(X * all_theta'), [], 2);


% =========================================================================


end

pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

