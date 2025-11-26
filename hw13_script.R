############################
####### HOMEWORK 13 ########
####### Maddie Thall #######
############################

library(matrixStats)

dragons = read.csv("dragon_data.csv")

## OBJECTIVE 1: The analytical solution

X = cbind(1, dragons$size)
Y = dragons$acres_on_fire

beta_hat = solve(t(X) %*% X) %*% t(X) %*% Y
beta_hat
# Intercept: -1.38
# Slope: 1.35 (big dragons burn more area)


## OBJECTIVE 2: Ordinary least squares

### part a: grid search
x = dragons$size
y = dragons$acres_on_fire

sse = function(b0, b1, x, y) {
  y_hat = b0 + b1 * x
  sum((y - y_hat)^2)
}

b0_center = beta_hat[1]
b1_center = beta_hat[2]

b0_vals = seq(b0_center - 5, b0_center + 5, by = 0.1)
b1_vals = seq(b1_center - 5, b1_center + 5, by = 0.1)

SSE_mat = outer(b0_vals, b1_vals, Vectorize(function(b0, b1) sse(b0, b1, x, y)))
min_idx = which(SSE_mat == min(SSE_mat), arr.ind = TRUE)[1, ]
grid_b0_best = b0_vals[min_idx[1]]
grid_b1_best = b1_vals[min_idx[2]]
grid_b0_best; grid_b1_best
# Intercept: -1.38
# Slope: 1.35

### part b: optimization with optim()
sse_optim = function(params, x, y) {
  b0 = params[1]
  b1 = params[2]
  y_hat = b0 + b1 * x
  sum((y - y_hat)^2)
}

start_val = c(0, 0)

opt_result = optim(
  par = start_val,
  fn = sse_optim,
  x = x,
  y = y
)

opt_result$par 
# Intercept: -1.37
# Slope: 1.35
opt_result$convergence # 0 (optimization successful!)

### part c: check sensitivity to starting values
solutions = matrix(NA, nrow = 5, ncol = 2)

start_vals = list(
  c(0, 0),
  c(10, -5),
  c(-10, 10),
  c(2, 2),
  c(-3, 1)
)

for (i in 1:5) {
  tmp = optim(
    par = start_vals[[i]],
    fn = sse_optim,
    x = x,
    y = y
  )
  solutions[i, ] = tmp$par
}

solutions
# Intercepts: (-1.345) - (-1.397), all approx. equal
# Slopes: 1.345 - 1.347, all approx. equal
# All converged to the same estimates, meaning optimization is stable!


## OBJECTIVE 3: Maximum likelihood

n = length(y)
negloglik = function(params, x, y) {
  b0_obj3 = params[1]
  b1_obj3 = params[2]
  sigma = params[3]
  
  if (!is.numeric(sigma) || length(sigma) != 1 || sigma <= 0) return(Inf)
  
  mu = b0_obj3 + b1_obj3 * x
  n = length(y)
  
  loglik = -n/2 * log(2*pi*sigma^2) - sum((y - mu)^2) / (2*sigma^2)
  return(-loglik)
}

### part a: grid search
x = as.numeric(x)
y = as.numeric(y)
grid_results = expand.grid(b0 = b0_vals, b1 = b1_vals)
grid_results$negLL = NA_real_

for (i in 1:nrow(grid_results)) {
  b0 = grid_results$b0[i]
  b1 = grid_results$b1[i]
  
  mu = b0 + b1 * x
  
  sigma_hat = sqrt(mean((y - mu)^2, na.rm = TRUE))
  
  if (!is.finite(sigma_hat) || sigma_hat <= 0)
    sigma_hat = 1e6  
  
  param_vec = c(b0, b1, sigma_hat)
  
  grid_results$negLL[i] = negloglik(param_vec, x, y)
}

best_result = grid_results[which.min(grid_results$negLL), ]
best_result
# Intercept: -1.38
# Slope: 1.35

### part b: optimization with optim()
mle_opt = optim(
  par = c(0, 0, sd(y, na.rm = TRUE)),
  fn = negloglik,
  x = x,
  y = y,
  method = "BFGS"
)
mle_opt$par
# Intercept: -1.38
# Slope: 1.35
mle_opt$convergence
# 0 (optimization successful!)

### part c: check sensitivity to starting values
starts = list(
  c(0, 0, 1),
  c(5, -5, 10),
  c(-10, 10, 2),
  c(1, 1, 1),
  c(-3, 3, 0.5)
)

sols = matrix(NA, nrow = length(starts), ncol = 4)
colnames(sols) = c("b0", "b1", "sigma", "convergence")

for (i in 1:length(starts)) {
  res = optim(
    par = starts[[i]],
    fn = negloglik,
    x = x,
    y = y,
    method = "BFGS"
  )
  sols[i, ] = c(res$par, res$convergence)
}

sols
# 4 out of 5 converge to same or very close estimates
# 1 starting value (-10, 10, 2) did not converge


## OBJECTIVE 4: compare

# All slope and intercept estimates are the same across methods!
# All intercepts are approx -1.38 and all slopes are approx 1.35
