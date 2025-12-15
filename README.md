## Assignment
Localization. Consider the PR2 robot navigating in an environment with obstacles. Implement a
function that simulates a simple location sensor in pybullet (i.e. the function should return a slightly
noisy estimate of the true location). Pick an interesting path for the robot to execute and estimate the
robot’s position as it executes the path using a) a Kalman filter, and b) a particle filter. You will need
to tune the noise in the sensing and action and the parameters of the algorithms to make sure there is
enough noise to make the problem interesting but not too much so that it’s impossible to estimate the
location. Compare the performance of the two algorithms in terms of accuracy in several interesting
scenarios. Create a case where the Kalman filter is unable to produce a reasonable estimate (e.g. the
mean is inside an obstacle) but the particle filter does produce a reasonable estimate. Include the
motion model, sensor model, and noise covariances you used in your report.