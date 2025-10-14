## Common Observation:
- The thetas with higher values get converge faster than thetas with smaller values.
- Number of iterations were lesser for first dataset(except for stocastic with momentum).
  
## Vanilla gradient descent.

- In full barch gradient descent the direction of gradients were not varying much, but in stocastic gradient descent there were a lot of variation in directions of gradients.
- In the dataset woth bigger theta1, the countour plot is more elliptical shape as opposed to where the thetas are very close to each other.

 ## In momentum gradient descent.

- Adding momentum in gradient descent step reduced the number of iteration for convergence in the second dataset
- In first dataset it only reduced the number of steps in full batch gradient descent, and it did not converge in stocastic gradient descent even aftet 1e6 iterations.
- In first dataset, for stocastice GD, It made the theta0 to overshoot its optimal range, and hence it was taking long to return to its optimal value.
