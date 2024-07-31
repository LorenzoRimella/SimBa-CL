# SimBa-CL: Simulation Based Composite Likelihood

Exact inference for high-dimensional hidden Markov models has a computational cost that is exponential in the dimension, making it unfeasible for even small models. For this reason factorial structures can be exploited in the model to design cheap local algorithms. However, even this last resort might fail when the dependence across time time steps is dense, keeping the cost exponential. We propose to see the likelihood of high-dimensional hidden Markov models as a composite marginal likelihood, where each marginal is computed via Monte Carlo approximation. The overall procedure relies on simulations from the models that allow us to decouple the dependence and so exploit the factorial structures in the model. The resulting algorithm is at most quadratic in the dimension and show significant improvements compared to baselines in terms of computational cost and inference.

# Repository description
The repository contains all the details to reproduce our study and figures. The tutorial and the scripts are tested on TensorFlow 2.9.1 and TensorFlow Probability 0.15.0
- folder "data" contains all the experiments results and input
- folder "scripts" contains the implementation of the methodologies
- "tutorial_KL.ipynb" is a notebook showing how to perform the KL comparison
- "tutorial_optimization_and_empirical_coverage.ipynb" is a notebook showing how to perform optimization of the parameters and test the coverage, at the end of the notebook a discussion on the FM data can be found
- "tutorial_parameters_grid.ipynb" is a notebook showing how to perform a grid comparison 
- "tutorial_performance.ipynb" is a notebook showing how to perform a comparison among SimBa-CL and some SMC baselines 
