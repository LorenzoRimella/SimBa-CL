# from distutils import log
# from email import iterators
import numpy as np
from synthetic_data import *

# initial distribution for SIS
def SIS_initial_grad(input_0, parameters_0):

    W      = input_0[0]

    beta_0 = parameters_0[0]

    rate_0 = 1/(1+tf.exp(-tf.einsum("nj,...j->...n", W, beta_0)))
    
    initial_distr    = tf.stack((1-rate_0, rate_0), axis = -1)

    return initial_distr

# transition distribution for fully connected SIS
def SIS_transition_grad(input_kernel, parameters_kernel, x_tm1):

    W                                = input_kernel[0]

    beta_lambda, beta_gamma, epsilon = parameters_kernel

    lambda__n = 1/(1+tf.exp(-tf.einsum("nj,...j->...n", W, beta_lambda)))
    gamma__n  = 1/(1+tf.exp(-tf.einsum("nj,...j->...n", W, beta_gamma)))

    c_tm1 = tf.reduce_sum( x_tm1, axis = -2, keepdims = True ) - x_tm1

    N = tf.cast(tf.shape(W)[0], dtype = tf.float32)

    rate_SI   = tf.expand_dims(tf.expand_dims(1-tf.exp(-lambda__n*((c_tm1[...,1]/N)+epsilon[0])), axis =-1), axis =-1)

    rate_IS   = tf.expand_dims(tf.expand_dims(1-tf.exp(-gamma__n), axis =-1), axis =-1)*tf.ones(tf.shape(rate_SI))

    K_eta_h__n_r1 = tf.concat((1 - rate_SI, rate_SI), axis = -1)
    K_eta_h__n_r2 = tf.concat((rate_IS, 1 - rate_IS), axis = -1)
    K_eta_h__n    = tf.concat((K_eta_h__n_r1, K_eta_h__n_r2), axis = -2)

    return K_eta_h__n

def SIS_emission_grad(parameters_emission, y_t):

    q_grad = parameters_emission[0]

    emission_from_observed = tf.expand_dims(y_t*tf.expand_dims(q_grad, axis =0), axis =0)
    emission_from_unobserved =  tf.expand_dims((1-tf.reduce_sum(y_t, axis = 1, keepdims = True))*(1-tf.expand_dims(q_grad, axis =0)), axis =0)

    return emission_from_observed + emission_from_unobserved


def prediction(filtering_tm1, transition_matrix_tm1):

    return tf.einsum("sni,snij->snj", filtering_tm1, transition_matrix_tm1)

def update(filtering_t_tm1, emission_t):

    update_t = (emission_t*filtering_t_tm1)

    likelihood_increment_t_tm1 =  tf.reduce_sum(update_t, axis = -1, keepdims = True)
    filtering_t = update_t/tf.where(update_t==0, tf.ones(tf.shape(likelihood_increment_t_tm1)), likelihood_increment_t_tm1)

    return likelihood_increment_t_tm1[...,0], filtering_t

def categ_sampling(probabilities, seed_cat):
    
    return tfp.distributions.OneHotCategorical( probs = probabilities, dtype=tf.float32).sample(seed = seed_cat)

def relaxed_categ_sampling(probabilities, tau, seed_cat):
    
    return tf.cast(tfp.distributions.RelaxedOneHotCategorical(temperature = tau,  probs = probabilities).sample(seed = seed_cat), dtype=tf.float32)

class simulation_likelihood():
        
    def __init__(self, parallel_simulations, N, input_0, input_kernel, initial_distribution, transition_kernel, emission_distribution, 
                 sampling = categ_sampling):

        self.model = compartmental_model(N, input_0, input_kernel, initial_distribution, transition_kernel)
        self.emission = emission_distribution
        self.parallel_simulations = parallel_simulations
        self.sampling = sampling


# def run_simba_while(simulation_likelihood_model, parameters_0, parameters_kernel, parameters_emission, y, seed_simba_while):
    
#         T = tf.shape(y)[0]

#         seed_simba_while_split = tfp.random.split_seed( seed_simba_while, n=(T+1), salt='run_simba_while')

#         def cond(input, t):

#             return t<T

#         def body(input, t):

#             filtering_tm1, x_tm1_sim, log_likelihood = input

#             transition_matrix_tm1 = simulation_likelihood_model.model.transition_kernel(simulation_likelihood_model.model.input_kernel, parameters_kernel, x_tm1_sim)
#             prob_t = tf.einsum("...ni,...nik->...nk", x_tm1_sim, transition_matrix_tm1)

#             x_t_sim = simulation_likelihood_model.sampling(prob_t, seed_simba_while_split[t])

#             # prediction on the filtering using the feedback
#             filtering_t_tm1 = prediction(filtering_tm1, transition_matrix_tm1)

#             # update as usual
#             y_t = y[t,...]
#             emission_t = simulation_likelihood_model.emission(parameters_emission, y_t)
#             likelihood_increment_t_tm1, filtering_t = update(filtering_t_tm1, emission_t)

#             return (filtering_t, x_t_sim, log_likelihood + tf.math.log(likelihood_increment_t_tm1)), t+1

#         probs_0 = simulation_likelihood_model.model.initial_distribution(simulation_likelihood_model.model.input_0, parameters_0)
#         filtering_0 = tf.expand_dims(probs_0, axis = 0)*tf.ones(tf.concat(([simulation_likelihood_model.parallel_simulations], tf.shape(probs_0)), axis =0))

#         x_0_sim = simulation_likelihood_model.sampling(filtering_0, seed_simba_while_split[0])

#         initializer = (filtering_0, x_0_sim, tf.zeros(tf.shape(x_0_sim)[:-1]))

#         # Compute the function value
#         output = tf.while_loop(cond, body, loop_vars = (initializer, 1))

#         log_likelihood_individuals = output[0][2]
        
#         max_log_likelihood_individuals = tf.reduce_mean(log_likelihood_individuals, axis = 0, keepdims = True) 
#         log_likelihood = tf.math.log(tf.reduce_mean(tf.exp(log_likelihood_individuals - max_log_likelihood_individuals), axis =0)) + tf.reduce_mean(log_likelihood_individuals, axis = 0) 

#         target = log_likelihood

#         return target

@tf.function(jit_compile=True)
def run_simba_full(simulation_likelihood_model, parameters_0, parameters_kernel, parameters_emission, y, seed_simba):
    
    T = tf.shape(y)[0]

    seed_simba_split = tfp.random.split_seed( seed_simba, n=(T+1), salt='run_simba')

    def step_func(input, t):
        filtering_tm1, x_tm1_sim, log_likelihood = input

        transition_matrix_tm1 = simulation_likelihood_model.model.transition_kernel(simulation_likelihood_model.model.input_kernel, parameters_kernel, x_tm1_sim)
        prob_t = tf.einsum("...ni,...nik->...nk", x_tm1_sim, transition_matrix_tm1)

        x_t_sim = simulation_likelihood_model.sampling(prob_t, seed_simba_split[t])

        filtering_t_tm1 = prediction(filtering_tm1, transition_matrix_tm1)

        y_t = y[t,...]
        emission_t = simulation_likelihood_model.emission(parameters_emission, y_t)
        likelihood_increment_t_tm1, filtering_t = update(filtering_t_tm1, emission_t)

        return (filtering_t, x_t_sim, tf.math.log(likelihood_increment_t_tm1))

    probs_0 = simulation_likelihood_model.model.initial_distribution(simulation_likelihood_model.model.input_0, parameters_0)
    filtering_0 = tf.expand_dims(probs_0, axis=0) * tf.ones(tf.concat(([simulation_likelihood_model.parallel_simulations], tf.shape(probs_0)), axis=0))
    x_0_sim = simulation_likelihood_model.sampling(filtering_0, seed_simba_split[0])
    initializer = (filtering_0, x_0_sim, tf.zeros(tf.shape(x_0_sim)[:-1]))

    sequence = tf.range(1, T)

    # Perform the computation using tf.scan
    outputs = tf.scan(step_func, sequence, initializer=initializer)

    log_likelihood_individuals = tf.reduce_sum(outputs[2], axis =0)
    max_log_likelihood_individuals = tf.reduce_mean(log_likelihood_individuals, axis=0, keepdims=True)
    log_likelihood = tf.math.log(tf.reduce_mean(tf.exp(log_likelihood_individuals - max_log_likelihood_individuals), axis=0)) + tf.reduce_mean(log_likelihood_individuals, axis=0)

    target = log_likelihood

    return outputs[0], target

@tf.function(jit_compile=True)
def run_simba_full_filter(simulation_likelihood_model, parameters_0, parameters_kernel, parameters_emission, y, seed_simba):
    
    T = tf.shape(y)[0]

    seed_simba_split = tfp.random.split_seed( seed_simba, n=(T+1), salt='run_simba')

    def step_func(input, t):
        filtering_tm1, x_tm1_sim, log_likelihood = input

        transition_matrix_tm1 = simulation_likelihood_model.model.transition_kernel(simulation_likelihood_model.model.input_kernel, parameters_kernel, x_tm1_sim)

        filtering_t_tm1 = prediction(filtering_tm1, transition_matrix_tm1)

        y_t = y[t,...]
        emission_t = simulation_likelihood_model.emission(parameters_emission, y_t)
        likelihood_increment_t_tm1, filtering_t = update(filtering_t_tm1, emission_t)
        
        x_t_sim = simulation_likelihood_model.sampling(filtering_t, seed_simba_split[t])

        return (filtering_t, x_t_sim, tf.math.log(likelihood_increment_t_tm1))

    probs_0 = simulation_likelihood_model.model.initial_distribution(simulation_likelihood_model.model.input_0, parameters_0)
    filtering_0 = tf.expand_dims(probs_0, axis=0) * tf.ones(tf.concat(([simulation_likelihood_model.parallel_simulations], tf.shape(probs_0)), axis=0))
    x_0_sim = simulation_likelihood_model.sampling(filtering_0, seed_simba_split[0])
    initializer = (filtering_0, x_0_sim, tf.zeros(tf.shape(x_0_sim)[:-1]))

    sequence = tf.range(1, T)

    # Perform the computation using tf.scan
    outputs = tf.scan(step_func, sequence, initializer=initializer)

    log_likelihood_individuals = tf.reduce_sum(outputs[2], axis =0)
    max_log_likelihood_individuals = tf.reduce_mean(log_likelihood_individuals, axis=0, keepdims=True)
    log_likelihood = tf.math.log(tf.reduce_mean(tf.exp(log_likelihood_individuals - max_log_likelihood_individuals), axis=0)) + tf.reduce_mean(log_likelihood_individuals, axis=0)

    target = log_likelihood

    return outputs[0], target

def run_simba(simulation_likelihood_model, parameters_0, parameters_kernel, parameters_emission, y, seed_simba):
    
    T = tf.shape(y)[0]

    seed_simba_split = tfp.random.split_seed( seed_simba, n=(T+1), salt='run_simba')

    def step_func(input, t):
        filtering_tm1, x_tm1_sim, log_likelihood = input

        transition_matrix_tm1 = simulation_likelihood_model.model.transition_kernel(simulation_likelihood_model.model.input_kernel, parameters_kernel, x_tm1_sim)
        prob_t = tf.einsum("...ni,...nik->...nk", x_tm1_sim, transition_matrix_tm1)

        x_t_sim = simulation_likelihood_model.sampling(prob_t, seed_simba_split[t])

        filtering_t_tm1 = prediction(filtering_tm1, transition_matrix_tm1)

        y_t = y[t,...]
        emission_t = simulation_likelihood_model.emission(parameters_emission, y_t)
        likelihood_increment_t_tm1, filtering_t = update(filtering_t_tm1, emission_t)

        return (filtering_t, x_t_sim, tf.math.log(likelihood_increment_t_tm1))

    probs_0 = simulation_likelihood_model.model.initial_distribution(simulation_likelihood_model.model.input_0, parameters_0)
    filtering_0 = tf.expand_dims(probs_0, axis=0) * tf.ones(tf.concat(([simulation_likelihood_model.parallel_simulations], tf.shape(probs_0)), axis=0))
    x_0_sim = simulation_likelihood_model.sampling(filtering_0, seed_simba_split[0])
    initializer = (filtering_0, x_0_sim, tf.zeros(tf.shape(x_0_sim)[:-1]))

    sequence = tf.range(1, T)

    # Perform the computation using tf.scan
    outputs = tf.scan(step_func, sequence, initializer=initializer)

    log_likelihood_individuals = tf.reduce_sum(outputs[2], axis =0)
    max_log_likelihood_individuals = tf.reduce_mean(log_likelihood_individuals, axis=0, keepdims=True)
    log_likelihood = tf.math.log(tf.reduce_mean(tf.exp(log_likelihood_individuals - max_log_likelihood_individuals), axis=0)) + tf.reduce_mean(log_likelihood_individuals, axis=0)

    target = log_likelihood

    return target

@tf.function(jit_compile=True)
def simba_compiled(simulation_likelihood_model, parameters_0, parameters_kernel, parameters_emission, y, seed_loss):

    log_likelihood_individuals = run_simba(simulation_likelihood_model, parameters_0, parameters_kernel, parameters_emission, y, seed_loss)

    return log_likelihood_individuals

@tf.function
def simba_loss(simulation_likelihood_model, parameters_0, parameters_kernel, parameters_emission, y, seed_loss):

    T = tf.cast(tf.shape(y)[0], dtype = tf.float32)

    log_likelihood_individuals = run_simba(simulation_likelihood_model, parameters_0, parameters_kernel, parameters_emission, y, seed_loss)

    return -tf.reduce_mean(log_likelihood_individuals)/T

@tf.function
def simba_loss_grad(simulation_likelihood_model, gradient_var, parameters_0, parameters_kernel, parameters_emission, y, seed_loss_grad):

    with tf.GradientTape() as tape:
        # Call your likelihood function with the parameters_var and other necessary inputs
        loss = simba_loss(simulation_likelihood_model, parameters_0, parameters_kernel, parameters_emission, y, seed_loss_grad)

    # Use TensorFlow's automatic differentiation to compute gradients
    return tape.gradient(loss, gradient_var)

@tf.function
def simba_loss_grad_hessian(simulation_likelihood_model, gradient_var, parameters_0, parameters_kernel, parameters_emission, y, seed_loss_grad_hess):

    with tf.GradientTape() as tape2:

        with tf.GradientTape() as tape1:

            tape1.watch(gradient_var)
            # Call your likelihood function with the parameters_var and other necessary inputs
            loss = simba_loss(simulation_likelihood_model, parameters_0, parameters_kernel, parameters_emission, y, seed_loss_grad_hess)

        # Use TensorFlow's automatic differentiation to compute gradients
        gradient =  tape1.gradient(loss, gradient_var)
    
    hessian  = tape2.jacobian(gradient, gradient_var)

    return gradient, hessian

def optimization_beta_lambda(simulation_likelihood_model, parameters_0, initial_lambda, beta_gamma, epsilon, parameters_emission, y, seed_optimization, lr_rate=100, optimization_steps=50):

    seed_optim_split = tfp.random.split_seed( seed_optimization, n=(optimization_steps+1), salt='optimization_beta_lambda')

    for epoch in range(optimization_steps):

        parameters_kernel_optim = initial_lambda, beta_gamma, epsilon

        gradient = simba_loss_grad(simulation_likelihood_model, initial_lambda, parameters_0, parameters_kernel_optim, parameters_emission, y, seed_optim_split[epoch][0])

        # Apply gradients to update the parameters_var
        initial_lambda = tf.Variable(initial_lambda - lr_rate*gradient)

    return initial_lambda

def gradient_hessian_sample(simulation_likelihood_model, gradient_var, parameters_0, parameters_kernel, parameters_emission, T, N, M, iterations, seed_exp_grad_hess):

    seed_gen_sim, seed_gen_grad_hess = tfp.random.split_seed( seed_exp_grad_hess, n=2, salt='gradient_hessian_sample')

    seed_gen_sim_split = tfp.random.split_seed( seed_gen_sim[0], n=iterations, salt='expected_gradient_hess_sim')
    seed_gen_grad_hess_split = tfp.random.split_seed( seed_gen_grad_hess[0], n=iterations, salt='expected_gradient_hess_grad')

    model = compartmental_model(N, simulation_likelihood_model.model.input_0, simulation_likelihood_model.model.input_kernel, SIS_initial, SIS_transition)

    def body(input, i):        
        
        q_0      = tf.convert_to_tensor([0.0, 0.0], dtype = tf.float32)   
        q_0_expanded      = tf.expand_dims(tf.expand_dims(q_0, axis = 0), axis = 0)
        q_static_expanded = tf.expand_dims(tf.expand_dims(parameters_emission[0], axis = 0), axis = 0)
        q_dynamic = tf.concat((q_0_expanded*tf.ones((1,N, M)), q_static_expanded*tf.ones((T, N, M))), axis =0)

        _, y_sim = sim(model, parameters_0, parameters_kernel, q_dynamic, T, seed_gen_sim_split[i][0])

        gradient, hessian = simba_loss_grad_hessian(simulation_likelihood_model, gradient_var, parameters_0, parameters_kernel, parameters_emission, y_sim, seed_gen_grad_hess_split[i][0])

        return -T*N*gradient, -T*N*hessian

    initialize = (tf.zeros((tf.shape(gradient_var)[0])), tf.zeros((tf.shape(gradient_var)[0], tf.shape(gradient_var)[0])))
    gradients, hessians = tf.scan(body, tf.range(1, iterations), initializer = initialize)

    return gradients, hessians

def optimization_gradient_hessian_sample(simulation_likelihood_model, gradient_var, parameters_0, parameters_kernel, parameters_emission, T, N, M, iterations, seed_exp_opt_grad_hess, lr_rate=100, optimization_steps=50):
    
    seed_gen_sim, seed_gen_opt, seed_gen_grad_hess_1, seed_gen_grad_hess_2 = tfp.random.split_seed( seed_exp_opt_grad_hess, n=4, salt='gradient_hessian_sample')

    seed_gen_sim_split       = tfp.random.split_seed( seed_gen_sim[0], n=iterations, salt='expected_gradient_hess_sim')
    seed_gen_opt_split       = tfp.random.split_seed( seed_gen_opt[0], n=iterations, salt='expected_opt_sim')
    seed_gen_grad_hess_split_1 = tfp.random.split_seed( seed_gen_grad_hess_1[0], n=iterations, salt='expected_gradient_hess_grad_1')
    seed_gen_grad_hess_split_2 = tfp.random.split_seed( seed_gen_grad_hess_2[0], n=iterations, salt='expected_gradient_hess_grad_2')
    
    model = compartmental_model(N, simulation_likelihood_model.model.input_0, simulation_likelihood_model.model.input_kernel, SIS_initial, SIS_transition)

    def body(input, i):        
        
        q_0      = tf.convert_to_tensor([0.0, 0.0], dtype = tf.float32)   
        q_0_expanded      = tf.expand_dims(tf.expand_dims(q_0, axis = 0), axis = 0)
        q_static_expanded = tf.expand_dims(tf.expand_dims(parameters_emission[0], axis = 0), axis = 0)
        q_dynamic = tf.concat((q_0_expanded*tf.ones((1,N, M)), q_static_expanded*tf.ones((T, N, M))), axis =0)
        
        _, y_sim = sim(model, parameters_0, parameters_kernel, q_dynamic, T, seed_gen_sim_split[i][0])

        beta_lambda_initial = tf.Variable(tf.convert_to_tensor(gradient_var.numpy(), dtype = tf.float32 ))
        beta_lambda_optim = optimization_beta_lambda(simulation_likelihood_model, parameters_0, beta_lambda_initial, parameters_kernel[1], parameters_kernel[2], parameters_emission, y_sim, seed_gen_opt_split[i][0], lr_rate, optimization_steps)

        beta_lambda_optim = tf.Variable(beta_lambda_optim)
        parameters_kernel_optim_1 = beta_lambda_optim, parameters_kernel[1], parameters_kernel[2]
        gradient_optim, hessian_optim = simba_loss_grad_hessian(simulation_likelihood_model, beta_lambda_optim, parameters_0, parameters_kernel_optim_1, parameters_emission, y_sim, seed_gen_grad_hess_split_1[i][0])

        parameters_kernel_optim_2 = gradient_var, parameters_kernel[1], parameters_kernel[2]
        gradient, hessian             = simba_loss_grad_hessian(simulation_likelihood_model, gradient_var,      parameters_0, parameters_kernel_optim_2, parameters_emission, y_sim, seed_gen_grad_hess_split_2[i][0])

        return -T*N*gradient, -T*N*hessian, beta_lambda_optim, -T*N*gradient_optim, -T*N*hessian_optim

    initialize = (tf.zeros((tf.shape(gradient_var)[0])), tf.zeros((tf.shape(gradient_var)[0], tf.shape(gradient_var)[0])), tf.zeros((tf.shape(gradient_var)[0])), tf.zeros((tf.shape(gradient_var)[0])), tf.zeros((tf.shape(gradient_var)[0], tf.shape(gradient_var)[0])))
    gradients, hessians, parameters_optim, gradients_optim, hessians_optim = tf.scan(body, tf.range(1, iterations), initializer = initialize)

    return gradients, hessians, parameters_optim, gradients_optim, hessians_optim

@tf.function
def simba_gradients(simulation_likelihood_model, gradient_var, parameters_0, parameters_kernel, parameters_emission, y, seed_gradients):

    with tf.GradientTape() as tape:
        # Call your likelihood function with the parameters_var and other necessary inputs
        log_likelihood_individuals = run_simba_while(simulation_likelihood_model, parameters_0, parameters_kernel, parameters_emission, y, seed_gradients)

    # Use TensorFlow's automatic differentiation to compute gradients
    return tape.jacobian(log_likelihood_individuals, gradient_var)

def individual_gradients_sample(simulation_likelihood_model, gradient_var, parameters_0, parameters_kernel, parameters_emission, T, N, M, iterations, seed_grad_sample):

    seed_gen_sim, seed_gen_grads = tfp.random.split_seed( seed_grad_sample, n=2, salt='individual_gradients_sample')
    
    seed_gen_sim_split   = tfp.random.split_seed( seed_gen_sim[0], n=iterations, salt='individual_gradients_sample_sim')
    seed_gen_grads_split = tfp.random.split_seed( seed_gen_grads[0], n=iterations, salt='individual_gradients_sample_grad')

    model = compartmental_model(N, simulation_likelihood_model.model.input_0, simulation_likelihood_model.model.input_kernel, SIS_initial, SIS_transition)

    def body(input, i):        
        
        q_0      = tf.convert_to_tensor([0.0, 0.0], dtype = tf.float32)   
        q_0_expanded      = tf.expand_dims(tf.expand_dims(q_0, axis = 0), axis = 0)
        q_static_expanded = tf.expand_dims(tf.expand_dims(parameters_emission[0], axis = 0), axis = 0)
        q_dynamic = tf.concat((q_0_expanded*tf.ones((1,N, M)), q_static_expanded*tf.ones((T, N, M))), axis =0)
        
        _, y_sim = sim(model, parameters_0, parameters_kernel, q_dynamic, T, seed_gen_sim_split[i][0])

        individuals_gradients_ = simba_gradients(simulation_likelihood_model, gradient_var, parameters_0, parameters_kernel, parameters_emission, y_sim, seed_gen_grads_split[i][0])

        return individuals_gradients_

    initialize = tf.zeros((N, tf.shape(gradient_var)[0]))
    individuals_gradients_ = tf.scan(body, tf.range(1, iterations), initializer = initialize)

    return individuals_gradients_

# if __name__ == "__main__":
    
# 	import numpy as np
# 	import tensorflow as tf
# 	import tensorflow_probability as tfp

# 	from tensorflow.keras import backend as K

# 	import sys
# 	sys.path.append('scripts/')
# 	from simulation_based_likelihood import *
# 	from synthetic_data import *
# 	from simba_gradient_univariate import *

# 	covariates_n = 2

# 	M = 2
# 	N = 300

# 	T = 100

# 	W_tensor = tf.convert_to_tensor(np.load("data/optimization/covariates.npy"), dtype = tf.float32)[:N,...]

# 	input_0      = tuple([W_tensor])
# 	input_kernel = tuple([W_tensor])

# 	model = compartmental_model(N, input_0, input_kernel, SIS_initial, SIS_transition)

# 	initial_infection_rate = 0.01
# 	beta_0      = tf.convert_to_tensor( [-np.log((1/initial_infection_rate)-1), +0], dtype = tf.float32 )
# 	beta_lambda = tf.convert_to_tensor( [-1,           +2],      dtype = tf.float32 )  
# 	beta_gamma  = tf.convert_to_tensor( [-1,           -1],      dtype = tf.float32 )  
# 	epsilon     = tf.convert_to_tensor( [0.001],                  dtype = tf.float32 )  

# 	parameters_0 = tuple([beta_0])
# 	parameters_kernel = beta_lambda, beta_gamma, epsilon

# 	q_0      = tf.convert_to_tensor([0.0, 0.0], dtype = tf.float32)   
# 	q_static = tf.convert_to_tensor([0.6, 0.4], dtype = tf.float32)

# 	q_0_expanded      = tf.expand_dims(tf.expand_dims(q_0, axis = 0), axis = 0)
# 	q_static_expanded = tf.expand_dims(tf.expand_dims(q_static, axis = 0), axis = 0)

# 	q_dynamic = tf.concat((q_0_expanded*tf.ones((1, N, M)), q_static_expanded*tf.ones((T, N, M))), axis =0)

# 	parameters_emission = tuple([q_static])

# 	parallel_simulations = 100

# 	optimization_steps = 200
# 	lr_rate = 0.1
# 	optimizer = tf.keras.optimizers.Adam(learning_rate = lr_rate)

# 	tf.random.set_seed((N+T))
# 	np.random.seed((N+T))

# 	seed_gen_optim_experiment = tfp.random.split_seed( (N+T), n=2, salt='optimization_SGD_N1000_T100')

# 	x, y = sim(model, parameters_0, parameters_kernel, q_dynamic, T, seed_gen_optim_experiment[1][0])

# 	simulation_likelihood_model = simulation_likelihood(parallel_simulations, N, input_0, input_kernel, 
# 							SIS_initial_grad, SIS_transition_grad, SIS_emission_grad,
# 							sampling = lambda probabilities, seed_cat: relaxed_categ_sampling(probabilities = probabilities, seed_cat = seed_cat, tau=0.5))

# 	initial_lambda = tf.Variable(tf.convert_to_tensor(np.random.uniform(-3, 3, 2), dtype = tf.float32))

# 	seed_init_cond_optim = tfp.random.split_seed( seed_gen_optim_experiment[0][0], n= optimization_steps+1, salt='optimization_SGD_N1000_T100_optim_steps')

# 	for epoch in range(optimization_steps):
# 		print("Epoch ", epoch)
# 		print("Parameters ", initial_lambda.numpy())

# 		seed_simba_loss, seed_optimization = tfp.random.split_seed( seed_init_cond_optim[epoch][0], n= 2, salt='optimization_SGD_N1000_T100_epoch')

# 		parameters_kernel_optim = initial_lambda, beta_gamma, epsilon

# 		gradient = simba_loss_grad(simulation_likelihood_model, initial_lambda, parameters_0, parameters_kernel_optim, parameters_emission, y, seed_optimization[0])

# 		# Apply gradients to update the parameters_var
# 		optimizer.apply_gradients(zip([gradient], [initial_lambda]))
