from distutils import log
from email import iterators
import numpy as np
from synthetic_data import *

def logit(p):

    return tf.math.log(p/(1-p))

# initial distribution for SIS
def SIS_initial_grad(input_0, parameters):

    W      = input_0[0]

    parameters_0 = tuple([parameters[0:2]])
    beta_0 = parameters_0[0]

    rate_0 = 1/(1+tf.exp(-tf.einsum("nj,...j->...n", W, beta_0)))
    
    initial_distr    = tf.stack((1-rate_0, rate_0), axis = -1)

    return initial_distr

# transition distribution for fully connected SIS
def SIS_transition_grad(input_kernel, parameters, x_tm1):

    W                                = input_kernel[0]

    parameters_kernel = parameters[2:4], parameters[4:6], tf.math.exp(parameters[6:7])
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

def SIS_emission_grad(parameters, y_t):

    parameters_emission = tuple([tf.math.sigmoid(parameters[7:9])])
    q_grad = parameters_emission[0]

    emission_from_observed = tf.expand_dims(y_t*tf.expand_dims(q_grad, axis =0), axis =0)
    emission_from_unobserved =  tf.expand_dims((1-tf.reduce_sum(y_t, axis = 1, keepdims = True))*(1-tf.expand_dims(q_grad, axis =0)), axis =0)

    return emission_from_observed + emission_from_unobserved

# def SIS_emission_grad_parallel(parameters, y_t_sim):

#     parameters_emission = tuple([tf.sigmoid(parameters[7:9])])
#     q_grad = parameters_emission[0]

#     emission_from_observed = tf.expand_dims(y_t_sim*tf.expand_dims(tf.expand_dims(q_grad, axis =0), axis =0), axis =1)

#     emission_from_unobserved =  tf.expand_dims((1-tf.reduce_sum(y_t_sim, axis = -1, keepdims = True))*(1-tf.expand_dims(tf.expand_dims(q_grad, axis =0), axis =0)), axis =1)

#     return emission_from_observed + emission_from_unobserved


# def sim_Y_grad(parameters, x_t_sim):

#     parameters_emission = tuple([tf.sigmoid(parameters[7:9])])
#     q_grad = parameters_emission[0]
#     q_grad = tf.expand_dims(tf.expand_dims(q_grad, axis =0), axis =0)

#     prob_be = x_t_sim*q_grad
#     be = tfp.distributions.Bernoulli( probs = prob_be, dtype = tf.float32)

#     return x_t_sim*be.sample()

# # initial distribution for SIS
# def SIS_initial_grad(input_0, parameters):

#     W      = input_0[0]

#     parameters_0 = tuple([parameters[:,0:2]])
#     beta_0 = parameters_0[0]

#     rate_0 = 1/(1+tf.exp(-tf.einsum("nj,...nj->...n", W, beta_0)))
    
#     initial_distr    = tf.stack((1-rate_0, rate_0), axis = -1)

#     return initial_distr

# # transition distribution for fully connected SIS
# def SIS_transition_grad(input_kernel, parameters, x_tm1):

#     W                                = input_kernel[0]

#     parameters_kernel = parameters[:,2:4], parameters[:,4:6], parameters[:,6:7]
#     beta_lambda, beta_gamma, epsilon = parameters_kernel

#     lambda__n = 1/(1+tf.exp(-tf.einsum("nj,...nj->...n", W, beta_lambda)))
#     gamma__n  = 1/(1+tf.exp(-tf.einsum("nj,...nj->...n", W, beta_gamma)))

#     c_tm1 = tf.reduce_sum( x_tm1, axis = -2, keepdims = True ) - x_tm1

#     N = tf.cast(tf.shape(W)[0], dtype = tf.float32)

#     rate_SI   = tf.expand_dims(tf.expand_dims(1-tf.exp(-lambda__n*((c_tm1[...,1]/N)+epsilon[:,0])), axis =-1), axis =-1)

#     rate_IS   = tf.expand_dims(tf.expand_dims(1-tf.exp(-gamma__n), axis =-1), axis =-1)*tf.ones(tf.shape(rate_SI))

#     K_eta_h__n_r1 = tf.concat((1 - rate_SI, rate_SI), axis = -1)
#     K_eta_h__n_r2 = tf.concat((rate_IS, 1 - rate_IS), axis = -1)
#     K_eta_h__n    = tf.concat((K_eta_h__n_r1, K_eta_h__n_r2), axis = -2)

#     return K_eta_h__n

# def SIS_emission_grad(parameters, y_t):

#     parameters_emission = tuple([parameters[:,7:8]])
#     q_grad = parameters_emission[0]

#     emission_from_observed = tf.expand_dims(y_t*q_grad, axis =0)
#     emission_from_unobserved =  tf.expand_dims((1-tf.reduce_sum(y_t, axis = 1, keepdims = True))*(1-q_grad), axis =0)

#     return emission_from_observed + emission_from_unobserved

def prediction(filtering_tm1, transition_matrix_tm1):

    return tf.einsum("sni,snij->snj", filtering_tm1, transition_matrix_tm1)

def update(filtering_t_tm1, emission_t):

    update_t = (emission_t*filtering_t_tm1)

    likelihood_increment_t_tm1 =  tf.reduce_sum(update_t, axis = -1, keepdims = True)
    filtering_t = update_t/tf.where(update_t==0, tf.ones(tf.shape(likelihood_increment_t_tm1)), likelihood_increment_t_tm1)

    return likelihood_increment_t_tm1[...,0], filtering_t

def categ_sampling(probabilities, seed_cat):
    
    return tfp.distributions.OneHotCategorical( probs = probabilities, dtype=tf.float32).sample(seed = seed_cat)

class simulation_likelihood():
        
    def __init__(self, parallel_simulations, N, input_0, input_kernel, initial_distribution, transition_kernel, emission_distribution):

        self.model = compartmental_model(N, input_0, input_kernel, initial_distribution, transition_kernel)
        self.emission = emission_distribution
        self.parallel_simulations = parallel_simulations

@tf.function
def run_simba(simulation_likelihood_model, parameters, y, seed_simba):

        T = tf.shape(y)[0]

        seed_gen_simba_split = tfp.random.split_seed( seed_simba, n=(T+1), salt='run_simba')

        def cond(input, t):

            return t<T

        def body(input, t):

            filtering_tm1, x_tm1_sim, log_likelihood = input

            transition_matrix_tm1 = simulation_likelihood_model.model.transition_kernel(simulation_likelihood_model.model.input_kernel, parameters, x_tm1_sim)
            prob_t = tf.einsum("...ni,...nik->...nk", x_tm1_sim, transition_matrix_tm1)

            x_t_sim = categ_sampling(prob_t, seed_gen_simba_split[t])

            # prediction on the filtering using the feedback
            filtering_t_tm1 = prediction(filtering_tm1, transition_matrix_tm1)

            # update as usual
            y_t = y[t,...]
            emission_t = simulation_likelihood_model.emission(parameters, y_t)
            likelihood_increment_t_tm1, filtering_t = update(filtering_t_tm1, emission_t)

            return (filtering_t, x_t_sim, log_likelihood + tf.math.log(likelihood_increment_t_tm1)), t+1

        probs_0 = simulation_likelihood_model.model.initial_distribution(simulation_likelihood_model.model.input_0, parameters)
        filtering_0 = tf.expand_dims(probs_0, axis = 0)*tf.ones(tf.concat(([simulation_likelihood_model.parallel_simulations], tf.shape(probs_0)), axis =0))

        x_0_sim = categ_sampling(filtering_0, seed_gen_simba_split[0])

        initializer = (filtering_0, x_0_sim, tf.zeros(tf.shape(x_0_sim)[:-1]))

        # Compute the function value
        output = tf.while_loop(cond, body, loop_vars = (initializer, 1))

        log_likelihood_individuals = output[0][2]
        
        max_log_likelihood_individuals = tf.reduce_mean(log_likelihood_individuals, axis = 0, keepdims = True) #-tf.cast(tf.shape(y)[0], dtype = tf.float32)
        log_likelihood = tf.math.log(tf.reduce_mean(tf.exp(log_likelihood_individuals - max_log_likelihood_individuals), axis =0)) + tf.reduce_mean(log_likelihood_individuals, axis = 0) # + max_log_likelihood_individuals #tf.math.log(tf.reduce_mean(tf.exp(log_likelihood_individuals), axis =0)) # 

        target = log_likelihood

        return target

@tf.function
def simba_loss(simulation_likelihood_model, parameters, y, seed_simba_loss):

    T = tf.cast(tf.shape(y)[0], dtype = tf.float32)

    log_likelihood_individuals = run_simba(simulation_likelihood_model, parameters, y, seed_simba_loss)

    return -tf.reduce_mean(log_likelihood_individuals)/T

@tf.function
def simba_loss_grad(simulation_likelihood_model, parameters, y, seed_simba_loss_grad):

    with tf.GradientTape() as tape:
        # Call your likelihood function with the parameters_var and other necessary inputs
        loss = simba_loss(simulation_likelihood_model, parameters, y, seed_simba_loss_grad)

    # Use TensorFlow's automatic differentiation to compute gradients
    return loss, tape.gradient(loss, [parameters])

def prediction_grad(filtering_tm1, transition_matrix_tm1):

    return tf.einsum("...sni,snij->...snj", filtering_tm1, transition_matrix_tm1)

# @tf.function
# def run_marginal_simba_body(simulation_likelihood_model, parameters, parameters_detached, filtering_tm1, x_tm1_sim, log_likelihood):

#     transition_matrix_tm1 = simulation_likelihood_model.model.transition_kernel(simulation_likelihood_model.model.input_kernel, parameters, x_tm1_sim)
#     prob_t = tf.einsum("...ni,...nik->...nk", x_tm1_sim, transition_matrix_tm1)

#     # prediction on the filtering using the feedback
#     filtering_t_tm1 = prediction_grad(filtering_tm1, transition_matrix_tm1)

#     x_t_sim = categ_sampling(prob_t)

#     # update as usual
#     y_t_sim = sim_Y_grad(parameters_detached, x_t_sim)

#     emission_t = simulation_likelihood_model.emission(parameters, y_t_sim)
#     likelihood_increment_t_tm1, filtering_t = update(filtering_t_tm1, emission_t)

#     return (filtering_t, x_t_sim, log_likelihood + tf.math.log(likelihood_increment_t_tm1))

# def run_marginal_simba(simulation_likelihood_model, parameters, T):

#         def cond(input, t):

#             return t<T

#         def body(input, t):

#             filtering_tm1, x_tm1_sim, log_likelihood = input

#             parameters_detached = tf.convert_to_tensor(parameters.numpy(), dtype = tf.float32)

#             return run_marginal_simba_body(simulation_likelihood_model, parameters, parameters_detached, filtering_tm1, x_tm1_sim, log_likelihood), t+1

#         probs_0 = simulation_likelihood_model.model.initial_distribution(simulation_likelihood_model.model.input_0, parameters)
#         filtering_0 = tf.expand_dims(probs_0, axis = 0)*tf.ones(tf.concat(([simulation_likelihood_model.parallel_simulations], tf.shape(probs_0)), axis =0))

#         x_0_sim = categ_sampling(filtering_0)

#         initializer = (filtering_0, x_0_sim, tf.zeros(tf.shape(filtering_0)[:-1]))

#         # Compute the function value
#         output = tf.while_loop(cond, body, loop_vars = (initializer, 1))

#         log_likelihood_individuals = output[0][2]
        
#         max_log_likelihood_individuals = tf.reduce_mean(log_likelihood_individuals, axis = 1, keepdims = True)
#         log_likelihood = (tf.math.log(tf.reduce_mean(tf.exp(log_likelihood_individuals - max_log_likelihood_individuals), axis =1)) + tf.reduce_mean(log_likelihood_individuals, axis = 1)) 

#         target = log_likelihood

#         return target

# @tf.function
# def compute_gradients(simulation_likelihood_model, parameters, y):

#     def body(gradient_im1, i):
    
#         with tf.GradientTape() as tape:
#             target = run_simba(simulation_likelihood_model, parameters, y)[i]
#         gradient_i = tape.gradient(target, parameters)

#         return gradient_i

#     with tf.GradientTape() as tape:
#         target = run_simba(simulation_likelihood_model, parameters, y)[0]
#     gradients_0 = tape.gradient(target, parameters)

#     gradients   = tf.scan(body, tf.range(0, tf.shape(y)[1]), initializer = gradients_0)

#     return gradients

@tf.function
def compute_gradients(simulation_likelihood_model, parameters, y, seed_simba_loss_grad):

    with tf.GradientTape() as tape:
        # tape.watch(parameters)
        target = run_simba(simulation_likelihood_model, parameters, y, seed_simba_loss_grad)

    return tape.jacobian(target, parameters)

@tf.function
def compute_gradient(simulation_likelihood_model, parameters, y, seed_simba_loss_grad):

    with tf.GradientTape() as tape:
        # tape.watch(parameters)
        target = tf.reduce_sum(run_simba(simulation_likelihood_model, parameters, y, seed_simba_loss_grad))

    return tape.jacobian(target, parameters)

@tf.function
def compute_hessian(simulation_likelihood_model, parameters, y, seed_simba_loss_grad):

    T = tf.cast(tf.shape(y)[0], dtype = tf.float32)

    with tf.GradientTape() as tape2:

        with tf.GradientTape() as tape1:  

            tape1.watch(parameters)

            log_likelihood_individuals = run_simba(simulation_likelihood_model, parameters, y, seed_simba_loss_grad)

            target = tf.reduce_sum(log_likelihood_individuals)

        gradient = tape1.gradient(target, parameters)
    
    hessian = tape2.jacobian(gradient, parameters)

    return gradient, hessian

# @tf.function
# def compute_gradient(simulation_likelihood_model, parameters, y):

#     with tf.GradientTape() as tape1:  

#         tape1.watch(parameters)

#         target = tf.reduce_sum(run_simba(simulation_likelihood_model, parameters, y))

#     gradient = tape1.gradient(target, parameters)

#     return gradient

# @tf.function
# def expected_gradient_hessian(simulation_likelihood_model, parameters, y, iterations):
    
#     T = tf.shape(y)[0]
#     N = tf.shape(y)[1]
#     M = tf.shape(y)[2]

#     model = compartmental_model(N, simulation_likelihood_model.model.input_0, simulation_likelihood_model.model.input_kernel, SIS_initial, SIS_transition)

#     individuals_gradients_0 = compute_gradients(simulation_likelihood_model, parameters, y)

#     gradient_0, hessian_0 = compute_gradient_hessian(simulation_likelihood_model, parameters, y)

#     def body(input, i):
        
#         parameters_0 = tuple([parameters[0:2]])
#         parameters_kernel = parameters[2:4], parameters[4:6], parameters[6:7]
#         parameters_emission = tuple([parameters[7:9]])

#         q_0      = tf.convert_to_tensor([0.0, 0.0], dtype = tf.float32)   
#         q_0_expanded      = tf.expand_dims(tf.expand_dims(q_0, axis = 0), axis = 0)
#         q_static_expanded = tf.expand_dims(tf.expand_dims(parameters_emission[0], axis = 0), axis = 0)
#         q_dynamic = tf.concat((q_0_expanded*tf.ones((1,N, M)), q_static_expanded*tf.ones((T, N, M))), axis =0)

#         x_sim, y_sim = sim(model, parameters_0, parameters_kernel, q_dynamic, T)

#         while tf.reduce_sum(tf.reduce_sum(x_sim[:,...], axis =0), axis =0)[1]<10:
#             x_sim, y_sim = sim(model, parameters_0, parameters_kernel, q_dynamic, T)

#         individuals_gradients_sim = compute_gradients(simulation_likelihood_model, parameters, y_sim)

#         gradient_sim, hessian_sim = compute_gradient_hessian(simulation_likelihood_model, parameters, y_sim)

#         return individuals_gradients_sim, gradient_sim, hessian_sim


#     initialize = (individuals_gradients_0, gradient_0, hessian_0)

#     individuals_gradients, gradient, hessian = tf.scan(body, tf.range(1, iterations), initializer = initialize)

#     return tf.concat((tf.expand_dims(individuals_gradients_0, axis =0), individuals_gradients), axis =0 ), tf.concat((tf.expand_dims(gradient_0, axis =0), gradient), axis =0 ), tf.concat((tf.expand_dims(hessian_0, axis =0), hessian), axis =0 )

def expected_gradient(simulation_likelihood_model, parameters, T, N, M, iterations, seed_simba_loss_expected_grad):

    seed_gen_sim, seed_gen_grad = tfp.random.split_seed( seed_simba_loss_expected_grad, n=2, salt='expected_gradient')

    seed_gen_sim_split = tfp.random.split_seed( seed_gen_sim[0], n=iterations, salt='expected_gradient_sim')
    seed_gen_grad_split = tfp.random.split_seed( seed_gen_grad[0], n=iterations, salt='expected_gradient_grad')
    
    model = compartmental_model(N, simulation_likelihood_model.model.input_0, simulation_likelihood_model.model.input_kernel, SIS_initial, SIS_transition)

    def body(input, i):
        
        parameters_0 = tuple([parameters[0:2]])
        parameters_kernel = parameters[2:4], parameters[4:6], parameters[6:7]
        parameters_emission = tuple([parameters[7:9]])

        q_0      = tf.convert_to_tensor([0.0, 0.0], dtype = tf.float32)   
        q_0_expanded      = tf.expand_dims(tf.expand_dims(q_0, axis = 0), axis = 0)
        q_static_expanded = tf.expand_dims(tf.expand_dims(parameters_emission[0], axis = 0), axis = 0)
        q_dynamic = tf.concat((q_0_expanded*tf.ones((1,N, M)), q_static_expanded*tf.ones((T, N, M))), axis =0)

        _, y_sim = sim(model, parameters_0, parameters_kernel, q_dynamic, T, seed_gen_sim_split[i][0])

        individuals_gradients_sim = compute_gradients(simulation_likelihood_model, parameters, y_sim, seed_gen_grad_split[i][0])

        return individuals_gradients_sim

    initialize = tf.zeros((N, tf.shape(parameters)[0]))
    individuals_gradients = tf.scan(body, tf.range(1, iterations), initializer = initialize)

    return individuals_gradients

# @tf.function
# def compute_gradients_num(simulation_likelihood_model, parameters, y, epsilon = 0.1):

#     def body(input, i):

#         one_hot = tf.one_hot(i, tf.shape(parameters)[0])

#         loglike_p = run_simba(simulation_likelihood_model, parameters+epsilon*one_hot, y)
#         loglike_m = run_simba(simulation_likelihood_model, parameters-epsilon*one_hot, y)

#         grad = (loglike_p-loglike_m)/(2*epsilon)

#         return grad

#     initialize = run_simba(simulation_likelihood_model, parameters, y)

#     return tf.transpose(tf.scan(body, tf.range(0, tf.shape(parameters)[0]), initializer = initialize))

# @tf.function
# def compute_sub_gradients_num(simulation_likelihood_model, sub_index, parameters, y, epsilon = 0.1):

#     def body(input, i):

#         one_hot = tf.one_hot(i, tf.shape(parameters)[0])

#         loglike_p = run_simba(simulation_likelihood_model, parameters+epsilon*one_hot, y)
#         loglike_m = run_simba(simulation_likelihood_model, parameters-epsilon*one_hot, y)

#         grad = (loglike_p-loglike_m)/(2*epsilon)

#         return grad

#     initialize = run_simba(simulation_likelihood_model, parameters, y)

#     return tf.transpose(tf.scan(body, sub_index, initializer = initialize))

# @tf.function
# def expected_gradient_numerical(simulation_likelihood_model, parameters, T, N, M, iterations, epsilon = 0.1):
    
#     model = compartmental_model(N, simulation_likelihood_model.model.input_0, simulation_likelihood_model.model.input_kernel, SIS_initial, SIS_transition)

#     def body(input, i):
        
#         parameters_0 = tuple([parameters[0:2]])
#         parameters_kernel = parameters[2:4], parameters[4:6], parameters[6:7]
#         parameters_emission = tuple([parameters[7:9]])

#         q_0      = tf.convert_to_tensor([0.0, 0.0], dtype = tf.float32)   
#         q_0_expanded      = tf.expand_dims(tf.expand_dims(q_0, axis = 0), axis = 0)
#         q_static_expanded = tf.expand_dims(tf.expand_dims(parameters_emission[0], axis = 0), axis = 0)
#         q_dynamic = tf.concat((q_0_expanded*tf.ones((1,N, M)), q_static_expanded*tf.ones((T, N, M))), axis =0)

#         x_sim, y_sim = sim(model, parameters_0, parameters_kernel, q_dynamic, T)

#         while tf.reduce_sum(tf.reduce_sum(x_sim[:,...], axis =0), axis =0)[1]<10:
#             x_sim, y_sim = sim(model, parameters_0, parameters_kernel, q_dynamic, T)

#         individuals_gradients_sim = compute_gradients_num(simulation_likelihood_model, parameters, y_sim, epsilon)

#         return individuals_gradients_sim

#     initialize = tf.zeros((N, tf.shape(parameters)[0]))
#     individuals_gradients = tf.scan(body, tf.range(1, iterations), initializer = initialize)

#     return individuals_gradients

# @tf.function
# def expected_sub_gradient_numerical(simulation_likelihood_model, sub_index, parameters, T, N, M, iterations, epsilon = 0.1):
    
#     model = compartmental_model(N, simulation_likelihood_model.model.input_0, simulation_likelihood_model.model.input_kernel, SIS_initial, SIS_transition)

#     def body(input, i):
        
#         parameters_0 = tuple([parameters[0:2]])
#         parameters_kernel = parameters[2:4], parameters[4:6], parameters[6:7]
#         parameters_emission = tuple([parameters[7:9]])

#         q_0      = tf.convert_to_tensor([0.0, 0.0], dtype = tf.float32)   
#         q_0_expanded      = tf.expand_dims(tf.expand_dims(q_0, axis = 0), axis = 0)
#         q_static_expanded = tf.expand_dims(tf.expand_dims(parameters_emission[0], axis = 0), axis = 0)
#         q_dynamic = tf.concat((q_0_expanded*tf.ones((1,N, M)), q_static_expanded*tf.ones((T, N, M))), axis =0)

#         x_sim, y_sim = sim(model, parameters_0, parameters_kernel, q_dynamic, T)

#         while tf.reduce_sum(tf.reduce_sum(x_sim[:,...], axis =0), axis =0)[1]<10:
#             x_sim, y_sim = sim(model, parameters_0, parameters_kernel, q_dynamic, T)

#         individuals_gradients_sim = compute_sub_gradients_num(simulation_likelihood_model, sub_index, parameters, y_sim, epsilon)

#         return individuals_gradients_sim

#     initialize = tf.zeros((N, tf.shape(sub_index)[0]))
#     individuals_gradients = tf.scan(body, tf.range(1, iterations), initializer = initialize)

#     return individuals_gradients

def expected_hessian(simulation_likelihood_model, parameters, T, N, M, iterations, seed_gen_exp_hess):

    seed_gen_sim, seed_gen_hessian = tfp.random.split_seed( seed_gen_exp_hess, n=2, salt='expected_hessian')

    seed_gen_sim_split = tfp.random.split_seed( seed_gen_sim[0], n=iterations, salt='expected_gradient_sim')
    seed_gen_grad_split = tfp.random.split_seed( seed_gen_hessian[0], n=iterations, salt='expected_gradient_grad')
    
    model = compartmental_model(N, simulation_likelihood_model.model.input_0, simulation_likelihood_model.model.input_kernel, SIS_initial, SIS_transition)

    def body(input, i):
        
        parameters_0 = tuple([parameters[0:2]])
        parameters_kernel = parameters[2:4], parameters[4:6], parameters[6:7]
        parameters_emission = tuple([parameters[7:9]])

        q_0      = tf.convert_to_tensor([0.0, 0.0], dtype = tf.float32)   
        q_0_expanded      = tf.expand_dims(tf.expand_dims(q_0, axis = 0), axis = 0)
        q_static_expanded = tf.expand_dims(tf.expand_dims(parameters_emission[0], axis = 0), axis = 0)
        q_dynamic = tf.concat((q_0_expanded*tf.ones((1,N, M)), q_static_expanded*tf.ones((T, N, M))), axis =0)

        _, y_sim = sim(model, parameters_0, parameters_kernel, q_dynamic, T, seed_gen_sim_split[i][0])

        gradient_sim, hessian_sim = compute_hessian(simulation_likelihood_model, parameters, y_sim, seed_gen_grad_split[i][0])

        return gradient_sim, hessian_sim

    initialize = tf.zeros((tf.shape(parameters)[0])), tf.zeros((tf.shape(parameters)[0], tf.shape(parameters)[0]))

    gradient, hessian = tf.scan(body, tf.range(1, iterations), initializer = initialize)

    return gradient, hessian
    
# def compute_gradients_hessians(simulation_likelihood_model, parameters, y):

#     def body(input, i):
    
#         with tf.GradientTape() as tape2:

#             with tf.GradientTape() as tape1:  

#                 tape1.watch(parameters)

#                 target = run_simba(simulation_likelihood_model, parameters, y)[i]

#             gradient_i = tape1.gradient(target, parameters)
        
#         hessian_i = tape2.jacobian(gradient_i, parameters)

#         return gradient_i, hessian_i

#     with tf.GradientTape() as tape2:

#         with tf.GradientTape() as tape1:  

#             tape1.watch(parameters)

#             target = run_simba(simulation_likelihood_model, parameters, y)[0]

#         gradient_0 = tape1.gradient(target, parameters)
    
#     hessian_0 = tape2.jacobian(gradient_0, parameters)

#     gradients, hessians  = tf.scan(body, tf.range(0, tf.shape(y)[1]), initializer = (gradient_0, hessian_0))

#     return gradients, hessians

# def expected_gradient(simulation_likelihood_model, parameters, T):

#     with tf.GradientTape() as tape:

#         target = run_marginal_simba(simulation_likelihood_model, parameters, T)

#     return tape.jacobian(target, parameters)

@tf.function
def V_computation(centered_gradients):
    
    N = tf.shape(centered_gradients)[1]
    
    def cond(input, i):
    
        return i<N

    def body(input, i):

        cum_V = input

        sub_V = tf.reduce_sum(tf.reduce_mean(tf.einsum("sp,snq->snpq", centered_gradients[:,i,:], centered_gradients), axis = 0), axis = 0)

        return sub_V + cum_V, i +1

    sub_V = tf.reduce_sum(tf.reduce_mean(tf.einsum("sp,snq->snpq", centered_gradients[:,0,:], centered_gradients), axis = 0), axis = 0)

    output = tf.while_loop(cond, body, loop_vars = (sub_V, 1))

    return output[0]