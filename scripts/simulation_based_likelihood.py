from distutils import log
from email import iterators
import numpy as np
from synthetic_data import *

def SIS_joint_kernel(input_kernel, parameters_kernel, partition, x_tm1):

    indeces_mixup = tf.reshape(tf.stack((partition[:,1], partition[:,0]), axis =1), tf.shape(x_tm1)[1])
    
    W                       = input_kernel[0]
    beta_lambda, beta_gamma, epsilon = parameters_kernel

    N = tf.cast(tf.shape(W)[0], dtype = tf.float32)

    lambda__n = tf.expand_dims(tf.expand_dims(1/(1+tf.exp(-tf.einsum("ij,...j->...i", W, beta_lambda))), axis = 0), axis = -1)
    gamma__n  = tf.expand_dims(tf.expand_dims(1/(1+tf.exp(-tf.einsum("ij,...j->...i", W, beta_gamma))), axis = 0), axis = -1)

    c_tm1 = tf.reduce_sum( x_tm1, axis = -2, keepdims = True ) - x_tm1 - tf.gather(x_tm1, indeces_mixup, axis =1)

    c_tm1_base_even = tf.expand_dims(tf.gather(c_tm1, partition[:,0], axis = 1), axis= -2)
    c_tm1_base_odd  = tf.expand_dims(tf.gather(c_tm1, partition[:,1], axis = 1), axis= -2)
    
    add_S = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor([1, 0], dtype = tf.float32), axis =0), axis =0)
    add_I = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor([0, 1], dtype = tf.float32), axis =0), axis =0)

    c_tm1_base_even_S = c_tm1_base_even + tf.expand_dims(add_S, axis = -2)
    c_tm1_base_even_I = c_tm1_base_even + tf.expand_dims(add_I, axis = -2)

    c_tm1_base_S_odd = c_tm1_base_odd + tf.expand_dims(add_S, axis = -2)
    c_tm1_base_I_odd = c_tm1_base_odd + tf.expand_dims(add_I, axis = -2)

    c_tm1_n_barn_even = tf.concat((c_tm1_base_even_S, c_tm1_base_even_I), axis = -2)
    c_tm1_n_barn_odd  = tf.concat((c_tm1_base_S_odd, c_tm1_base_I_odd), axis = -2)

    rate_SI_even = tf.expand_dims(tf.expand_dims(1-tf.exp(-(tf.gather(lambda__n, partition[:,0], axis = 1)*((c_tm1_n_barn_even[...,1]/N)+epsilon))), axis = -1), axis =-1)
    rate_SI_odd  = tf.expand_dims(tf.expand_dims(1-tf.exp(-(tf.gather(lambda__n, partition[:,1], axis = 1)*((c_tm1_n_barn_odd[...,1]/N)+epsilon))), axis = -1), axis =-1)

    rate_IS_even = tf.expand_dims(tf.expand_dims(1-tf.exp(-tf.gather(gamma__n, partition[:,0], axis = 1)), axis = -1), axis =-1)*tf.ones(tf.shape(rate_SI_even))    
    rate_IS_odd  = tf.expand_dims(tf.expand_dims(1-tf.exp(-tf.gather(gamma__n, partition[:,1], axis = 1)), axis = -1), axis =-1)*tf.ones(tf.shape(rate_SI_odd))

    K_eta_h__n_r1_even = tf.concat((1 - rate_SI_even, rate_SI_even), axis = -1)
    K_eta_h__n_r2_even = tf.concat((rate_IS_even, 1 - rate_IS_even), axis = -1)
    K_eta_h__n_even    = tf.concat((K_eta_h__n_r1_even, K_eta_h__n_r2_even), axis = -2)

    K_eta_h__n_r1_odd = tf.concat((1 - rate_SI_odd, rate_SI_odd), axis = -1)
    K_eta_h__n_r2_odd = tf.concat((rate_IS_odd, 1 - rate_IS_odd), axis = -1)
    K_eta_h__n_odd    = tf.concat((K_eta_h__n_r1_odd, K_eta_h__n_r2_odd), axis = -2)

    K_eta_h__n_even = tf.transpose(tf.expand_dims(K_eta_h__n_even,  axis = -1), [0,1,3,2,4,5])
    K_eta_h__n_odd  = tf.expand_dims(K_eta_h__n_odd, axis = -2)

    return K_eta_h__n_odd*K_eta_h__n_even

def SIS_emission(parameters_emission_parallel, y_t):
    
    q_t = parameters_emission_parallel[0]

    emission_from_observed = tf.expand_dims(y_t, axis = 0)*tf.expand_dims(q_t, axis =1)
    emission_from_unobserved =  (1-tf.expand_dims(tf.reduce_sum(y_t, axis = 1, keepdims = True), axis =0))*tf.expand_dims(1-q_t, axis =1)

    return emission_from_observed + emission_from_unobserved

class simulation_likelihood():
    
    def __init__(self, parallel_simulations, N, input_0, input_kernel, initial_distribution, transition_kernel, emission_distribution):

        self.model = compartmental_model(N, input_0, input_kernel, initial_distribution, transition_kernel)
        self.emission = emission_distribution
        self.parallel_simulations = parallel_simulations

    def wrapper(self, parameters, wrapping_size):

        parameters_parallel_list = []

        for parameters_i in parameters:

            parameters_i_ones = np.ones(tf.concat(([wrapping_size], tf.ones(tf.shape(tf.shape(parameters_i))[0], dtype = tf.int64)), axis =0))

            parameters_parallel_list.append(tf.expand_dims(parameters_i, axis = 0 )*parameters_i_ones)
    
        return tuple(parameters_parallel_list)   

@tf.function(jit_compile=True)
def log_feedback_emission(simulation_likelihood_model, x_tm1, x_t, parameters_kernel_parallel_wrapped):

    # define the depth of the bar_x_tm1, we want the compartment size as initial dimension
    initial_shape_bar_x_tm1 = tf.shape(tf.expand_dims(tf.expand_dims(x_tm1, axis = 0), axis = 0))
    M = initial_shape_bar_x_tm1[-1]
    N = initial_shape_bar_x_tm1[-2]
    depth = tf.shape(initial_shape_bar_x_tm1)
    ones_shape = initial_shape_bar_x_tm1 + tf.one_hot(indices = (1), depth = depth[0], on_value=(M-1)) + tf.one_hot(indices = (0), depth = depth[0], on_value=(N-1))

    # create M copies of the initial vector
    to_fill_bar_x_tm1 = tf.expand_dims(tf.expand_dims(x_tm1, axis = 0), axis = 0)*tf.ones(ones_shape)

    # separate the nth element from the others
    null_n = 1-tf.one_hot(indices = tf.range(0, N), depth = tf.shape(x_tm1)[-2])

    # place all zeros in the nth position
    bar_x_tm1_minus_n = tf.einsum("Nn,N...nm->N...nm", null_n, to_fill_bar_x_tm1)

    # place all ones in the nth position and all zeros in the others
    ones_bar_x_tm1_n = tf.einsum("Nn,N...nm->N...nm", (1-null_n), tf.ones(tf.shape(to_fill_bar_x_tm1)))
    # define all the possible state of x_tm1_n
    bar_x_tm1_n_unidim = tf.eye(tf.shape(x_tm1)[-1])
    # define the nth position in the vector of zeros
    bar_x_tm1_n = tf.einsum("Mm,NM...nm->NM...nm", bar_x_tm1_n_unidim, ones_bar_x_tm1_n)

    # create a vector with all the possible states in the nth positions and copies of the others 
    bar_x_tm1 = bar_x_tm1_n + bar_x_tm1_minus_n

    # compute all the possible transitions for all the possible states
    bar_transition_matrix_tm1 = simulation_likelihood_model.model.transition_kernel(simulation_likelihood_model.model.input_kernel, parameters_kernel_parallel_wrapped, bar_x_tm1)

    # select the transition in x_tm1
    p_given_bar_x_tm1         = tf.einsum("snM,N...snMm->N...snm", x_tm1, bar_transition_matrix_tm1)
    # select the transition in x_t
    p_given_bar_x_tm1_bar_x_t = tf.einsum("snm,N...snm->N...sn",   x_t,   p_given_bar_x_tm1)

    # feedback_emission_t = (tf.reduce_prod(p_given_bar_x_tm1_bar_x_t, axis =-1)/tf.einsum("N...N->N...", p_given_bar_x_tm1_bar_x_t))
    log_feedback_emission_t = tf.reduce_sum(tf.math.log(p_given_bar_x_tm1_bar_x_t), axis =-1)-tf.math.log(tf.einsum("N...N->N...", p_given_bar_x_tm1_bar_x_t))

    return tf.einsum("nms->snm", log_feedback_emission_t)

def feedback_update(feedback_filtering_tm1, log_feedback_emission_t):

    log_feedback_update_t = log_feedback_emission_t+tf.math.log(feedback_filtering_tm1)
    max_log_feedback_update_t = tf.reduce_max(log_feedback_update_t, axis = -1, keepdims = True)

    feedback_denominator_normalized = tf.reduce_sum(tf.math.exp(log_feedback_update_t - max_log_feedback_update_t), axis = -1, keepdims = True)
    
    log_feedback_tm1 =  log_feedback_emission_t - max_log_feedback_update_t - tf.math.log(feedback_denominator_normalized)
    log_feedback_filtering_tm1_t = log_feedback_update_t - max_log_feedback_update_t - tf.math.log(feedback_denominator_normalized)

    return tf.math.exp(log_feedback_tm1), tf.math.exp(log_feedback_filtering_tm1_t)

def feedback_prediction(transition_matrix_tm1, feedback_filtering_tm1):

    return tf.einsum("sni,snij->snj", feedback_filtering_tm1, transition_matrix_tm1)

def prediction(filtering_tm1, transition_matrix_tm1):

    return tf.einsum("sni,snij->snj", filtering_tm1, transition_matrix_tm1)

def update(filtering_t_tm1, emission_t):

    update_t = (emission_t*filtering_t_tm1)

    likelihood_increment_t_tm1 =  tf.reduce_sum(update_t, axis = -1, keepdims = True)
    filtering_t = update_t/tf.where(update_t==0, tf.ones(tf.shape(likelihood_increment_t_tm1)), likelihood_increment_t_tm1)

    return likelihood_increment_t_tm1[...,0], filtering_t


def run_simulation_likelihood(simulation_likelihood_model, parameters_0, parameters_kernel, parameters_emission, y, seed_simba):
    
    T = tf.shape(y)[0]

    parameters_0_parallel        = simulation_likelihood_model.wrapper( parameters_0,        simulation_likelihood_model.parallel_simulations)
    parameters_kernel_parallel   = simulation_likelihood_model.wrapper( parameters_kernel,   simulation_likelihood_model.parallel_simulations)
    parameters_emission_parallel = simulation_likelihood_model.wrapper( parameters_emission, simulation_likelihood_model.parallel_simulations)

    seed_simba_split = tfp.random.split_seed( seed_simba, n=(T+1), salt='run_simulation_likelihood')

    def cond(input, t):

        return t<T

    def body(input, t):

        filtering_tm1, feedback_filtering_tm1, x_tm1, log_likelihood = input

        # simulate a new instance of the model
        transition_matrix_tm1, x_t = sim_X_t(simulation_likelihood_model.model, parameters_kernel_parallel, x_tm1, seed_simba_split[t])

        # add the extra compartmental dimension needed to compute the feedback
        parameters_kernel_parallel_wrapped = simulation_likelihood_model.wrapper(parameters_kernel_parallel, tf.shape(x_tm1)[-1])
        parameters_kernel_parallel_wrapped = simulation_likelihood_model.wrapper(parameters_kernel_parallel_wrapped, tf.shape(x_tm1)[-2])

        # compute the likelihood score from x_t given x_tm1, and use that to compute the feedback
        log_feedback_emission_t = log_feedback_emission(simulation_likelihood_model, x_tm1, x_t, parameters_kernel_parallel_wrapped)
        feedback_tm1, feedback_filtering_tm1_t = feedback_update(feedback_filtering_tm1, log_feedback_emission_t)

        # complete the prediction on the feedback filtering
        feedback_filtering_t = feedback_prediction(transition_matrix_tm1, feedback_filtering_tm1_t)

        # update filtering with feedback
        feedback_likelihood_increment_t_tm1, filtering_t_after_feedback = update(filtering_tm1, feedback_tm1)

        # prediction on the filtering using the feedback
        filtering_t_tm1 = prediction(filtering_t_after_feedback, transition_matrix_tm1)

        # update as usual
        y_t = y[t,...]
        emission_t = simulation_likelihood_model.emission(parameters_emission_parallel, y_t)
        likelihood_increment_t_tm1, filtering_t = update(filtering_t_tm1, emission_t)

        return (filtering_t, feedback_filtering_t, x_t, log_likelihood + tf.math.log(likelihood_increment_t_tm1) + tf.math.log(feedback_likelihood_increment_t_tm1)), t+1

    filtering_0, x_0     = sim_X_0(simulation_likelihood_model.model, parameters_0_parallel, seed_simba_split[0])
    feedback_filtering_0 = filtering_0

    log_likelihood = tf.zeros(tf.shape(filtering_0)[:-1])

    output = tf.while_loop(cond, body, loop_vars = ((filtering_0, feedback_filtering_0, x_0, log_likelihood), 1))

    return output[0][3]

def run_simulation_likelihood_approx(simulation_likelihood_model, parameters_0, parameters_kernel, parameters_emission, y, seed_simba):
    
    T = tf.shape(y)[0]

    parameters_0_parallel        = simulation_likelihood_model.wrapper( parameters_0,        simulation_likelihood_model.parallel_simulations)
    parameters_kernel_parallel   = simulation_likelihood_model.wrapper( parameters_kernel,   simulation_likelihood_model.parallel_simulations)
    parameters_emission_parallel = simulation_likelihood_model.wrapper( parameters_emission, simulation_likelihood_model.parallel_simulations)

    seed_simba_split = tfp.random.split_seed( seed_simba, n=(T+1), salt='run_simulation_likelihood_approx')

    def cond(input, t):

        return t<T

    def body(input, t):

        filtering_tm1, x_tm1, log_likelihood = input

        # simulate a new instance of the model
        transition_matrix_tm1, x_t = sim_X_t(simulation_likelihood_model.model, parameters_kernel_parallel, x_tm1, seed_simba_split[t])

        # prediction on the filtering using the feedback
        filtering_t_tm1 = prediction(filtering_tm1, transition_matrix_tm1)

        # update as usual
        y_t = y[t,...]
        emission_t = simulation_likelihood_model.emission(parameters_emission_parallel, y_t)
        likelihood_increment_t_tm1, filtering_t = update(filtering_t_tm1, emission_t)

        return (filtering_t, x_t, log_likelihood + tf.math.log(likelihood_increment_t_tm1)), t+1

    filtering_0, x_0     = sim_X_0(simulation_likelihood_model.model, parameters_0_parallel, seed_simba_split[0])
    log_likelihood = tf.zeros(tf.shape(filtering_0)[:-1])

    output = tf.while_loop(cond, body, loop_vars = ((filtering_0, x_0, log_likelihood), 1))

    return output[0][2]


def run_simulation_likelihood_coupled(simulation_likelihood_model, parameters_0, parameters_kernel, parameters_emission, y, seed_simba):
    
    T = tf.shape(y)[0]

    parameters_0_parallel        = simulation_likelihood_model.wrapper( parameters_0,        simulation_likelihood_model.parallel_simulations)
    parameters_kernel_parallel   = simulation_likelihood_model.wrapper( parameters_kernel,   simulation_likelihood_model.parallel_simulations)
    parameters_emission_parallel = simulation_likelihood_model.wrapper( parameters_emission, simulation_likelihood_model.parallel_simulations)

    seed_simba_split = tfp.random.split_seed( seed_simba, n=(T+1), salt='run_simulation_likelihood_coupled')

    def cond(input, t):

        return t<T

    def body(input, t):

        coupled_filtering_tm1, x_tm1, log_likelihood = input

        # simulate a new instance of the model
        _, x_t = sim_X_t(simulation_likelihood_model.model, parameters_kernel_parallel, x_tm1, seed_simba_split[t])
        coupled_transition_matrix_tm1 = SIS_joint_kernel(simulation_likelihood_model.model.input_kernel, parameters_kernel, partition, x_tm1)

        # prediction on the filtering using the feedback
        coupled_filtering_t_tm1 = tf.einsum("...ijkl,...ij->...kl", coupled_transition_matrix_tm1, coupled_filtering_tm1)

        # update as usual
        y_t = y[t,...]
        emission_t = SIS_emission(parameters_emission_parallel, y_t)
                
        emission_t_even = tf.gather(emission_t, indices_even, axis =1)
        emission_t_odd  = tf.gather(emission_t, indices_odd,  axis =1)

        coupled_emission_t = tf.expand_dims(emission_t_even, axis = -1)*tf.expand_dims(emission_t_odd, axis = -2)

        normalization = tf.reduce_sum(coupled_emission_t*coupled_filtering_t_tm1, axis = [-2,-1], keepdims=True)
        likelihood_increment_t_tm1 = normalization[...,0,0]
        coupled_filtering_t = coupled_emission_t*coupled_filtering_t_tm1/tf.where((coupled_emission_t*coupled_filtering_t_tm1)==0, tf.ones(tf.shape(normalization)), normalization)

        return (coupled_filtering_t, x_t, log_likelihood + tf.math.log(likelihood_increment_t_tm1)), t+1

    filtering_0, x_0     = sim_X_0(simulation_likelihood_model.model, parameters_0_parallel, seed_simba_split[0])

    # Select every other index along the second axis
    indices_even = tf.range(start=0, limit=tf.shape(filtering_0)[1], delta=2)
    indices_odd  = tf.range(start=1, limit=tf.shape(filtering_0)[1], delta=2)

    filtering_0_even = tf.gather(filtering_0, indices_even,  axis =1)
    filtering_0_odd  = tf.gather(filtering_0, indices_odd, axis =1)

    partition = tf.stack((indices_even, indices_odd), axis =1)

    coupled_filtering_0 = tf.expand_dims(filtering_0_even, axis = -1)*tf.expand_dims(filtering_0_odd, axis = -2)
    log_likelihood = tf.zeros(tf.shape(coupled_filtering_0)[:-2])

    output = tf.while_loop(cond, body, loop_vars = ((coupled_filtering_0, x_0, log_likelihood), 1))

    return output[0][2]





    