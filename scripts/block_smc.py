from distutils import log
from email import iterators
import numpy as np
from synthetic_data import *

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
        
def update(filtering_t_tm1, emission_t):
    
    update_t = (emission_t*filtering_t_tm1)

    likelihood_increment_t_tm1 =  tf.reduce_sum(update_t, axis = -1, keepdims = True)
    filtering_t = update_t/tf.where(update_t==0, tf.ones(tf.shape(likelihood_increment_t_tm1)), likelihood_increment_t_tm1)

    return likelihood_increment_t_tm1[...,0], filtering_t

def run_BBPF(simulation_likelihood_model, parameters_0, parameters_kernel, parameters_emission, y, seed_smc):
    
    T = tf.shape(y)[0]

    parameters_0_parallel        = simulation_likelihood_model.wrapper( parameters_0,        simulation_likelihood_model.parallel_simulations)
    parameters_kernel_parallel   = simulation_likelihood_model.wrapper( parameters_kernel,   simulation_likelihood_model.parallel_simulations)
    parameters_emission_parallel = simulation_likelihood_model.wrapper( parameters_emission, simulation_likelihood_model.parallel_simulations)

    seed_smc_split = tfp.random.split_seed( seed_smc, n=(T+1), salt='run_BBPF')

    def cond(input, t):

        return t<T

    def body(input, t):

        x_tm1, log_likelihood = input

        # simulate a new instance of the model
        _, x_t = sim_X_t(simulation_likelihood_model.model, parameters_kernel_parallel, x_tm1, seed_smc_split[t])

        # update as usual
        y_t = y[t,...]
        emission_t = simulation_likelihood_model.emission(parameters_emission_parallel, y_t)

        w_t = tf.reduce_sum(emission_t*x_t, axis =-1)
        norm_w_t = w_t/tf.reduce_sum(w_t, axis = 0, keepdims=True)

        indeces = tfp.distributions.Categorical( probs = tf.transpose(norm_w_t)).sample(tf.shape(w_t)[0])
        res_x_t = tf.transpose(tf.gather(tf.transpose(x_t, [1, 0, 2 ]), tf.transpose(indeces), axis = 1, batch_dims=1 ), [1, 0, 2 ])

        log_increment =  tf.reduce_sum(tf.math.log(tf.reduce_mean(w_t, axis =0)))
        
        return (res_x_t, log_likelihood + log_increment), t+1

    _, x_0     = sim_X_0(simulation_likelihood_model.model, parameters_0_parallel, seed_smc_split[0])
    log_likelihood = tf.zeros(1)

    output = tf.while_loop(cond, body, loop_vars = ((x_0, log_likelihood), 1))

    return output[0][1]


def run_BAPF(simulation_likelihood_model, parameters_0, parameters_kernel, parameters_emission, y, seed_smc):
    
    T = tf.shape(y)[0]

    parameters_0_parallel        = simulation_likelihood_model.wrapper( parameters_0,        simulation_likelihood_model.parallel_simulations)
    parameters_kernel_parallel   = simulation_likelihood_model.wrapper( parameters_kernel,   simulation_likelihood_model.parallel_simulations)
    parameters_emission_parallel = simulation_likelihood_model.wrapper( parameters_emission, simulation_likelihood_model.parallel_simulations)

    seed_smc_split = tfp.random.split_seed( seed_smc, n=(T+1), salt='run_BBPF')

    def cond(input, t):

        return t<T

    def body(input, t):

        x_tm1, log_likelihood = input

        # simulate a new instance of the model    
        transition_matrix_tm1, _ = sim_X_t(simulation_likelihood_model.model, parameters_kernel_parallel, x_tm1, seed_smc_split[t])
        prob_t = tf.einsum("...ni,...nik->...nk", x_tm1, transition_matrix_tm1)

        # update as usual
        y_t = y[t,...]
        emission_t = simulation_likelihood_model.emission(parameters_emission_parallel, y_t)

        likelihood_increment_t_tm1, filtering_t = update(prob_t, emission_t)

        x_t = tfp.distributions.OneHotCategorical( probs = filtering_t, dtype=tf.float32).sample()

        w_t = likelihood_increment_t_tm1
        norm_w_t = w_t/tf.reduce_sum(w_t, axis = 0, keepdims=True)

        indeces = tfp.distributions.Categorical( probs = tf.transpose(norm_w_t)).sample(tf.shape(w_t)[0])
        res_x_t = tf.transpose(tf.gather(tf.transpose(x_t, [1, 0, 2 ]), tf.transpose(indeces), axis = 1, batch_dims=1 ), [1, 0, 2 ])

        log_increment =  tf.reduce_sum(tf.math.log(tf.reduce_mean(w_t, axis =0)))
        
        return (res_x_t, log_likelihood + log_increment), t+1

    _, x_0     = sim_X_0(simulation_likelihood_model.model, parameters_0_parallel, seed_smc_split[0])
    log_likelihood = tf.zeros(1)

    output = tf.while_loop(cond, body, loop_vars = ((x_0, log_likelihood), 1))

    return output[0][1]




    