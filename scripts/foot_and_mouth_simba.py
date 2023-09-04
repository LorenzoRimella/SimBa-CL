import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import time

import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# initial distribution for FM
def FM_initial(input_0, parameters_0):
    
    n_cattles, n_sheep     = input_0
    probs_0 = parameters_0

    probs_0 = probs_0*tf.ones(tf.shape(tf.transpose(n_cattles)))
    
    initial_distr    = tf.stack((1-probs_0, probs_0, tf.zeros(tf.shape(probs_0)), tf.zeros(tf.shape(probs_0))), axis = -1)

    return initial_distr
    
@tf.function(jit_compile=True)
def sparse_infect_matrix_norm(psi, farms_coord):

    batch_distances = (tf.square(farms_coord[:,0:1] - tf.transpose(farms_coord)[0:1,:]) + tf.square(farms_coord[:,1:2] - tf.transpose(farms_coord)[1:2,:]))
    weighted_distance =  (1/(2*psi*psi*1000*1000))*batch_distances
    distance_matrix   = tf.math.exp(-weighted_distance)
    # distance_matrix = psi/(psi*psi + batch_distances/(1000*1000)) 

    return distance_matrix/tf.reduce_sum(distance_matrix, axis = 1, keepdims = True)

def FM_transition(input_kernel, parameters_kernel, x_tm1):
    
    n_cattles, n_sheep = input_kernel

    delta, beta_lambda_1, beta_lambda_2, kernel_matrix, beta_gamma_1, beta_gamma_2, mean_infection_time, epsilon = parameters_kernel

    susceptibility = delta/((1 + tf.math.exp(-(beta_lambda_1*tf.transpose(n_cattles) + beta_lambda_2*tf.transpose(n_sheep)))))

    infectiousness_after_kernel = tf.einsum("ij,si->sj", kernel_matrix, x_tm1[...,1])

    rate_SI = susceptibility*infectiousness_after_kernel + epsilon
    prob_SI = rate_SI

    prob_IN = 1/(mean_infection_time*(1 + tf.math.exp(-(beta_gamma_1*tf.transpose(n_cattles) + beta_gamma_2*tf.transpose(n_sheep)))))

    prob_NR = tf.zeros(tf.shape(prob_SI))

    K_eta_h__n_r1 = tf.stack((1 -               prob_SI  ,                   prob_SI,       tf.zeros(tf.shape(prob_SI)),     tf.zeros(tf.shape(prob_SI))), axis = -1)
    K_eta_h__n_r2 = tf.stack((tf.zeros(tf.shape(prob_SI)), 1 -               prob_IN,                           prob_IN,     tf.zeros(tf.shape(prob_SI))), axis = -1)
    K_eta_h__n_r3 = tf.stack((tf.zeros(tf.shape(prob_SI)), tf.zeros(tf.shape(prob_SI)),                         prob_NR, 1 -                   prob_NR  ), axis = -1)
    K_eta_h__n_r4 = tf.stack((tf.zeros(tf.shape(prob_SI)), tf.zeros(tf.shape(prob_SI)),     tf.zeros(tf.shape(prob_SI)), 1 - tf.zeros(tf.shape(prob_SI))), axis = -1)
    K_eta_h__n    = tf.stack((K_eta_h__n_r1, K_eta_h__n_r2, K_eta_h__n_r3, K_eta_h__n_r4), axis = -2)

    return K_eta_h__n

def FM_emission(q_t, y_t):
    
    emission_from_observed = (y_t)*(q_t)
    emission_from_unobserved =  (1-tf.reduce_sum(y_t, axis = -1, keepdims = True))*(1-q_t)

    return emission_from_observed + emission_from_unobserved


def sim_FM_X_0(input_0, parameters_0):

    probs_0 = FM_initial(input_0, parameters_0)

    X_0 = tfp.distributions.OneHotCategorical( probs = probs_0, dtype=tf.float32)

    return probs_0, X_0.sample() 


def sim_FM_X_t(transition_kernel, x_tm1):

    probs_t = tf.einsum("...ni,...nik->...nk", x_tm1, transition_kernel)

    X_t = tfp.distributions.OneHotCategorical( probs = probs_t, dtype=tf.float32)

    return probs_t, X_t.sample()


def sim_FM_Y(q, x_t):
    
    probs_be = tf.einsum("...i,...i->...", x_t, q)
    Be = tfp.distributions.Bernoulli( probs = probs_be, dtype = tf.float32)

    return tf.einsum("...ni,...n->...ni", x_t, Be.sample())

@tf.function(jit_compile=True)
def run_FM(input_0, parameters_0, input_kernel, parameters_kernel, q, T):

    _, x_0 = sim_FM_X_0(input_0, parameters_0)

    def body(input, t):
        
        x_tm1, _ = input

        transition_kernel = FM_transition(input_kernel, parameters_kernel, x_tm1)
        _, x_t = sim_FM_X_t(transition_kernel, x_tm1)

        y_t = sim_FM_Y(q, x_t)

        return x_t, y_t

    x, y = tf.scan(body, tf.range(0, T), initializer = (x_0, tf.zeros(tf.shape(x_0))))

    x_0 = tf.expand_dims(x_0, axis = 0)
    x   = tf.concat((x_0, x), axis = 0)

    y_0 = tf.zeros(tf.shape(x_0))
    y = tf.concat((y_0, y), axis = 0)

    return x, y

def FM_prediction(filtering_tm1, transition_matrix_tm1):
    
    return tf.einsum("sni,snij->snj", filtering_tm1, transition_matrix_tm1)

def FM_update(filtering_t_tm1, emission_t):

    update_t = (emission_t*filtering_t_tm1)

    likelihood_increment_t_tm1 =  tf.reduce_sum(update_t, axis = -1, keepdims = True)
    filtering_t = update_t/tf.where(update_t==0, tf.ones(tf.shape(likelihood_increment_t_tm1)), likelihood_increment_t_tm1)

    return likelihood_increment_t_tm1[...,0], filtering_t

def categ_sampling(probabilities):

    return tfp.distributions.OneHotCategorical( probs = probabilities, dtype=tf.float32).sample()

@tf.function(jit_compile=True)
def run_simba_FM(input_0, parameters_0, input_kernel, parameters_kernel, q, y, parallel_simulations=100):
    
    T = tf.shape(y)[0]
 
    def cond(input, t):

        return t<T

    def body(input, t):

        filtering_tm1, x_tm1_sim, log_likelihood = input

        transition_kernel = FM_transition(input_kernel, parameters_kernel, x_tm1_sim)
        _, x_t_sim = sim_FM_X_t(transition_kernel, x_tm1_sim)

        # prediction
        filtering_t_tm1 = FM_prediction(filtering_tm1, transition_kernel)

        # update as usual
        y_t = y[t,:,:]

        emission_t = FM_emission(q, y_t)
        likelihood_increment_t_tm1, filtering_t = FM_update(filtering_t_tm1, emission_t)

        return (filtering_t, x_t_sim, log_likelihood + tf.math.log(likelihood_increment_t_tm1)), t+1

    probs_0 = FM_initial(input_0, parameters_0)
    filtering_0 = probs_0*tf.ones(tf.concat(([parallel_simulations], tf.shape(probs_0)[1:]), axis =0))

    x_0_sim = categ_sampling(filtering_0)

    initializer = (filtering_0, x_0_sim, tf.zeros(tf.shape(x_0_sim)[:-1]))

    output = tf.while_loop(cond, body, loop_vars = (initializer, 1))

    return output[0][2]

@tf.function(jit_compile=True)
def run_joint_likelihood_FM(input_0, parameters_0, input_kernel, parameters_kernel, q, y, x):
    
    T = tf.shape(y)[0]
 
    def cond(input, t):

        return t<T

    def body(input, t):

        log_likelihood = input

        kernel_matrix = FM_transition(input_kernel, parameters_kernel, x[t-1,...])
        selected = tf.reduce_sum(kernel_matrix*tf.expand_dims(x[t-1,...], axis = -1)*tf.expand_dims(x[t,...], axis = -2), axis = [-2,-1])
        log_likelihood = log_likelihood + tf.reduce_sum(tf.math.log(selected))

        # update as usual
        y_t = y[t,:,:]

        emission_t = FM_emission(q, y_t)
        log_likelihood = log_likelihood + tf.reduce_sum(tf.math.log(tf.reduce_sum(emission_t*x[t,...], axis =-1)))

        return log_likelihood, t+1

    probs_0 = FM_initial(input_0, parameters_0)
    log_likelihood = tf.reduce_sum(tf.math.log(tf.reduce_sum(probs_0*x[0,...], axis =-1)))

    initializer = log_likelihood

    output = tf.while_loop(cond, body, loop_vars = (initializer, 1))

    return output[0]