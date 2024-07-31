import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import time

import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# initial distribution for FM
def FM_initial_dense_recovery(input_0, parameters_0):

    y = input_0

    mean_infection_period = parameters_0

    time_index = tf.reshape(tf.linspace(0, tf.shape(y)[0]-1, tf.shape(y)[0]), (tf.shape(y)[0], 1))

    notification_time = tf.reduce_sum(y[:,:,2]*tf.cast(time_index, dtype = tf.float32), axis = 0)
    notification_time = tf.where(notification_time!=0, notification_time, 2*tf.cast(tf.shape(y)[0], dtype = tf.float32))
    
    probs_0_indiv = tf.exp(-(notification_time-1)/mean_infection_period)*(1-tf.exp(-1/mean_infection_period))

    initial_distr    = tf.stack((1-probs_0_indiv, probs_0_indiv, tf.zeros(tf.shape(probs_0_indiv)), tf.zeros(tf.shape(probs_0_indiv))), axis = -1)

    return tf.expand_dims(initial_distr, axis  = 0)

# initial distribution for FM
def FM_initial_dense_initial_time(input_0, parameters_0):

    n_cattles, n_sheep = input_0

    N = tf.cast(tf.shape(n_cattles)[0], dtype = tf.float32)

    delta, zeta, chi, xi, spatial_kernel_matrix, initial_time = parameters_0

    infectiousness = (zeta*tf.math.pow(tf.transpose(n_cattles), chi) + tf.math.pow(tf.transpose(n_sheep), chi))/N

    susceptibility = (xi*tf.math.pow(tf.transpose(n_cattles), chi) + tf.math.pow(tf.transpose(n_sheep), chi))

    infectiousness_after_kernel = tf.einsum("ij,si->sj", spatial_kernel_matrix, infectiousness)/N

    rate_SI = delta*susceptibility*infectiousness_after_kernel

    probs_0_indiv = 1-tf.exp(-initial_time*rate_SI)

    initial_distr    = tf.stack((1-probs_0_indiv, probs_0_indiv, tf.zeros(tf.shape(probs_0_indiv)), tf.zeros(tf.shape(probs_0_indiv))), axis = -1)

    return initial_distr

# initial distribution for FM
def FM_initial_dense_comb(input_0, parameters_0):

    n_cattles, n_sheep, y = input_0

    time_index = tf.reshape(tf.linspace(0, tf.shape(y)[0]-1, tf.shape(y)[0]), (tf.shape(y)[0], 1))

    notification_time = tf.reduce_sum(y[:,:,2]*tf.cast(time_index, dtype = tf.float32), axis = 0)
    notification_time = tf.where(notification_time!=0, notification_time, 2*tf.cast(tf.shape(y)[0], dtype = tf.float32))

    N = tf.cast(tf.shape(n_cattles)[0], dtype = tf.float32)

    delta, zeta, chi, xi, spatial_kernel_matrix = parameters_0

    infectiousness = (zeta*tf.math.pow(tf.transpose(n_cattles), chi) + tf.math.pow(tf.transpose(n_sheep), chi))/N

    susceptibility = (xi*tf.math.pow(tf.transpose(n_cattles), chi) + tf.math.pow(tf.transpose(n_sheep), chi))

    infectiousness_after_kernel = tf.einsum("ij,si->sj", spatial_kernel_matrix, infectiousness)/N

    rate_SI = delta*susceptibility*infectiousness_after_kernel

    probs_0_indiv = 1-tf.exp(-notification_time*rate_SI)

    initial_distr    = tf.stack((1-probs_0_indiv, probs_0_indiv, tf.zeros(tf.shape(probs_0_indiv)), tf.zeros(tf.shape(probs_0_indiv))), axis = -1)

    return initial_distr
    

def sparse_infect_matrix_dense(psi, farms_coord):

    batch_distances = (tf.square((farms_coord[:,0:1] - tf.transpose(farms_coord)[0:1,:])/1000) + tf.square((farms_coord[:,1:2] - tf.transpose(farms_coord)[1:2,:])/1000))
    # weighted_distance =  (1/(2*psi*psi*1000*1000))*batch_distances
    distance_matrix = psi/(psi*psi + batch_distances) #tf.exp( -weighted_distance)/tf.math.sqrt(2*(np.pi)*psi*psi) #

    return distance_matrix


def FM_transition_dense(input_kernel, parameters_kernel, x_tm1, y_tm1):
    
    n_cattles, n_sheep = input_kernel

    N = tf.cast(tf.shape(n_cattles)[0], dtype = tf.float32)

    delta, zeta, chi, xi, spatial_kernel_matrix, mean_infected_period = parameters_kernel

    infectiousness = (zeta*tf.math.pow(tf.transpose(n_cattles), chi) + tf.math.pow(tf.transpose(n_sheep), chi))

    x_tm1_infectiusness = (infectiousness)*(x_tm1[...,2])

    susceptibility = (xi*tf.math.pow(tf.transpose(n_cattles), chi) + tf.math.pow(tf.transpose(n_sheep), chi))

    infectiousness_after_kernel = tf.einsum("ij,si->sj", spatial_kernel_matrix, x_tm1_infectiusness)/N

    rate_SI = delta*susceptibility*infectiousness_after_kernel
    prob_SI = 1-tf.exp(-rate_SI)

    prob_IN = (1-tf.exp(-(1/mean_infected_period)))*tf.ones(tf.shape(prob_SI))

    prob_NN = tf.zeros(tf.shape(prob_SI))

    K_eta_h__n_r1 = tf.stack((1 -               prob_SI  ,                   prob_SI,       tf.zeros(tf.shape(prob_SI)),     tf.zeros(tf.shape(prob_SI))), axis = -1)
    K_eta_h__n_r2 = tf.stack((tf.zeros(tf.shape(prob_IN)), 1 -               prob_IN,                           prob_IN,     tf.zeros(tf.shape(prob_IN))), axis = -1)
    K_eta_h__n_r3 = tf.stack((tf.zeros(tf.shape(prob_NN)), tf.zeros(tf.shape(prob_NN)),                         prob_NN, 1 -                   prob_NN  ), axis = -1)
    K_eta_h__n_r4 = tf.stack((tf.zeros(tf.shape(prob_NN)), tf.zeros(tf.shape(prob_NN)),     tf.zeros(tf.shape(prob_NN)), 1 - tf.zeros(tf.shape(prob_NN))), axis = -1)
    K_eta_h__n    = tf.stack((K_eta_h__n_r1, K_eta_h__n_r2, K_eta_h__n_r3, K_eta_h__n_r4), axis = -2)

    return K_eta_h__n


def FM_emission(q_t, y_t):
    
    emission_from_observed = (y_t)*(q_t)
    emission_from_unobserved =  (1-tf.reduce_sum(y_t, axis = -1, keepdims = True))*(1-q_t)

    return emission_from_observed + emission_from_unobserved


def sim_FM_X_0_dense(input_0, parameters_0, seed_FM_x0, FM_initial):

    probs_0 = FM_initial(input_0, parameters_0)

    X_0 = tfp.distributions.OneHotCategorical( probs = probs_0, dtype=tf.float32)

    return probs_0, X_0.sample(seed = seed_FM_x0) 

@tf.function
def sim_FM_X_t(kernel_matrix, x_tm1, seed_FM_xt):

    probs_t = tf.einsum("...ni,...nik->...nk", x_tm1, kernel_matrix)

    X_t = tfp.distributions.OneHotCategorical( probs = probs_t, dtype=tf.float32)

    return probs_t, X_t.sample(seed = seed_FM_xt)

@tf.function
def sim_FM_Y(q, x_t, seed_FM_yt):
    
    probs_be = tf.einsum("...i,...i->...", x_t, q)
    Be = tfp.distributions.Bernoulli( probs = probs_be, dtype = tf.float32)

    return tf.einsum("...ni,...n->...ni", x_t, Be.sample(seed = seed_FM_yt))

def run_FM_dense(input_0, parameters_0, input_kernel, parameters_kernel, q, T, seed_gen, FM_initial):

    seed_gen_xy = tfp.random.split_seed( seed_gen, n=2, salt='sim_both_FM')

    seed_gen_x  = tfp.random.split_seed( seed_gen_xy[0][0], n=T+1, salt='sim_x_FM')
    seed_gen_y  = tfp.random.split_seed( seed_gen_xy[1][0], n=T+1, salt='sim_y_FM')

    _, x_0 = sim_FM_X_0_dense(input_0, parameters_0, seed_gen_x[0], FM_initial)

    def body(input, t):
        
        x_tm1, y_tm1 = input

        kernel_matrix = FM_transition_dense(input_kernel, parameters_kernel, x_tm1, y_tm1)
        _, x_t = sim_FM_X_t(kernel_matrix, x_tm1, seed_gen_x[t+1])

        y_t = sim_FM_Y(q, x_t, seed_gen_y[t+1])

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

def categ_sampling(probabilities, seed_gen):

    return tfp.distributions.OneHotCategorical( probs = probabilities, dtype=tf.float32).sample(seed = seed_gen)

@tf.function
def run_simba_FM_dense(input_0, parameters_0, input_kernel, parameters_kernel, q, y, parallel_simulations, seed_simba, FM_initial):
    
    T = tf.shape(y)[0]

    seed_gen = tfp.random.split_seed( seed_simba, n=2, salt='run_simba_FM_dense')
    seed_gen_sim_x = tfp.random.split_seed( seed_gen[0][0], n=T+1, salt='run_simba_FM_dense_x')
    seed_gen_sim_y = tfp.random.split_seed( seed_gen[1][0], n=T+1, salt='run_simba_FM_dense_y')
 
    def cond(input, t):

        return t<T

    def body(input, t):

        filtering_tm1, x_tm1_sim, y_tm1_sim, log_likelihood = input

        kernel_matrix = FM_transition_dense(input_kernel, parameters_kernel, x_tm1_sim, y_tm1_sim)
        _, x_t_sim = sim_FM_X_t(kernel_matrix, x_tm1_sim, seed_gen_sim_x[t])

        y_t_sim = sim_FM_Y(q, x_t_sim, seed_gen_sim_y[t])

        # prediction
        filtering_t_tm1 = FM_prediction(filtering_tm1, kernel_matrix)

        # update as usual
        y_t = y[t,:,:]

        emission_t = FM_emission(q, y_t)
        likelihood_increment_t_tm1, filtering_t = FM_update(filtering_t_tm1, emission_t)

        return (filtering_t, x_t_sim, y_t_sim, log_likelihood + tf.math.log(likelihood_increment_t_tm1)), t+1

    probs_0 = FM_initial(input_0, parameters_0)
    filtering_0 = probs_0*tf.ones(tf.concat(([parallel_simulations], tf.shape(probs_0)[1:]), axis =0))

    x_0_sim = categ_sampling(filtering_0, seed_gen_sim_x[0])

    initializer = (filtering_0, x_0_sim, tf.zeros(tf.shape(x_0_sim)), tf.zeros(tf.shape(x_0_sim)[:-1]))

    output = tf.while_loop(cond, body, loop_vars = (initializer, 1))

    log_likelihood = output[0][3]

    max_log_likelihood = tf.reduce_max(log_likelihood, axis = 0, keepdims = True)

    return (tf.math.log(tf.reduce_mean(tf.exp(log_likelihood - max_log_likelihood), axis =0)) + tf.reduce_max(log_likelihood, axis = 0))


@tf.function
def simba_loss_FM(input_0, input_kernel, farms_coord, parameters, y, T, parallel_simulations, seed_simba_loss, FM_initial):

    # initial_time = tf.convert_to_tensor( [tf.math.exp(parameters[0])],  dtype = tf.float32 )  

    psi = tf.convert_to_tensor( [tf.math.exp(parameters[0])],  dtype = tf.float32 )  
    spatial_kernel_matrix = sparse_infect_matrix_dense(psi, farms_coord)
    delta  = tf.convert_to_tensor( [tf.math.exp(parameters[1])],  dtype = tf.float32 )  
    zeta   = tf.convert_to_tensor( [tf.math.exp(parameters[2])],  dtype = tf.float32 )  
    chi    = tf.convert_to_tensor( [tf.math.exp(parameters[3])],  dtype = tf.float32 )  
    xi     = tf.convert_to_tensor( [tf.math.exp(parameters[4])],  dtype = tf.float32 )    
    mean_infection_period   = tf.convert_to_tensor( [tf.math.exp(parameters[5])],  dtype = tf.float32 )  

    parameters_kernel = delta, zeta, chi, xi, spatial_kernel_matrix, mean_infection_period 

    parameters_0 = mean_infection_period #delta, zeta, chi, xi, spatial_kernel_matrix

    q     = tf.convert_to_tensor([[0.0, 0.0, 1.0, 0.0]], dtype = tf.float32)

    log_likelihood = run_simba_FM_dense(input_0, parameters_0, input_kernel, parameters_kernel, q, y, parallel_simulations, seed_simba_loss, FM_initial)

    return -tf.reduce_sum(log_likelihood)/T

@tf.function
def simba_loss_grad_FM(input_0, input_kernel, farms_coord,  parameters, y, T, parallel_simulations, seed_simba_loss, FM_initial):

    with tf.GradientTape() as tape:
        # Call your likelihood function with the parameters_var and other necessary inputs
        loss = simba_loss_FM(input_0, input_kernel, farms_coord, parameters, y, T, parallel_simulations, seed_simba_loss, FM_initial)

    # Use TensorFlow's automatic differentiation to compute gradients
    return loss, tape.gradient(loss, [parameters])


@tf.function
def compute_gradients_FM(input_0, input_kernel, farms_coord, parameters, y, parallel_simulations, seed_simba_loss_grad):

    with tf.GradientTape() as tape:

        initial_time = tf.convert_to_tensor( [tf.math.exp(parameters[0])],  dtype = tf.float32 )    

        psi    = tf.math.exp(parameters[1])  
        spatial_kernel_matrix = sparse_infect_matrix_dense(psi, farms_coord)
        delta  = tf.math.exp(parameters[2])  
        zeta   = tf.math.exp(parameters[3])  
        chi    = tf.math.exp(parameters[4])  
        xi     = tf.math.exp(parameters[5])    
        mean_infection_period = tf.math.exp(parameters[6])  

        parameters_kernel = delta, zeta, chi, xi, spatial_kernel_matrix, mean_infection_period 
        parameters_0 = delta, zeta, chi, xi, spatial_kernel_matrix, initial_time

        q     = tf.convert_to_tensor([[0.0, 0.0, 1.0, 0.0]], dtype = tf.float32)
        # tape.watch(parameters)
        target = run_simba_FM_dense(input_0, parameters_0, input_kernel, parameters_kernel, q, y, parallel_simulations, seed_simba_loss_grad)

    return tape.jacobian(target, parameters)


@tf.function
def compute_gradients_FM_serial(input_0, input_kernel, farms_coord, parameters, y, parallel_simulations, seed_simba_loss_grad):

    size = 1000
    batches = 9

    with tf.GradientTape() as tape:

        initial_time = tf.convert_to_tensor( [tf.math.exp(parameters[0])],  dtype = tf.float32 )     

        psi    = tf.math.exp(parameters[1])  
        spatial_kernel_matrix = sparse_infect_matrix_dense(psi, farms_coord)
        delta  = tf.math.exp(parameters[2])  
        zeta   = tf.math.exp(parameters[3])  
        chi    = tf.math.exp(parameters[4])  
        xi     = tf.math.exp(parameters[5])    
        mean_infection_period = tf.math.exp(parameters[6])  

        parameters_kernel = delta, zeta, chi, xi, spatial_kernel_matrix, mean_infection_period 
        parameters_0 = delta, zeta, chi, xi, spatial_kernel_matrix, initial_time

        q     = tf.convert_to_tensor([[0.0, 0.0, 1.0, 0.0]], dtype = tf.float32)
        # tape.watch(parameters)
        target = run_simba_FM_dense(input_0, parameters_0, input_kernel, parameters_kernel, q, y, parallel_simulations, seed_simba_loss_grad)[0:size]

        gradient = tape.jacobian(target, parameters)

    for i in range(1, batches):
        with tf.GradientTape() as tape:

            initial_time = tf.convert_to_tensor( [tf.math.exp(parameters[0])],  dtype = tf.float32 )     

            psi    = tf.math.exp(parameters[1])  
            spatial_kernel_matrix = sparse_infect_matrix_dense(psi, farms_coord)
            delta  = tf.math.exp(parameters[2])  
            zeta   = tf.math.exp(parameters[3])  
            chi    = tf.math.exp(parameters[4])  
            xi     = tf.math.exp(parameters[5])    
            mean_infection_period = tf.math.exp(parameters[6])  

            parameters_kernel = delta, zeta, chi, xi, spatial_kernel_matrix, mean_infection_period 
            parameters_0 = delta, zeta, chi, xi, spatial_kernel_matrix, initial_time

            q     = tf.convert_to_tensor([[0.0, 0.0, 1.0, 0.0]], dtype = tf.float32)
            # tape.watch(parameters)
            
            if i<(batches-1):
                target = run_simba_FM_dense(input_0, parameters_0, input_kernel, parameters_kernel, q, y, parallel_simulations, seed_simba_loss_grad)[i*size:(i*size+size)]
            
            else:
                target = run_simba_FM_dense(input_0, parameters_0, input_kernel, parameters_kernel, q, y, parallel_simulations, seed_simba_loss_grad)[i*size:]

        gradient_0 = tape.jacobian(target, parameters)

        gradient = tf.concat((gradient, gradient_0), axis = 0)

    return gradient


@tf.function
def compute_gradient_FM(input_0, input_kernel, farms_coord, parameters, y, parallel_simulations, seed_simba_loss_grad):

    with tf.GradientTape() as tape:

        initial_time = tf.convert_to_tensor( [tf.math.exp(parameters[0])],  dtype = tf.float32 )    

        psi    = tf.math.exp(parameters[1])  
        spatial_kernel_matrix = sparse_infect_matrix_dense(psi, farms_coord)
        delta  = tf.math.exp(parameters[2])  
        zeta   = tf.math.exp(parameters[3])  
        chi    = tf.math.exp(parameters[4])  
        xi     = tf.math.exp(parameters[5])    
        mean_infection_period = tf.math.exp(parameters[6])  

        parameters_kernel = delta, zeta, chi, xi, spatial_kernel_matrix, mean_infection_period 
        parameters_0 = delta, zeta, chi, xi, spatial_kernel_matrix, initial_time

        q     = tf.convert_to_tensor([[0.0, 0.0, 1.0, 0.0]], dtype = tf.float32)

        # tape.watch(parameters)
        target = tf.reduce_sum(run_simba_FM_dense(input_0, parameters_0, input_kernel, parameters_kernel, q, y, parallel_simulations, seed_simba_loss_grad))

    return tape.jacobian(target, parameters)

@tf.function
def compute_hessian_FM(input_0, input_kernel, farms_coord, parameters, y, parallel_simulations, seed_simba_loss_grad):

    with tf.GradientTape() as tape2:

        with tf.GradientTape() as tape1:  

            initial_time = tf.convert_to_tensor( [tf.math.exp(parameters[0])],  dtype = tf.float32 )    

            psi    = tf.math.exp(parameters[1])  
            spatial_kernel_matrix = sparse_infect_matrix_dense(psi, farms_coord)
            delta  = tf.math.exp(parameters[2])  
            zeta   = tf.math.exp(parameters[3])  
            chi    = tf.math.exp(parameters[4])  
            xi     = tf.math.exp(parameters[5])    
            mean_infection_period = tf.math.exp(parameters[6])  

            parameters_kernel = delta, zeta, chi, xi, spatial_kernel_matrix, mean_infection_period 
            parameters_0 = delta, zeta, chi, xi, spatial_kernel_matrix, initial_time

            q     = tf.convert_to_tensor([[0.0, 0.0, 1.0, 0.0]], dtype = tf.float32)

            tape1.watch(parameters)

            log_likelihood_individuals = run_simba_FM_dense(input_0, parameters_0, input_kernel, parameters_kernel, q, y, parallel_simulations, seed_simba_loss_grad)

            target = tf.reduce_sum(log_likelihood_individuals)

        gradient = tape1.gradient(target, parameters)
    
    hessian = tape2.jacobian(gradient, parameters)

    return gradient, hessian


def expected_gradients_FM(input_0, input_kernel, farms_coord, parameters, parallel_simulations, T, y, iterations, seed_simba_loss_expected_grad):

    N = tf.shape(y)[1]
    
    initial_time = tf.convert_to_tensor( [tf.math.exp(parameters[0])],  dtype = tf.float32 )    

    psi = tf.convert_to_tensor( [tf.math.exp(parameters[1])],  dtype = tf.float32 )  
    spatial_kernel_matrix = sparse_infect_matrix_dense(psi, farms_coord)
    delta  = tf.convert_to_tensor( [tf.math.exp(parameters[2])],  dtype = tf.float32 )  
    zeta  = tf.convert_to_tensor( [tf.math.exp(parameters[3])],  dtype = tf.float32 )  
    chi  = tf.convert_to_tensor( [tf.math.exp(parameters[4])],  dtype = tf.float32 )  
    xi   = tf.convert_to_tensor( [tf.math.exp(parameters[5])],  dtype = tf.float32 )    
    mean_infection_period   = tf.convert_to_tensor( [tf.math.exp(parameters[6])],  dtype = tf.float32 )  

    parameters_kernel = delta, zeta, chi, xi, spatial_kernel_matrix, mean_infection_period 
    parameters_0 = delta, zeta, chi, xi, spatial_kernel_matrix, initial_time

    q     = tf.convert_to_tensor([[0.0, 0.0, 1.0, 0.0]], dtype = tf.float32)

    seed_gen_sim, seed_gen_grad = tfp.random.split_seed( seed_simba_loss_expected_grad, n=2, salt='expected_gradient')

    seed_gen_sim_split = tfp.random.split_seed( seed_gen_sim[0], n=iterations, salt='expected_gradient_sim')
    seed_gen_grad_split = tfp.random.split_seed( seed_gen_grad[0], n=iterations, salt='expected_gradient_grad')

    def body(input, i):
        
        _, y_sim = run_FM_dense(input_0, parameters_0, input_kernel, parameters_kernel, q, T, seed_gen_sim_split[i])
        y_sim = y_sim[:,0,...]

        individuals_gradients_sim = compute_gradients_num_FM(input_0, input_kernel, farms_coord, parameters, y_sim, parallel_simulations, seed_gen_grad_split[i][0])

        return individuals_gradients_sim

    initialize = tf.zeros((N, tf.shape(parameters)[0]))
    individuals_gradients = tf.scan(body, tf.range(0, iterations), initializer = initialize)

    return individuals_gradients


def expected_hessian_FM(input_0, input_kernel, farms_coord, parameters, parallel_simulations, T, y, iterations, seed_gen_exp_hess):

    N = tf.shape(y)[1]
    
    initial_time = tf.convert_to_tensor( [tf.math.exp(parameters[0])],  dtype = tf.float32 )    

    psi = tf.convert_to_tensor( [tf.math.exp(parameters[1])],  dtype = tf.float32 )  
    spatial_kernel_matrix = sparse_infect_matrix_dense(psi, farms_coord)
    delta  = tf.convert_to_tensor( [tf.math.exp(parameters[2])],  dtype = tf.float32 )  
    zeta  = tf.convert_to_tensor( [tf.math.exp(parameters[3])],  dtype = tf.float32 )  
    chi  = tf.convert_to_tensor( [tf.math.exp(parameters[4])],  dtype = tf.float32 )  
    xi   = tf.convert_to_tensor( [tf.math.exp(parameters[5])],  dtype = tf.float32 )    
    mean_infection_period   = tf.convert_to_tensor( [tf.math.exp(parameters[6])],  dtype = tf.float32 )  

    parameters_kernel = delta, zeta, chi, xi, spatial_kernel_matrix, mean_infection_period 
    parameters_0 = delta, zeta, chi, xi, spatial_kernel_matrix, initial_time

    q     = tf.convert_to_tensor([[0.0, 0.0, 1.0, 0.0]], dtype = tf.float32)

    seed_gen_sim, seed_gen_hessian = tfp.random.split_seed( seed_gen_exp_hess, n=2, salt='expected_hessian')

    seed_gen_sim_split = tfp.random.split_seed( seed_gen_sim[0], n=iterations, salt='expected_gradient_sim')
    seed_gen_grad_split = tfp.random.split_seed( seed_gen_hessian[0], n=iterations, salt='expected_gradient_grad')

    def body(input, i):
        
        _, y_sim = run_FM_dense(input_0, parameters_0, input_kernel, parameters_kernel, q, T, seed_gen_sim_split[i])
        y_sim = y_sim[:,0,...]

        gradient_sim, hessian_sim = compute_hessian_FM(input_0, input_kernel, farms_coord, parameters, y, parallel_simulations, seed_gen_grad_split[i][0])

        return gradient_sim, hessian_sim

    initialize = tf.zeros((tf.shape(parameters)[0])), tf.zeros((tf.shape(parameters)[0], tf.shape(parameters)[0]))

    gradient, hessian = tf.scan(body, tf.range(1, iterations), initializer = initialize)

    return gradient, hessian

@tf.function
def compute_gradients_num_FM(input_0, input_kernel, farms_coord, parameters, y, parallel_simulations, seed_simba_loss_grad, epsilon = 0.05):

    seed_p, seed_m = tfp.random.split_seed( seed_simba_loss_grad, n=2, salt='compute_gradients_num_FM')

    seed_p_iter = tfp.random.split_seed( seed_p[0], n=tf.shape(parameters)[0], salt='seed_p_iter')
    seed_m_iter = tfp.random.split_seed( seed_m[0], n=tf.shape(parameters)[0], salt='seed_m_iter')

    def body(input, i):

        one_hot = tf.one_hot(i, tf.shape(parameters)[0])

        parameters_p = parameters+epsilon*one_hot

        initial_time = tf.convert_to_tensor( [tf.math.exp(parameters_p[0])],  dtype = tf.float32 )     

        psi    = tf.math.exp(parameters_p[1])  
        spatial_kernel_matrix = sparse_infect_matrix_dense(psi, farms_coord)
        delta  = tf.math.exp(parameters_p[2])  
        zeta   = tf.math.exp(parameters_p[3])  
        chi    = tf.math.exp(parameters_p[4])  
        xi     = tf.math.exp(parameters_p[5])    
        mean_infection_period = tf.math.exp(parameters_p[6])  

        parameters_kernel = delta, zeta, chi, xi, spatial_kernel_matrix, mean_infection_period 
        parameters_0 = delta, zeta, chi, xi, spatial_kernel_matrix, initial_time

        q     = tf.convert_to_tensor([[0.0, 0.0, 1.0, 0.0]], dtype = tf.float32)

        loglike_p = run_simba_FM_dense(input_0, parameters_0, input_kernel, parameters_kernel, q, y, parallel_simulations, seed_p_iter[i][0])

        one_hot = tf.one_hot(i, tf.shape(parameters)[0])

        parameters_m = parameters+epsilon*one_hot

        initial_time = tf.convert_to_tensor( [tf.math.exp(parameters_m[0])],  dtype = tf.float32 )     

        psi    = tf.math.exp(parameters_m[1])  
        spatial_kernel_matrix = sparse_infect_matrix_dense(psi, farms_coord)
        delta  = tf.math.exp(parameters_m[2])  
        zeta   = tf.math.exp(parameters_m[3])  
        chi    = tf.math.exp(parameters_m[4])  
        xi     = tf.math.exp(parameters_m[5])    
        mean_infection_period = tf.math.exp(parameters_m[6])  

        parameters_kernel = delta, zeta, chi, xi, spatial_kernel_matrix, mean_infection_period 
        parameters_0 = delta, zeta, chi, xi, spatial_kernel_matrix, initial_time

        q     = tf.convert_to_tensor([[0.0, 0.0, 1.0, 0.0]], dtype = tf.float32)

        loglike_m = run_simba_FM_dense(input_0, parameters_0, input_kernel, parameters_kernel, q, y, parallel_simulations, seed_m_iter[i][0])

        grad = (loglike_p-loglike_m)/(2*epsilon)

        return grad

    initialize = tf.zeros((tf.shape(farms_coord)[0]))

    return tf.transpose(tf.scan(body, tf.range(0, tf.shape(parameters)[0]), initializer = initialize))
