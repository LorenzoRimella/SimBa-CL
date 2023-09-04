import tensorflow as tf
import tensorflow_probability as tfp

# initial distribution for SIS
def SIS_initial(input_0, parameters_0):
    
    W      = input_0[0]
    beta_0 = parameters_0[0]

    rate_0 = 1/(1+tf.exp(-tf.einsum("nj,...j->...n", W, beta_0)))
    
    initial_distr    = tf.stack((1-rate_0, rate_0), axis = -1)

    return initial_distr

# transition distribution for fully connected SIS
def SIS_transition(input_kernel, parameters_kernel, x_tm1):
    
    W                       = input_kernel[0]
    beta_lambda, beta_gamma, epsilon = parameters_kernel

    lambda__n = 1/(1+tf.exp(-tf.einsum("ij,...j->...i", W, beta_lambda)))
    gamma__n  = 1/(1+tf.exp(-tf.einsum("ij,...j->...i", W, beta_gamma)))

    c_tm1 = tf.reduce_sum( x_tm1, axis = -2, keepdims = True ) - x_tm1

    N = tf.cast(tf.shape(W)[0], dtype = tf.float32)

    rate_SI   = tf.expand_dims(tf.expand_dims(1-tf.exp(-lambda__n*((c_tm1[...,1]/N)+epsilon)), axis =-1), axis =-1)

    rate_IS   = tf.expand_dims(tf.expand_dims(1-tf.exp(-gamma__n), axis =-1), axis =-1)*tf.ones(tf.shape(rate_SI))

    K_eta_h__n_r1 = tf.concat((1 - rate_SI, rate_SI), axis = -1)
    K_eta_h__n_r2 = tf.concat((rate_IS, 1 - rate_IS), axis = -1)
    K_eta_h__n    = tf.concat((K_eta_h__n_r1, K_eta_h__n_r2), axis = -2)

    return K_eta_h__n


# initial distribution for SIS
def SEIR_initial(input_0, parameters_0):
    
    W      = input_0[0]
    beta_0 = parameters_0[0]

    rate_0 = 1/(1+tf.exp(-tf.einsum("nj,...j->...n", W, beta_0)))
    
    initial_distr    = tf.stack((1-rate_0, tf.zeros(tf.shape(rate_0)), rate_0, tf.zeros(tf.shape(rate_0))), axis = -1)

    return initial_distr

# transition distribution for fully connected SIS
def SEIR_transition(input_kernel, parameters_kernel, x_tm1):
    
    W                       = input_kernel[0]
    beta_lambda, beta_gamma, epsilon, rho = parameters_kernel

    lambda__n = 1/(1+tf.exp(-tf.einsum("ij,...j->...i", W, beta_lambda)))
    gamma__n  = 1/(1+tf.exp(-tf.einsum("ij,...j->...i", W, beta_gamma)))

    c_tm1 = tf.reduce_sum( x_tm1, axis = -2, keepdims = True ) - x_tm1

    N = tf.cast(tf.shape(W)[0], dtype = tf.float32)

    rate_SE   = tf.expand_dims(tf.expand_dims(1-tf.exp(-lambda__n*((c_tm1[...,1]/N)+epsilon)), axis =-1), axis =-1)
    rate_EI   = tf.expand_dims(tf.expand_dims(1 - tf.exp(-rho), axis = -1), axis = -1)*tf.ones(tf.shape(rate_SE))
    rate_IR   = tf.expand_dims(tf.expand_dims(1-tf.exp(-gamma__n), axis =-1), axis =-1)*tf.ones(tf.shape(rate_SE))

    K_eta_h__n_r1 = tf.concat((1 - rate_SE, rate_SE, tf.zeros(tf.shape(rate_SE)), tf.zeros(tf.shape(rate_SE))), axis = -1)
    K_eta_h__n_r2 = tf.concat((tf.zeros(tf.shape(rate_SE)), 1 - rate_EI, rate_EI, tf.zeros(tf.shape(rate_SE))), axis = -1)    
    K_eta_h__n_r3 = tf.concat((tf.zeros(tf.shape(rate_SE)), tf.zeros(tf.shape(rate_SE)), 1 - rate_IR, rate_IR), axis = -1)
    K_eta_h__n_r4 = tf.concat((tf.zeros(tf.shape(rate_SE)), tf.zeros(tf.shape(rate_SE)), tf.zeros(tf.shape(rate_SE)), tf.ones(tf.shape(rate_SE))), axis = -1)
    K_eta_h__n    = tf.concat((K_eta_h__n_r1, K_eta_h__n_r2, K_eta_h__n_r3, K_eta_h__n_r4), axis = -2)

    return K_eta_h__n

def SIS_spatial_transition(input_kernel, parameters_kernel, x_tm1):
    
    W, households_coord, loc_I_H = input_kernel
    beta_lambda, beta_gamma, psi, epsilon = parameters_kernel

    lambda__n = 1/(1+tf.exp(-tf.einsum("ij,...j->...i", W, beta_lambda)))
    gamma__n  = 1/(1+tf.exp(-tf.einsum("ij,...j->...i", W, beta_gamma )))
    
    psi_expanded = tf.expand_dims(psi, axis = -1)
    weighted_distance =  (1/(2*psi_expanded*psi_expanded))*(tf.square(households_coord[:,0:1] - tf.transpose(households_coord)[0:1,:]) + tf.square(households_coord[:,1:2] - tf.transpose(households_coord)[1:2,:]))
    distance_matrix = tf.exp( -weighted_distance)

    infected_per_household = tf.einsum("...n,nh->...h", x_tm1[...,1], loc_I_H)
    weighted_infective_pressure_household = tf.einsum("...ij,...j->...i", distance_matrix, infected_per_household)
    infective_pressure_individuals = tf.einsum("...h,nh->...n", weighted_infective_pressure_household, loc_I_H)

    N = W.shape[0]
    
    rate_SI   = tf.expand_dims(tf.expand_dims(1-tf.exp(-lambda__n*(epsilon + infective_pressure_individuals/N)), axis =-1), axis =-1)
    
    rate_IS   = tf.expand_dims(tf.expand_dims(1-tf.exp(-gamma__n), axis =-1), axis =-1)

    K_eta_h__n_r1 = tf.concat((1 - rate_SI, rate_SI), axis = -1)
    K_eta_h__n_r2 = tf.concat((rate_IS, 1 - rate_IS), axis = -1)
    K_eta_h__n    = tf.concat((K_eta_h__n_r1, K_eta_h__n_r2), axis = -2)

    return K_eta_h__n

# the class compartmental_model is defined from:
# - N: population size
# - input_0:      input for the initial distribution function
# - input_kernel: input for the transition distribution function
# - initial_distribution: the initial distribution function
# - transition_kernel:    the transition distribution function
class compartmental_model():

    def __init__(self, N, input_0, input_kernel, initial_distribution, transition_kernel):

        self.N     = N

        self.input_0      = input_0
        self.input_kernel = input_kernel

        self.initial_distribution  = initial_distribution
        self.transition_kernel     = transition_kernel

def sim_X_0(model, parameters_0, seed_gen_x0):

    prob_0 = model.initial_distribution(model.input_0, parameters_0)

    X_0 = tfp.distributions.OneHotCategorical( probs = prob_0, dtype=tf.float32)

    return prob_0, X_0.sample(seed=seed_gen_x0) 

@tf.function
def sim_X_t(model, parameters_kernel, x_tm1, seed_gen_xt):

    transition_matrix_tm1 = model.transition_kernel(model.input_kernel, parameters_kernel, x_tm1)
    prob_t = tf.einsum("...ni,...nik->...nk", x_tm1, transition_matrix_tm1)

    X_t = tfp.distributions.OneHotCategorical( probs = prob_t, dtype=tf.float32)

    return transition_matrix_tm1, X_t.sample(seed = seed_gen_xt)

def sim_X(model, parameters_0, parameters_kernel, T, seed_gen_allx):

    seed_gen_split = tfp.random.split_seed( seed_gen_allx, n=(T+1), salt='sim_X')

    def body(x_tm1, t):

        _, x_t = sim_X_t(model, parameters_kernel, x_tm1, seed_gen_split[t+1])

        return x_t

    _, x_0 = sim_X_0(model, parameters_0, seed_gen_split[0])
    x   = tf.scan(body, tf.range(0, T), initializer = x_0)

    x_0 = tf.expand_dims(x_0,axis = 0)
    x   = tf.concat((x_0, x), axis = 0)

    return x

# q should account for time
def sim_Y(q, x, seed_gen_y):
    
    prob_be = q*x
    be = tfp.distributions.Bernoulli( probs = prob_be, dtype = tf.float32)

    return x*be.sample(seed = seed_gen_y)

def sim(model, parameters_0, parameters_kernel, q, T, seed_gen):

    seed_gen_xy = tfp.random.split_seed( seed_gen, n=2, salt='sim_both')

    x = sim_X(model, parameters_0, parameters_kernel, T, seed_gen_xy[0][0])
    # the first q in time is always all zeros
    y = sim_Y(q, x, seed_gen_xy[1])

    return x, y
