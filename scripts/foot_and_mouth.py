import tensorflow as tf
import tensorflow_probability as tfp

import time

import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# initial distribution for FM
def FM_initial(input_0, parameters_0):
    
    n_cattles, n_sheep     = input_0
    delta_0, xi_0, chi_0 = parameters_0

    rate_0 = delta_0*(xi_0*tf.math.pow(tf.transpose(n_cattles), chi_0) + tf.math.pow(tf.transpose(n_sheep), chi_0))

    probs_0 = (1-tf.exp(-rate_0))
    
    initial_distr    = tf.stack((1-probs_0, probs_0, tf.zeros(tf.shape(probs_0)), tf.zeros(tf.shape(probs_0)), tf.zeros(tf.shape(probs_0))), axis = -1)

    return initial_distr
    

def sparse_infect_matrix(psi, n_farms, farms_coord, batch_size = 150, cut_off = 1e-06):

    num_batches = int(n_farms/batch_size)
    
    def cond(input, t):
    
        return t<num_batches+1

    def body(input, t):

        prev_infect_matrix = input

        batch_number = t
        
        batch_distances_1 = tf.square(farms_coord[batch_number*batch_size:(batch_number+1)*batch_size,0:1] - tf.transpose(farms_coord)[0:1,:]) 
        batch_distances   = batch_distances_1 + tf.square(farms_coord[batch_number*batch_size:(batch_number+1)*batch_size,1:2] - tf.transpose(farms_coord)[1:2,:])
        weighted_distance = batch_distances/(2*psi*psi*1000*1000) #psi/(psi*psi + batch_distances/(1000*1000)) #

        distance_matrix   = tf.exp( -weighted_distance) #weighted_distance #

        indeces = tf.where(distance_matrix>cut_off)

        values  = tf.gather_nd(distance_matrix, tf.where(distance_matrix>cut_off))

        infect_matrix = tf.sparse.SparseTensor(indeces, values, [tf.shape(distance_matrix)[0], n_farms])

        return tf.sparse.concat(axis=0, sp_inputs=[prev_infect_matrix, infect_matrix]), t+1

    batch_number = 0
    
    batch_distances_1 = tf.square(farms_coord[batch_number*batch_size:(batch_number+1)*batch_size,0:1] - tf.transpose(farms_coord)[0:1,:]) 
    batch_distances   = batch_distances_1 + tf.square(farms_coord[batch_number*batch_size:(batch_number+1)*batch_size,1:2] - tf.transpose(farms_coord)[1:2,:])
    weighted_distance = psi/(psi*psi + batch_distances/(1000*1000)) #batch_distances/(2*psi*psi*1000*1000)
    distance_matrix   = weighted_distance #tf.exp( -weighted_distance)

    indeces = tf.where(distance_matrix>cut_off)

    values  = tf.gather_nd(distance_matrix, tf.where(distance_matrix>cut_off))

    initial_matrix = tf.sparse.SparseTensor(indeces, values, [tf.shape(distance_matrix)[0], n_farms])

    infect_matrix, _ = tf.while_loop(cond, body, loop_vars = [(initial_matrix), 1])

    return tf.sparse.slice(infect_matrix, [0, 0], [n_farms, n_farms])

def sparse_infectiousness(input_kernel, parameters_kernel, x_tm1):

    n_cattles, n_sheep = input_kernel
    n_cattles = tf.transpose(n_cattles)
    n_sheep = tf.transpose(n_sheep)

    n_farms = tf.cast(tf.shape(n_cattles)[0], dtype=tf.float32)

    delta, zeta, chi, xi, kernel_matrix, mean_infection_period = parameters_kernel

    infectiousness = (zeta*tf.math.pow(n_cattles, chi) + tf.math.pow(n_sheep, chi))

    x_tm1_infectiusness = (infectiousness)*(x_tm1[...,1])
    
    return tf.transpose(tf.sparse.sparse_dense_matmul(kernel_matrix, x_tm1_infectiusness, adjoint_b=True,))# + 1/n_farms

@tf.function(jit_compile=True)
def FM_transition(input_kernel, parameters_kernel, infectiousness_after_kernel):
    
    input_kernel

    n_cattles, n_sheep = input_kernel
    n_cattles = tf.transpose(n_cattles)
    n_sheep = tf.transpose(n_sheep)

    delta, zeta, chi, xi, kernel_matrix, mean_infection_period = parameters_kernel

    susceptibility = (xi*tf.math.pow(n_cattles, chi) + tf.math.pow(n_sheep, chi))

    rate_SI = delta*(susceptibility)*infectiousness_after_kernel
    prob_SI = 1-tf.exp(-rate_SI)

    prob_IN = (1-tf.exp(-(1/mean_infection_period)))*tf.ones(tf.shape(prob_SI))
    
    prob_NQ = tf.zeros(tf.shape(prob_SI))

    prob_QR = tf.zeros(tf.shape(prob_SI))

    K_eta_h__n_r1 = tf.stack((1 -               prob_SI  ,                   prob_SI,       tf.zeros(tf.shape(prob_SI)),     tf.zeros(tf.shape(prob_SI)),     tf.zeros(tf.shape(prob_SI))), axis = -1)
    K_eta_h__n_r2 = tf.stack((tf.zeros(tf.shape(prob_SI)), 1 -               prob_IN,                         prob_IN,       tf.zeros(tf.shape(prob_SI)),     tf.zeros(tf.shape(prob_SI))), axis = -1)
    K_eta_h__n_r3 = tf.stack((tf.zeros(tf.shape(prob_SI)), tf.zeros(tf.shape(prob_SI)), 1 -                   prob_NQ,                         prob_NQ,       tf.zeros(tf.shape(prob_SI))), axis = -1)
    K_eta_h__n_r4 = tf.stack((tf.zeros(tf.shape(prob_SI)), tf.zeros(tf.shape(prob_SI)),     tf.zeros(tf.shape(prob_SI)), 1 -                   prob_QR,                         prob_QR  ), axis = -1)
    K_eta_h__n_r5 = tf.stack((tf.zeros(tf.shape(prob_SI)), tf.zeros(tf.shape(prob_SI)),     tf.zeros(tf.shape(prob_SI)),     tf.zeros(tf.shape(prob_SI)), 1 - tf.zeros(tf.shape(prob_SI))), axis = -1)
    K_eta_h__n    = tf.stack((K_eta_h__n_r1, K_eta_h__n_r2, K_eta_h__n_r3, K_eta_h__n_r4, K_eta_h__n_r5), axis = -2)

    return K_eta_h__n

@tf.function(jit_compile=True)
def FM_emission(q_t, y_t):
    
    emission_from_observed = (y_t)*(q_t)
    emission_from_unobserved =  (1-tf.reduce_sum(y_t, axis = -1, keepdims = True))*(1-q_t)

    return emission_from_observed + emission_from_unobserved

@tf.function(jit_compile=True)
def sim_FM_X_0(input_0, parameters_0):

    probs_0 = FM_initial(input_0, parameters_0)

    X_0 = tfp.distributions.OneHotCategorical( probs = probs_0, dtype=tf.float32)

    return probs_0, X_0.sample() 

@tf.function(jit_compile=True)
def sim_FM_X_t(kernel_matrix, x_tm1):

    probs_t = tf.einsum("...ni,...nik->...nk", x_tm1, kernel_matrix)

    X_t = tfp.distributions.OneHotCategorical( probs = probs_t, dtype=tf.float32)

    return probs_t, X_t.sample()

@tf.function(jit_compile=True)
def sim_FM_Y(q, x_t):
    
    probs_be = tf.einsum("...i,...i->...", x_t, q)
    Be = tfp.distributions.Bernoulli( probs = probs_be, dtype = tf.float32)

    return tf.einsum("...ni,...n->...ni", x_t, Be.sample())


def run_FM(input_0, parameters_0, input_kernel, parameters_kernel, q, time_density, T):

    _, x_0 = sim_FM_X_0(input_0, parameters_0)

    removal_times = tfp.distributions.Categorical( probs = time_density, dtype=tf.float32).sample(x_0.shape[:-1])

    def body(input, t):
        
        (x_tm1, _, y_tm1_q, _), removal_times = input

        infectiousness_after_kernel = sparse_infectiousness(input_kernel, parameters_kernel, x_tm1)

        kernel_matrix = FM_transition(input_kernel, parameters_kernel, infectiousness_after_kernel)
        _, x_t = sim_FM_X_t(kernel_matrix, x_tm1)

        y_t = sim_FM_Y(q, x_t)
        y_t_n = y_t[...,2]

        to_quarantine    = tf.expand_dims(tf.where(((y_t_n==1) & (removal_times>=0)), tf.zeros(tf.shape(x_t)[:-1]), tf.ones(tf.shape(x_t)[:-1])), axis =-1)
        quarantine_state = tf.expand_dims(tf.expand_dims(tf.one_hot(3, 5), axis = 0), axis = 0)*tf.ones(tf.shape(x_t))

        x_t = ((x_t*to_quarantine) + (quarantine_state*(1-to_quarantine)))

        y_t_q = y_tm1_q + y_t_n

        to_remove     = tf.expand_dims(tf.where(((y_t_q==1) & (removal_times==0)), tf.zeros(tf.shape(x_t)[:-1]), tf.ones(tf.shape(x_t)[:-1])), axis =-1)
        removal_state = tf.expand_dims(tf.expand_dims(tf.one_hot(4, 5), axis = 0), axis = 0)*tf.ones(tf.shape(x_t))

        x_t = ((x_t*to_remove) + (removal_state*(1-to_remove)))

        y_t_r = (removal_state*(1-to_remove))[:,:,-1]

        removal_times = removal_times - y_t_q

        return (x_t, y_t_n, y_t_q, y_t_r), removal_times

    (x, y_n, _, y_r), _ = tf.scan(body, tf.range(0, T), initializer = ((x_0, tf.zeros((tf.shape(x_0)[:-1])), tf.zeros((tf.shape(x_0)[:-1])), tf.zeros((tf.shape(x_0)[:-1]))), removal_times))

    x_0 = tf.expand_dims(x_0, axis = 0)
    x   = tf.concat((x_0, x), axis = 0)

    y_0 = tf.zeros((tf.shape(x_0)[:-1]))
    y_n = tf.concat((y_0, y_n), axis = 0)

    y_0 = tf.zeros((tf.shape(x_0)[:-1]))
    y_r = tf.concat((y_0, y_r), axis = 0)

    return x, y_n, y_r

@tf.function(jit_compile=True)
def FM_prediction(filtering_tm1, transition_matrix_tm1):
    
    return tf.einsum("sni,snij->snj", filtering_tm1, transition_matrix_tm1)

@tf.function(jit_compile=True)
def FM_update(filtering_t_tm1, emission_t):

    update_t = (emission_t*filtering_t_tm1)

    likelihood_increment_t_tm1 =  tf.reduce_sum(update_t, axis = -1, keepdims = True)
    filtering_t = update_t/tf.where(update_t==0, tf.ones(tf.shape(likelihood_increment_t_tm1)), likelihood_increment_t_tm1)

    return likelihood_increment_t_tm1[...,0], filtering_t

@tf.function(jit_compile=True)
def categ_sampling(probabilities):

    return tfp.distributions.OneHotCategorical( probs = probabilities, dtype=tf.float32).sample()

def run_simulation_likelihood_approx_FM(input_0, parameters_0, input_kernel, parameters_kernel, q, y_n, y_r, time_density, parallel_simulations=100):
    
    T = tf.shape(y_n)[0]
 
    def cond(input, t):

        return t<T

    def body(input, t):

        filtering_tm1, x_tm1_sim, y_tm1_q_sim, removal_times, log_likelihood = input

        infectiousness_after_kernel = sparse_infectiousness(input_kernel, parameters_kernel, x_tm1_sim)

        kernel_matrix = FM_transition(input_kernel, parameters_kernel, infectiousness_after_kernel)
        _, x_t_sim = sim_FM_X_t(kernel_matrix, x_tm1_sim)

        # prediction
        filtering_t_tm1 = FM_prediction(filtering_tm1, kernel_matrix)

        # update as usual
        y_t_n = y_n[t,:,:]

        q_t_n = q[...,2]
        q_t = tf.stack((tf.zeros(tf.shape(q_t_n)), tf.zeros(tf.shape(q_t_n)), q_t_n, tf.zeros(tf.shape(q_t_n)), tf.zeros(tf.shape(q_t_n)) ), axis = -1)
        y_t = tf.stack((tf.zeros(tf.shape(y_t_n)), tf.zeros(tf.shape(y_t_n)), y_t_n, tf.zeros(tf.shape(y_t_n)), tf.zeros(tf.shape(y_t_n)) ), axis = -1)

        emission_t = FM_emission(q_t, y_t)
        likelihood_increment_t_tm1, filtering_t = FM_update(filtering_t_tm1, emission_t)

        # deterministic movements in filter
        rounded_filtering = tf.math.round(filtering_t[:,:,2]) == 1
        to_quarantine = tf.where((rounded_filtering), tf.zeros(tf.shape(filtering_t)[:-1]), tf.ones(tf.shape(filtering_t)[:-1]))
        to_quarantine    = tf.expand_dims(to_quarantine, axis =-1)
        quarantine_state = tf.expand_dims(tf.expand_dims(tf.one_hot(3, 5), axis = 0), axis = 0)*tf.ones(tf.shape(filtering_t))

        filtering_t_afterq = ((filtering_t*to_quarantine) + (quarantine_state*(1-to_quarantine)))
        y_t_r = y_r[t,:,:]

        to_remove    = tf.expand_dims(tf.where(((tf.math.round(filtering_t_afterq[:,:,3])==1) & (y_t_r==1)), tf.zeros(tf.shape(filtering_t_afterq)[:-1]), tf.ones(tf.shape(filtering_t_afterq)[:-1])), axis =-1)
        remove_state = tf.expand_dims(tf.expand_dims(tf.one_hot(4, 5), axis = 0), axis = 0)*tf.ones(tf.shape(filtering_t_afterq))

        filtering_t_afterr = ((filtering_t_afterq*to_remove) + (remove_state*(1-to_remove)))

        # stochastic movements in the simulation: move to quarantine and removed
        y_t_sim = sim_FM_Y(q, x_t_sim)
        y_t_n_sim = y_t_sim[...,2]

        to_quarantine    = tf.expand_dims(tf.where(((y_t_n_sim==1) & (removal_times>=0)), tf.zeros(tf.shape(x_t_sim)[:-1]), tf.ones(tf.shape(x_t_sim)[:-1])), axis =-1)
        quarantine_state = tf.expand_dims(tf.expand_dims(tf.one_hot(3, 5), axis = 0), axis = 0)*tf.ones(tf.shape(x_t_sim))

        x_t_sim = ((x_t_sim*to_quarantine) + (quarantine_state*(1-to_quarantine)))

        y_t_q_sim = y_tm1_q_sim + y_t_n_sim

        to_remove     = tf.expand_dims(tf.where(((y_t_q_sim==1) & (removal_times==0)), tf.zeros(tf.shape(x_t_sim)[:-1]), tf.ones(tf.shape(x_t_sim)[:-1])), axis =-1)
        removal_state = tf.expand_dims(tf.expand_dims(tf.one_hot(4, 5), axis = 0), axis = 0)*tf.ones(tf.shape(x_t_sim))

        x_t_sim = ((x_t_sim*to_remove) + (removal_state*(1-to_remove)))

        removal_times = removal_times - y_t_q_sim

        return (filtering_t_afterr, x_t_sim, y_t_q_sim, removal_times, log_likelihood + tf.math.log(likelihood_increment_t_tm1)), t+1

    probs_0 = FM_initial(input_0, parameters_0)
    filtering_0 = probs_0*tf.ones(tf.concat(([parallel_simulations], tf.shape(probs_0)[1:]), axis =0))

    x_0_sim = categ_sampling(filtering_0)

    removal_times = tfp.distributions.Categorical( probs = time_density, dtype=tf.float32).sample(x_0_sim.shape[:-1])

    y_0_q_sim = tf.zeros((tf.shape(x_0_sim)[:-1]))

    initializer = (filtering_0, x_0_sim, y_0_q_sim, removal_times, tf.zeros(tf.shape(x_0_sim)[:-1]))

    output = tf.while_loop(cond, body, loop_vars = (initializer, 1))

    return output[0][4]



def proposal_distribution_0(input_kernel, parameters_kernel, q, filtering_0, y_0_n, y_0_r, y_1_n):

    # simulate a new instance of the model
    input_kernel_combine = input_kernel, (y_0_n, y_0_r)

    infectiousness_after_kernel = sparse_infectiousness(input_kernel_combine, parameters_kernel, filtering_0)

    transition_matrix_tm1 = FM_transition(input_kernel_combine, parameters_kernel, infectiousness_after_kernel)

    # update as usual
    q_t_n = q[...,2]
    q_t = tf.stack((tf.zeros(tf.shape(q_t_n)), tf.zeros(tf.shape(q_t_n)), q_t_n, tf.zeros(tf.shape(q_t_n)), tf.zeros(tf.shape(q_t_n)) ), axis = -1)
    y_t = tf.stack((tf.zeros(tf.shape(y_1_n)), tf.zeros(tf.shape(y_1_n)), y_1_n, tf.zeros(tf.shape(y_1_n)), tf.zeros(tf.shape(y_1_n)) ), axis = -1)

    emission_t = FM_emission(q_t, y_t)

    emission_proposal = tf.einsum("ni,...nji->...nj", emission_t[0,...], transition_matrix_tm1)
    _, proposal_0 = FM_update(filtering_0, emission_proposal)

    return proposal_0

def proposal_distribution(input_kernel, parameters_kernel, q, probs_t, emission_t, y_t_n, y_t_r, y_tp1_n):

    _, updated_probs_t = FM_update(probs_t, emission_t)

    # simulate a new instance of the model
    input_kernel_combine = input_kernel, (y_t_n, y_t_r)

    infectiousness_after_kernel = sparse_infectiousness(input_kernel_combine, parameters_kernel, updated_probs_t)

    transition_matrix_t = FM_transition(input_kernel_combine, parameters_kernel, infectiousness_after_kernel)

    # update as usual
    q_tp1_n = q[...,2]
    q_tp1 = tf.stack((tf.zeros(tf.shape(q_tp1_n)), tf.zeros(tf.shape(q_tp1_n)), q_tp1_n, tf.zeros(tf.shape(q_tp1_n)), tf.zeros(tf.shape(q_tp1_n)) ), axis = -1)
    y_tp1 = tf.stack((tf.zeros(tf.shape(y_tp1_n)), tf.zeros(tf.shape(y_tp1_n)), y_tp1_n, tf.zeros(tf.shape(y_tp1_n)), tf.zeros(tf.shape(y_tp1_n)) ), axis = -1)

    emission_tp1 = FM_emission(q_tp1, y_tp1)

    proposal_emission_emission_tp1 = tf.einsum("ni,...nji->...nj", emission_tp1[0,...], transition_matrix_t)

    _, proposal_t = FM_update(probs_t, proposal_emission_emission_tp1*emission_t)

    return proposal_t



@tf.function(jit_compile=True)
def run_simulation_likelihood_approx_importance_FM(input_0, parameters_0, input_kernel, parameters_kernel, q, y_n, y_r, parallel_simulations=100):
    
    T = tf.shape(y_n)[0]

    y_n = tf.concat((y_n, tf.zeros(tf.shape(y_n[0:1,...]))), axis = 0)
 
    def cond(input, t):

        return t<T

    def body(input, t):

        filtering_tm1, x_tm1_sim, log_likelihood, log_importance_weights = input

        log_likelihood = log_likelihood + log_importance_weights

        y_tm1_n_sim, y_tm1_r_sim = y_n[t-1,:,:], y_r[t-1,:,:]

        # simulate a new instance of the model
        input_kernel_combine = input_kernel, (y_tm1_n_sim, y_tm1_r_sim)

        infectiousness_after_kernel = sparse_infectiousness(input_kernel_combine, parameters_kernel, x_tm1_sim)

        transition_matrix_tm1 = FM_transition(input_kernel_combine, parameters_kernel, infectiousness_after_kernel)

        probs_t,_ = sim_FM_X_t(transition_matrix_tm1, x_tm1_sim)

        # prediction on the filtering using the feedback
        filtering_t_tm1 = FM_prediction(filtering_tm1, transition_matrix_tm1)

        # update as usual
        y_t_n = y_n[t,:,:]

        q_t_n = q[...,2]
        q_t = tf.stack((tf.zeros(tf.shape(q_t_n)), tf.zeros(tf.shape(q_t_n)), q_t_n, tf.zeros(tf.shape(q_t_n)), tf.zeros(tf.shape(q_t_n)) ), axis = -1)
        y_t = tf.stack((tf.zeros(tf.shape(y_t_n)), tf.zeros(tf.shape(y_t_n)), y_t_n, tf.zeros(tf.shape(y_t_n)), tf.zeros(tf.shape(y_t_n)) ), axis = -1)

        emission_t = FM_emission(q_t, y_t)
        likelihood_increment_t_tm1, filtering_t = FM_update(filtering_t_tm1, emission_t)

        proposal_t = proposal_distribution(input_kernel, parameters_kernel, q, probs_t, emission_t , y_n[t,:,:], y_r[t,:,:], y_n[t+1,:,:])

        x_t_sim = categ_sampling(proposal_t)

        log_importance_weights = tf.math.log(tf.einsum("...ni,...ni->...n", x_t_sim, probs_t)) - tf.math.log(tf.einsum("...ni,...ni->...n", x_t_sim, proposal_t))

        return  (filtering_t, x_t_sim, log_likelihood + tf.math.log(likelihood_increment_t_tm1), log_importance_weights), t+1

    probs_0 = FM_initial(input_0, parameters_0)
    filtering_0 = probs_0*tf.ones(tf.concat(([parallel_simulations], tf.shape(probs_0)[1:]), axis =0))

    log_likelihood = tf.zeros(tf.shape(filtering_0)[:-1])

    proposal_0 = proposal_distribution_0(input_kernel, parameters_kernel, q, filtering_0, y_n[0,:,:], y_r[0,:,:], y_n[1,:,:])

    x_tm1_sim = categ_sampling(proposal_0)

    log_importance_weights = tf.math.log(tf.einsum("...ni,...ni->...n", x_tm1_sim, filtering_0)) - tf.math.log(tf.einsum("...ni,...ni->...n", x_tm1_sim, proposal_0))

    initializer = (filtering_0, x_tm1_sim, log_likelihood, log_importance_weights)

    output = tf.while_loop(cond, body, loop_vars = (initializer, 1))

    return output[0][2]



# Metropolis within Gibbs
class prior_parameter():
    
    def __init__(self, prior_list):

        self.prior_list = prior_list

    def sample(self):

        prior_sample = []
        for prior_dist in self.prior_list:

            prior_sample.append(prior_dist.sample())
        
        return prior_sample

    def logpdf(self, parameter_sample):

        log_likelihood = tf.zeros(tf.shape(parameter_sample[0]))
        for i in range(len(self.prior_list)):
            log_likelihood = log_likelihood + self.prior_list[i].prob(parameter_sample[i])

        return log_likelihood


def Gaussian_RW(scale_star):
    
    def proposal(loc_star, scale_scale = 1):

        return tfp.distributions.Normal(loc = loc_star, scale = scale_scale*scale_star)

    return proposal

def identity(a):

    return a

def logit(a):

    return tf.math.log(a/(1-a))
    
def invlogit(a):

    return tf.exp(a)/(1+tf.exp(a))

class proposal_parameter():
    
    def __init__(self, proposal_list):

        self.proposal_list = proposal_list

    def sample(self, parameter_sample):

        proposal_sample = []
        for i in range(len(self.proposal_list)):

            proposal_sample_to_parameter = self.proposal_list[i](parameter_sample[i]).sample()

            proposal_sample.append(proposal_sample_to_parameter)
        
        return proposal_sample

    def sample_i(self, parameter_sample, i):

        proposal_sample = parameter_sample[:]

        proposal_sample[i] = self.proposal_list[i](parameter_sample[i]).sample()
        
        return proposal_sample

    def logpdf_ratio(self, parameter_sample, proposal_sample):
    
        log_likelihood_ratio = tf.zeros(tf.shape(parameter_sample[0]))
        for i in range(len(self.proposal_list)):

            parameter_sample_to_proposal       = parameter_sample[i]
            proposal_sample_sample_to_proposal = proposal_sample[i]

            log_likelihood_ratio = log_likelihood_ratio - self.proposal_list[i](parameter_sample_to_proposal).prob(proposal_sample_sample_to_proposal)
            log_likelihood_ratio = log_likelihood_ratio + self.proposal_list[i](proposal_sample_sample_to_proposal).prob(parameter_sample_to_proposal)

        return log_likelihood_ratio

class transformer_parameter():

    def __init__(self, to_likelihood_space_list):

        self.to_likelihood_space_list = to_likelihood_space_list

    def to_likelihood_space(self, theta_from_proposal):

        theta = []
        for i in range(len(theta_from_proposal)):

            theta.append(self.to_likelihood_space_list[i](theta_from_proposal[i]))

        return theta


def run_MCMC(my_prior_parameter, my_proposal_parameter, my_transformer_parameter, my_log_likelihood_function, iterations, file_name, initial_condition = None):
    
    if type(initial_condition)!=None:
        logtheta_0 = initial_condition
        
    else:
        logtheta_0 = my_prior_parameter.sample()

    def body(input, s):

        logtheta_0 = input
        logprior_0 = my_prior_parameter.logpdf(logtheta_0)
        theta_0    = my_transformer_parameter.to_likelihood_space(logtheta_0)
        loglikelihood_0 = my_log_likelihood_function(theta_0)
        nr_acceepted = tf.convert_to_tensor([0.], tf.float32)

        for i in range(len(logtheta_0)):

            logtheta_1 = my_proposal_parameter.sample_i(logtheta_0, i)
            logprior_1 = my_prior_parameter.logpdf(logtheta_1)
            theta_1    = my_transformer_parameter.to_likelihood_space(logtheta_1)
            loglikelihood_1 = my_log_likelihood_function(theta_1)

            logproposal_ratio = my_proposal_parameter.logpdf_ratio(logtheta_0, logtheta_1)

            logprior_ratio      = logprior_1 - logprior_0
            loglikelihood_ratio = loglikelihood_1 - loglikelihood_0

            acceptance = tf.math.exp((logprior_ratio + loglikelihood_ratio + logproposal_ratio))
            U          = tfp.distributions.Uniform(low = 0., high = 1.).sample()

            logtheta_0[i]   = tf.cast((U>acceptance), dtype = tf.float32)*logtheta_0[i]   + (1-tf.cast((U>acceptance), dtype = tf.float32))*logtheta_1[i]
            nr_acceepted = nr_acceepted + tf.cast((U<acceptance), dtype = tf.float32)

        return logtheta_0, nr_acceepted

    parameter_chain = []

    input_ = logtheta_0

    for iteration in range(iterations):

        logtheta_0, nr_acceepted = body(input_, iteration)

        parameter_chain.append(tf.stack((logtheta_0)))

        string1 = ["At iteration: "+str(iteration), "\n"]

        string2 = ["We have accepted: "+str(nr_acceepted.numpy()[0]), "\n"]

        f= open(file_name+".txt", "a")
        f.writelines(string1)
        f.writelines(string2)
        f.close()

        input_ = logtheta_0

    parameter_chain = tf.stack((parameter_chain), axis =1 )

    return parameter_chain
