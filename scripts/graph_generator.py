import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class households_simulation:
    
    def __init__(self, household_per_city, household_per_city_longitude, household_per_city_latitude):

        self.household_per_city = household_per_city
        self.household_per_city_longitude = tf.reshape(household_per_city_longitude, (household_per_city_longitude.shape[0]*household_per_city_longitude.shape[1], 1))
        self.household_per_city_latitude  = tf.reshape(household_per_city_latitude,  (household_per_city_longitude.shape[0]*household_per_city_longitude.shape[1], 1))

    def assign_city(self, initial_distribution, n_individuals):

        city = tfp.distributions.OneHotCategorical(probs = initial_distribution)

        return tf.cast(city.sample(n_individuals), dtype = tf.float32)

    def assign_household(self, city): 

        household_prob = tf.einsum("nl,lc->nlc", city, self.household_per_city)
        household_prob = tf.reshape(household_prob, (household_prob.shape[0], household_prob.shape[1]*household_prob.shape[2]))
        
        # household_longitude = tf.einsum("nl,lc->nlc", city, self.household_per_city_longitude)
        # household_longitude = tf.reshape(household_longitude, (household_longitude.shape[0], household_longitude.shape[1]*household_longitude.shape[2]))
        
        # household_latitude = tf.einsum("nl,lc->nlc", city, self.household_per_city_latitude)
        # household_latitude = tf.reshape(household_latitude, (household_latitude.shape[0], household_latitude.shape[1]*household_latitude.shape[2]))
    
        household            = tfp.distributions.Categorical(probs = household_prob, dtype='int64').sample()
        household            = tf.reshape(household,            (household.shape[0],            1)) 
        household_one_assing = tf.keras.backend.arange(0, stop=household_prob.shape[1], step=1, dtype='int64')
        household_one_assing = tf.reshape(household_one_assing, (household_one_assing.shape[0], 1))       
        household            = tf.concat((household, household_one_assing), axis = 0)

        individuals_indeces            = tf.keras.backend.arange(0, stop=household_prob.shape[0], step=1, dtype='int64')
        individuals_indeces            = tf.reshape(individuals_indeces, (individuals_indeces.shape[0], 1))
        individuals_indeces_one_assign = tf.keras.backend.arange(household_prob.shape[0], stop=household_prob.shape[0] + household_prob.shape[1], step=1, dtype='int64')
        individuals_indeces_one_assign = tf.reshape(individuals_indeces_one_assign, (individuals_indeces_one_assign.shape[0], 1))
        individuals_indeces            = tf.concat((individuals_indeces, individuals_indeces_one_assign), axis = 0)
  
        household_sample = tf.concat((individuals_indeces, household), axis = 1)

        loc_I_H = tf.sparse.SparseTensor(household_sample, tf.ones(household_sample.shape[0]), (household_sample.shape[0], self.household_per_city_longitude.shape[0]))
        loc_I_H_reset = tf.sparse.reset_shape(loc_I_H)
        # household_longitude = tf.reduce_sum(household_sample*household_longitude, axis = 1, keepdims = True)
        # household_latitude  = tf.reduce_sum(household_sample*household_latitude,  axis = 1, keepdims = True)

        household_longitude = self.household_per_city_longitude
        household_latitude  = self.household_per_city_latitude

        return tf.sparse.to_dense(loc_I_H_reset), household_sample, tf.concat((household_longitude, household_latitude), axis = 1)