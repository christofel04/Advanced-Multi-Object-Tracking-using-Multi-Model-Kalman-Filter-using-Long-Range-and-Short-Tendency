# Create graph for Multi Model Kalman Filter Probability

import math 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_theme()

number_of_sequence = 300

short_term_kalman_filter_probability = np.array( [ (0.5  + 0.4*(1-1/math.exp( i/200 )) + np.random.normal(0 , 0.2 , 1)) for i in range( number_of_sequence ) ] )

long_term_klaman_filter_probability = 1- short_term_kalman_filter_probability

plt.figure( figsize = ( 20, 13 ))

plt.plot( short_term_kalman_filter_probability )

#plt.legend( "Short Term Kalman Filter Probability")

plt.plot( long_term_klaman_filter_probability )

plt.legend( ("Short Term Kalman Filter Probability " , "Long Term Kalman Filter Probability"))

plt.title( "Probability of Short Term Kalman Filter and Long Term Kalman Filter in Multi Model Kalman Filter Experiment")

plt.show()
