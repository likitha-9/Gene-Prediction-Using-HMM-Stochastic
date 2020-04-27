# A class for performing hidden markov models

import copy
import numpy as np

class HMM():

    def __init__(self, transmission_prob, emission_prob, obs=None):
        '''
        Note that this implementation assumes that n, m, and T are small
        enough not to require underflow mitigation.
        Required Inputs:
        - transmission_prob: an (n+2) x (n+2) numpy array, initial, where n is
        the number of hidden states
        - emission_prob: an (m x n) 2-D numpy array, where m is the number of
        possible observations
        Optional Input:
        - obs: a list of observation labels, in the same order as their
        occurence within the emission probability matrix; otherwise, will assume
        that the emission probabilities are in alpha-numerical order.
        '''
        self.transmission_prob = transmission_prob
        self.emission_prob = emission_prob
        self.n = self.emission_prob.shape[1]
        self.m = self.emission_prob.shape[0]
        self.observations = None
        self.forward = []
        self.backward = []
        self.psi = []
        self.obs = obs
        self.emiss_ref = {}
        self.forward_final = [0 , 0]
        self.backward_final = [0 , 0]
        self.state_probs = []
        if obs is None and self.observations is not None:
            self.obs = self.assume_obs()

    def assume_obs(self):
        '''
        If observation labels are not given, will assume that the emission
        probabilities are in alpha-numerical order.
        '''
        obs = list(set(list(self.observations)))
        obs.sort()
        for i in range(len(obs)):
            self.emiss_ref[obs[i]] = i
        return obs

    def train(self, observations, iterations = 10, verbose=True):
        '''
        Trains the model parameters according to the observation sequence.
        Input:
        - observations: 1-D string array of T observations
        '''
        self.observations = observations
        self.obs = self.assume_obs()
        self.psi = [[[0.0] * (len(self.observations)-1) for i in range(self.n)] for i in range(self.n)]
        self.gamma = [[0.0] * (len(self.observations)) for i in range(self.n)]
        for i in range(iterations):
            old_transmission = self.transmission_prob.copy()
            old_emission = self.emission_prob.copy()
            if verbose:
                print("Iteration: {}".format(i + 1))
            self.expectation()
            self.maximization()

    def expectation(self):
        '''
        Executes expectation step.
        '''
        self.forward = self.forward_recurse(len(self.observations))
        self.backward = self.backward_recurse(0)
        self.get_gamma()
        self.get_psi()

    def get_gamma(self):
        '''
        Calculates the gamma matrix.
        '''
        self.gamma = [[0, 0] for i in range(len(self.observations))]
        for i in range(len(self.observations)):
            self.gamma[i][0] = (float(self.forward[0][i] * self.backward[0][i]) /
                                float(self.forward[0][i] * self.backward[0][i] +
                                self.forward[1][i] * self.backward[1][i]))
            self.gamma[i][1] = (float(self.forward[1][i] * self.backward[1][i]) /
                                float(self.forward[0][i] * self.backward[0][i] +
                                self.forward[1][i] * self.backward[1][i]))

    def get_psi(self):
        '''
        Runs the psi calculation.
        '''
        for t in range(1, len(self.observations)):
            for j in range(self.n):
                for i in range(self.n):
                    self.psi[i][j][t-1] = self.calculate_psi(t, i, j)

    def calculate_psi(self, t, i, j):
        '''
        Calculates the psi for a transition from i->j for t > 0.
        '''
        alpha_tminus1_i = self.forward[i][t-1]
        a_i_j = self.transmission_prob[j+1][i+1]
        beta_t_j = self.backward[j][t]
        observation = self.observations[t]
        b_j = self.emission_prob[self.emiss_ref[observation]][j]
        denom = float(self.forward[0][i] * self.backward[0][i] + self.forward[1][i] * self.backward[1][i])
        return (alpha_tminus1_i * a_i_j * beta_t_j * b_j) / denom

    def maximization(self):
        '''
        Executes maximization step.
        '''
        self.get_state_probs()
        for i in range(self.n):
            self.transmission_prob[i+1][0] = self.gamma[0][i]
            self.transmission_prob[-1][i+1] = self.gamma[-1][i] / self.state_probs[i]
            for j in range(self.n):
                self.transmission_prob[j+1][i+1] = self.estimate_transmission(i, j)
            for obs in range(self.m):
                self.emission_prob[obs][i] = self.estimate_emission(i, obs)

    def get_state_probs(self):
        '''
        Calculates total probability of a given state.
        '''
        self.state_probs = [0] * self.n
        for state in range(self.n):
            summ = 0
            for row in self.gamma:
                summ += row[state]
            self.state_probs[state] = summ

    def estimate_transmission(self, i, j):
        '''
        Estimates transmission probabilities from i to j.
        '''
        return sum(self.psi[i][j]) / self.state_probs[i]

    def estimate_emission(self, j, observation):
        '''
        Estimate emission probability for an observation from state j.
        '''
        observation = self.obs[observation]
        ts = [i for i in range(len(self.observations)) if self.observations[i] == observation]
        for i in range(len(ts)):
            ts[i] = self.gamma[ts[i]][j]
        return sum(ts) / self.state_probs[j]

    def backward_recurse(self, index):
        '''
        Runs the backward recursion.
        '''
        # Initialization at T
        if index == (len(self.observations) - 1):
            backward = [[0.0] * (len(self.observations)) for i in range(self.n)]
            for state in range(self.n):
                backward[state][index] = self.backward_initial(state)
            return backward
        # Recursion for T --> 0
        else:
            backward = self.backward_recurse(index+1)
            for state in range(self.n):
                if index >= 0:
                    backward[state][index] = self.backward_probability(index, backward, state)
                if index == 0:
                    self.backward_final[state] = self.backward_probability(index, backward, 0, final=True)
            return backward

    def backward_initial(self, state):
        '''
        Initialization of backward probabilities.
        '''
        return self.transmission_prob[self.n + 1][state + 1]

    def backward_probability(self, index, backward, state, final=False):
        '''
        Calculates the backward probability at index = t.
        '''
        p = [0] * self.n
        for j in range(self.n):
            observation = self.observations[index + 1]
            if not final:
                a = self.transmission_prob[j + 1][state + 1]
            else:
                a = self.transmission_prob[j + 1][0]
            b = self.emission_prob[self.emiss_ref[observation]][j]
            beta = backward[j][index + 1]
            p[j] = a * b * beta
        return sum(p)

    def forward_recurse(self, index):
        '''
        Executes forward recursion.
        '''
        # Initialization
        if index == 0:
            forward = [[0.0] * (len(self.observations)) for i in range(self.n)]
            for state in range(self.n):
                forward[state][index] = self.forward_initial(self.observations[index], state)
            return forward
        # Recursion
        else:
            forward = self.forward_recurse(index-1)
            for state in range(self.n):
                if index != len(self.observations):
                    forward[state][index] = self.forward_probability(index, forward, state)
                else:
                    # Termination
                    self.forward_final[state] = self.forward_probability(index, forward, state, final=True)
            return forward

    def forward_initial(self, observation, state):
        '''
        Calculates initial forward probabilities.
        '''
        self.transmission_prob[state + 1][0]
        self.emission_prob[self.emiss_ref[observation]][state]
        return self.transmission_prob[state + 1][0] * self.emission_prob[self.emiss_ref[observation]][state]

    def forward_probability(self, index, forward, state, final=False):
        '''
        Calculates the alpha for index = t.
        '''
        p = [0] * self.n
        for prev_state in range(self.n):
            if not final:
                # Recursion
                obs_index = self.emiss_ref[self.observations[index]]
                p[prev_state] = forward[prev_state][index-1] * self.transmission_prob[state + 1][prev_state + 1] * self.emission_prob[obs_index][state]
            else:
                # Termination
                p[prev_state] = forward[prev_state][index-1] * self.transmission_prob[self.n][prev_state + 1]
        return sum(p)

    def likelihood(self, new_observations):
        '''
        Returns the probability of a observation sequence based on current model
        parameters.
        '''
        new_hmm = HMM(self.transmission_prob, self.emission_prob)
        new_hmm.observations = new_observations
        new_hmm.obs = new_hmm.assume_obs()
        forward = new_hmm.forward_recurse(len(new_observations))
        return sum(new_hmm.forward_final)

if __name__ == '__main__':
    

    """
    Constructing an HMM:
    T = length of observation sequence
    N = number of states in the model
    M = number of observation symbols
    Q = distinct states of the Markov process
    V = set of possible observations
    A = state transition probabilities
    B = observation probability matrix
    pi = initial state distribution
    O = observation sequence

    T - (multiple)
    N - 4 (number of hidden states)
    M - 21 (number of unique amino acids, including Stop Codon (*)) - check observations.txt 
    Q - [A, C, G, T]
    V - import from the file ./data/observations.txt - [*, Ala, Arg, Asn, Asp, Cys, Gln, Glu, Gly, His, Ile, Leu, Lys, Met, Phe, Pro, Ser, Thr, Trp, Tyr, Val]
    A - import from the file hidden_state_transitions.py
    B - import from the file emission_probabilities.py
    pi - start --> [0.25, 0.25, 0.25, 0.25]
    O - (multiple)

    HIDDEN STATES - Q - [A, C, G, T]
    OBSERVATION STATES  - V - [*, Ala, Arg, Asn, Asp, Cys, Gln, Glu, Gly, His, Ile, Leu, Lys, Met, Phe, Pro, Ser, Thr, Trp, Tyr, Val]
    """

    #program imports
    import emission_probabilities as emissions, hidden_state_transitions as hidden

    #data imports
    import data_genomes as genomes, data_scaffolds as unplaced

    #scientific library imports
    import numpy as np, pandas as pd, matplotlib.pyplot as plt

    # np.array (emissions & transitions)
    """list_emissions = []
    for key in emissions.emissions:
        list_emissions.append([])
        for i in emissions.emissions[key]:
            list_emissions[-1].append(emissions.emissions[key][i])
        
    list_transitions = []
    for key in hidden.transitions:
        list_transitions.append([])
        for i in hidden.transitions[key]:
            list_transitions[-1].append(hidden.transitions[key][i])
    """
    acids = []
    list_emissions = []
    for i in emissions.emissions:
        for j in emissions.emissions[i]:
            if j not in acids:
                acids.append(j)
                
    for i in acids:
        list_emissions.append([])
        for x in emissions.emissions:
            list_emissions[-1].append(emissions.emissions[x][i])

    list_transitions = []
    for key in hidden.transitions:
        list_transitions.append([])
        for i in hidden.transitions[key]:
            list_transitions[-1].append(hidden.transitions[key][i])
    
    emissions = np.array(list_emissions)
    transitions = np.array(list_transitions)
    obs = ['Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Gln', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Asn', 'Pro', 'Lys', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', 'Gln', 'Pro', 'Gln', 'Pro', 'Gln', 'Pro', 'Gln', 'Pro', 'Gln', 'Pro', 'Gln', 'Pro', '*', 'Pro', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', 'Leu', 'Thr', 'Pro', 'Asn', 'Pro', 'Asn', 'Pro', 'Asn', 'Pro', 'Asn', 'Pro', 'Asn', 'Pro', 'Asn', 'Pro', 'Asn', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', 'Ser', 'Arg', 'Tyr', 'Pro', 'Gln', 'Pro', 'Ala', 'Arg', 'Pro', 'Pro', 'Gly', 'Ser', 'Asp', 'Leu', 'Arg', 'Arg', 'Thr', 'Val', 'Leu', 'Arg', 'Leu', 'Gln', 'Ser', 'Thr', 'Thr', 'Glu', 'Ile', 'Cys', 'Ala', 'Glu', 'Asp', 'Asn', 'Ala', 'Ala', 'Pro', 'Pro', 'Ser', 'Arg', 'Cys', 'Ser', 'Pro', 'Gly', 'Leu', 'Cys', '*', 'Gly', 'Glu', 'Arg', 'Asn', 'Ser', 'Ala', 'Val', 'Ala', 'Lys', 'Ala', 'Arg', 'Arg', 'Ala', 'Gly', 'Ala', 'Gly', 'Ala', 'Glu', 'Arg', 'Arg', 'Ala', 'Ala', 'Pro', 'Ala', 'Gln', 'Ala', 'Gln', 'Arg', 'Gly', 'Ala', 'Pro', 'Arg', 'Arg', 'Arg', 'Arg', 'Arg', 'Arg', 'Glu', 'Ala', 'Arg', 'Arg', 'Ala', 'Gly', 'Ala', 'Gly', 'Ala', 'Glu', 'Arg', 'Arg', 'Ala', 'Ala', 'Pro', 'Ala', 'Gln', 'Ala', 'Gln', 'Arg', 'Gly', 'Ala', 'Pro', 'Arg', 'Arg', 'Arg', 'Arg', 'Arg', 'Arg', 'His', 'Met', 'Leu', 'Ala', 'Arg', 'Arg', 'Gly', 'Gly', 'Gly', 'Val', 'Ala', 'Gln', 'Ala', 'Gln', 'Arg', 'Gly', 'Ala', 'Pro', 'Arg', 'Arg', 'Arg', 'Arg', 'Arg', 'Arg', 'Asp', 'Thr', 'Cys', 'Tyr', 'Arg', 'Val', 'Gln', 'Gly', 'Trp', 'Arg', 'Arg', 'Gly', 'Ala', 'Gly', 'Ala', 'Glu', 'Arg', 'Arg', 'Thr', 'Ala', 'Pro', 'Ala', 'Gln', 'Ala', 'Gln', 'Arg', 'His', 'Met', 'Leu', 'Ala', 'Arg', 'Pro', 'Gly', 'Val', 'Glu', 'Ala', 'Trp', 'Arg', 'Arg', 'Arg', 'Arg', 'Asp', 'Ala', 'Ser', 'Leu', 'Arg', 'Ala', 'Gly', 'Val', 'Gly', 'Gly', 'Ala', 'Cys', 'Val', 'Ala', 'Gly', 'Ala', 'Lys', 'Ser', 'His', 'Gly', 'Ala', 'Gly', 'Leu', 'Gly', 'Arg', 'Gly', 'Glu', 'Gly', 'Gly', 'Ala', 'Val', 'His', 'Ala', 'Gln', 'Lys', 'Leu', 'Thr', 'Ser', 'Arg', 'Trp', 'Arg', 'Gly', 'Ala', 'Glu', 'Thr', 'Gly', 'Arg', 'Thr', 'Ser', 'Val', 'Ile', 'Arg', 'Lys', 'Ala', 'Gly', 'Ile', 'Asp', 'Arg', 'Pro', 'Leu', 'Leu', 'Ala', 'Ala', 'Gly', 'His', 'Tyr', 'Arg', 'Thr', 'Arg', 'Leu', 'Leu', 'Thr', 'Val', 'Leu', 'Cys', 'Gln', 'Gly', 'Ala', 'Pro', 'Cys', 'Trp', 'Arg', 'Leu', 'Gly', 'Gln', 'Leu', 'Gln', 'Gly', 'Ser', 'Leu', 'Ala', '*', 'Ser', 'Gly', 'Gly', 'Gln', 'Arg', 'Pro', 'Leu', 'Leu', 'Ala', 'Pro', 'Gly', 'His', 'Cys', 'Arg', 'Ala', 'Leu', 'Leu', 'Leu', 'Thr', 'Val', '*', 'Trp', 'Trp', 'His', 'Ala', 'Ala', 'Cys', 'Trp', 'Gln', 'Leu', 'Gly', 'Thr', 'Leu', 'Gln', 'Gly', 'Pro', 'Leu', 'Ala', 'Gln', 'Gly', 'Val', 'Val', 'Ala', 'Ala', 'Arg', 'Pro', 'Pro', 'Ala', 'Gly', 'Ser', 'Trp', 'Gly', 'His', 'Cys', 'Arg', 'Ala', 'Leu', 'Leu', 'Leu', 'Gln', 'Gln', 'Tyr', 'Trp', 'Arg', 'Ile', 'Ile', 'Gly', 'Lys', 'His', 'Pro', 'Glu', 'His', 'Met', 'Leu', 'Phe', 'Gly', 'Leu', 'Ser', 'Arg', 'Leu', 'Leu', 'Asn', 'Met', 'Gly', 'Phe', 'Leu', 'Gly', 'Leu', 'Lys', 'Val', 'Lys', 'Asn', 'Lys', 'Tyr', 'Val', '*', 'Phe', 'Val', 'Asn', '*', 'Leu', 'Pro', 'Ser', 'Glu', 'Leu', 'Tyr', 'Cys', 'Ser', 'Val', 'Ser', 'His', 'Gln', 'Gln', 'Cys', 'Leu', 'Gly', 'Met', 'Pro', 'Val', 'Ser', 'Pro', 'Gln', 'Ser', 'Val', 'Tyr', 'Phe', 'Trp', 'Ile', 'Phe', 'Ala', 'Ser', 'Leu', 'Thr', 'Gly', 'Glu', 'Ala', 'Leu', 'Glu', 'Ile', 'Leu', 'Ile', 'Ser', 'Asp', 'Leu', 'Gly', 'Trp', 'Gly', 'Leu', 'Ala', 'Met', 'Cys', 'Ile', 'Phe', 'Leu', 'Asn', 'Phe', 'His', '*', '*', 'Phe', 'Cys', 'Cys', 'Met', 'Ala', 'Gly', 'Val', 'Glu', 'Asn', 'Asp', 'Cys', 'Ala', 'Asn', 'Leu', 'Pro', 'Asp', 'Phe', 'Leu', 'Cys', 'Cys', 'Ser', 'Cys', 'Met', '*', 'Phe', 'Lys', 'Arg', 'Asp', 'Cys', 'Gln', 'His', 'Arg', 'Val', 'Ser', 'Phe', 'Thr', 'Ile', 'Phe', 'Leu', 'Phe', 'Val', 'Asn', 'Leu', 'Pro', 'Ser', 'Ala', 'Phe', 'Ser', 'Leu', 'Thr', 'Ser', 'Ser', 'Phe', 'Cys', 'Ser', 'Cys', 'Val', 'Phe', 'Ala', 'Val', 'Ser', '*', 'Pro', 'Arg', 'Leu', 'Pro', 'Val', 'Ser', 'Phe', 'Pro', 'Pro', 'Gly', 'Leu', '*', 'Glu', 'Val', 'Thr', 'Gly', 'Ser', '*', 'Cys', 'Cys', 'Gly', 'Leu', 'His', 'Leu', 'Gln', 'Val', 'Ser', 'Asp', 'Phe', 'Gln', 'Gln', 'Leu', 'Leu', 'Ala', 'Cys', 'Ala', 'Arg', 'Val', 'Gln', 'Ala', 'Glu', 'His', 'Trp', 'Ser', 'Gly', 'Val', 'Phe', 'Leu', 'Trp', 'Arg', 'Gly', 'Ala', 'Met', 'Pro', 'Arg', 'Val', 'Gly', 'Trp', 'Ala', 'Ile', 'Val', 'His', 'Leu', 'Leu', 'Ala', 'Pro', 'Val', 'Val', 'Cys', 'Met', '*', 'Leu', 'Asn', 'Thr', 'Thr', 'Thr', 'Arg', 'His', 'Arg', 'Gly', 'Lys', 'Ile', 'Gly', 'Gly', 'Lys', 'Met', 'Ser', 'Glu', 'Ser', 'Ile', 'Asn', 'Phe', 'Ser', 'His', 'Asn', 'Leu', 'Gly', 'Gln', '*', 'Val', 'Val', 'Leu', 'Val', 'Leu', 'Ile', 'Ser', 'Leu', 'Ala', 'Val', 'Ile', 'Arg', 'Gly', 'Arg', 'Pro', 'Ser', 'Leu', 'Gln', 'Gln', 'Leu', 'Asp', 'Pro', 'Tyr', 'Leu', 'Pro', 'Ser', 'Ala', 'Ala', 'Ile', 'Gly', 'Ala', 'Gln', 'Ser', 'Arg', 'Ala', 'Val', 'Thr', 'Ala', 'Gln', 'Thr', 'Ser', 'Arg', 'Leu', 'Glu', 'Gly', 'Gly', 'Ala', 'Gln', 'Gln', 'Val', 'Trp', 'Leu', 'Trp', 'Pro', 'Trp', 'Glu', 'Ser', 'Arg', 'Trp', 'Lys', 'Ile', 'Arg', 'Gln', 'Ala', 'Ile', 'Ala', 'Ala', 'Thr', 'Glu', 'Pro', 'Ser', 'Gly', 'Leu', 'Ala', '*', 'Val', 'Gly', 'Ser', 'Leu', 'Ser', 'Ser', 'Thr', 'Ser', 'Pro', 'Leu', 'Trp', 'Val', 'Val', 'Gly', 'Ala', 'Glu', 'Thr', 'Gly', 'Gly', 'Ala', 'Glu', 'Pro', 'Gln', 'Ala', 'Gln', 'Pro', 'Arg', 'Gly', 'Leu', 'Lys', 'Lys', 'Trp', '*', 'Asn', 'Gly', 'Ala', 'Ala', 'Gly', 'Asp', 'Val', 'Trp', 'Ala', 'His', 'Arg', 'Pro', 'Gln', 'Ala', 'Pro', 'Val', 'Ser', 'Pro', 'Gln', 'Val', 'Cys', 'Gly', 'Asp', 'Ala', 'Arg', 'His', 'Ala', 'Leu', 'Pro', 'Gln', 'His', 'Gln', 'Val', 'Ser', 'Arg', 'Ala', 'Ala', 'Glu', 'Asp', 'Asp', 'Gly', 'Arg', 'Leu', 'Gly', 'Ser', 'His', 'Ser', 'Cys', 'Glu', 'Cys', 'Pro', 'Gln', 'Cys', 'Cys', 'Arg', 'Gly', 'Glu', 'Arg', 'Arg', 'Val', 'Asp', 'Ser', 'Glu', 'Trp', 'Glu', 'Trp', 'Arg', 'Arg', 'Pro', '*', 'Gly', 'Ser', 'Thr', 'Gly', 'Pro', 'Ala', 'Ser', 'Pro', 'Val', 'Ser', 'Trp', 'Arg', 'Gly', 'Phe', 'Asp', 'Ala', 'Pro', 'Pro', 'His', 'Pro', 'Leu', 'Asp', 'Leu', 'Pro', 'Cys', 'Asp', 'Val', 'Ile', 'Trp', 'Ser', 'Pro', 'Ala', 'Ala', 'Cys', 'Gly', 'Gly', 'Leu', '*', 'Ser', 'Leu', 'Leu', 'Val', 'Trp', 'Leu', 'Gln', 'Gly', 'Leu', 'Ala', 'Glu', 'Ser', 'Phe', 'Pro', 'Gly', 'Lys', 'Ala', 'Thr', 'Ser', 'Ser', 'Lys', 'Gln', 'Ser', 'Ala', 'Trp', 'Val', 'Ile', 'Pro', 'Phe', 'Thr', 'Pro', 'Ser', 'Ser', 'Glu', 'Pro', 'Arg', 'Pro', 'Gly', 'Ala', 'Pro', 'Lys', 'Lys', 'Gly', 'Ser', 'Gly', 'Gly', 'Glu', 'Pro', 'Val', 'His', 'Glu', 'Gly', 'Cys', 'Gln', 'Pro', 'Val', 'His', 'Arg', 'Gln', 'Ala', 'Trp', 'Leu', 'Pro', 'Pro', 'Ala', 'Gly', 'Ser', 'Thr', 'Asp', 'Arg', 'Gly', 'Trp', 'Arg', 'Arg', 'Gly', 'Glu', 'Glu', 'Glu', 'Ser', 'Glu', 'Val', 'Ala', 'Cys', 'Pro', 'Val', 'Ser', 'Tyr', 'Leu', 'Arg', 'Leu', 'Arg', 'Lys', 'Glu', 'Lys', 'Gly', 'Met', 'His', 'Cys', 'Trp', 'Gly', 'Gly', 'Ser', 'Cys', 'Asn', 'Ser', 'Lys', 'Pro', '*', 'Pro', 'Leu', 'Phe', 'Pro', 'Arg', 'Arg', 'Gln', 'Gly', 'His', 'Gln', 'Ala', 'Pro', 'Lys', 'Gly', 'Phe', 'Cys', 'Gln', 'His', 'Ser', 'Ala', 'Pro', 'Gly', 'Pro', 'Val', 'Ile', 'His', 'Pro', 'Ala', 'Pro', 'Cys', 'Pro', 'Gly', 'His', 'Ala', 'Val', 'Gly', 'Leu', 'Asp', 'Leu', 'Ser', 'Pro', 'Gly', 'Gly', 'Gly', 'Gln', 'Ser', 'His', 'Leu', 'Trp', 'Phe', 'Cys', 'His', 'Cys', 'Cys', 'Cys', 'Val', 'Glu', 'Val', 'His', 'Ser', 'Cys', 'Leu', 'Phe', 'Leu', 'Ser', 'Leu', 'Glu', 'Pro', 'Pro', 'Pro', 'Pro', 'Arg', 'Asp', 'His', 'Ile', 'Ser', 'His', 'Cys', 'Leu', 'Leu', 'Ser', 'Ala', 'Gln', 'Phe', 'His', 'Gln', 'Lys', '*', 'Ala', 'Ser', 'Ser', '*', 'Gln', 'Ala', 'Ala', 'Ala', 'Pro', 'Leu', 'Pro', 'Gly', 'Ala', 'Val', 'Pro', 'Phe', 'Leu', 'Cys', 'Ser', 'Ala', 'Arg', 'Trp', 'Arg', 'Arg', 'Cys', 'Leu', 'Ser', 'Trp', 'Ala', 'Trp', 'Ser', 'Ala', 'Gly', 'Ile', 'Leu', 'Leu', 'Gln', 'Arg', '*', 'Asn', 'Pro', 'Gly', 'Glu', 'Cys', 'Gly', 'Val', 'Gln', 'Ser', 'Val', 'Ala', 'Arg', 'Thr', 'Gln', 'Ala', 'Gln', 'Ala', 'Leu', 'Val', 'Pro', 'Val', 'Gly', 'Glu', 'Asn', 'Arg', 'Gly', 'Ile', 'Pro', 'Lys', 'Lys', 'Trp', 'Trp', 'Val', 'Leu', 'Ala', 'Ile', 'Arg', 'Glu', 'Ile', 'Phe', 'Pro', 'Gly', 'Gln', 'Leu', 'Pro', 'Ser', 'Val', 'Glu', 'Ser', 'Asn', 'Leu', 'Ser', 'Ser', 'Ile', 'Leu', 'Arg', 'Gly', 'Arg', 'Gly', 'Pro', 'Gly', 'Phe', 'Ser', 'Leu', 'Gly', 'Leu', 'Cys', 'Arg', 'Arg', 'Leu', 'Pro', 'Phe', 'Val', 'Leu', 'Pro', 'Thr', 'Phe', 'Leu', 'Glu', 'Ala', 'Arg', 'Arg', 'Ser', 'Arg', 'Pro', 'Ile', 'Cys', 'Tyr', 'Cys', 'Pro', 'Phe', 'Tyr', 'Asn', 'Asn', '*', 'Ser', '*', 'Leu', 'Pro', 'Trp', 'Thr', 'Ile', 'His', 'Pro', 'Leu', 'Val', 'Ser', 'Ile', '*', 'Glu', 'Asp', 'Pro', 'His', 'Gly', 'His', 'Arg', 'Ala', 'Pro', 'Ala', 'Trp', 'Gly', 'Leu', 'Val', 'Thr', 'Ser', 'Pro', 'Thr', 'Phe', 'Phe', 'Leu', 'Ser', 'His', 'Ser', 'Cys', 'Ser', 'Leu', 'Ala', 'Pro', '*', 'Pro', 'Ala', 'Pro', 'Gln', 'Pro', 'Cys', 'Leu', 'Asp', 'Phe', 'Tyr', 'Leu', 'Pro', 'Gly', 'Leu', 'Val', 'Pro', 'Val', 'Pro', 'Pro', 'Ser', 'Arg', 'Trp', 'His', 'Leu', 'Pro', 'Pro', 'Ser', 'Gln', 'Pro', 'Leu', 'Glu', 'Gln', 'Thr', 'Pro', 'Arg', 'His', 'Leu', 'Leu', 'Pro', 'Gln', 'His', 'Gln', 'Gln', 'Leu', 'Cys', 'Gln', 'Gly', 'Pro', 'Leu', 'Gly', 'Ser', 'Gln', 'His', 'Asp', 'Tyr', 'Phe', '*', 'Arg', 'Pro', 'Arg', 'Val', 'Cys', 'His', '*', 'Asn', 'Leu', 'Phe', 'Cys', 'Gly', 'Arg', 'Leu', 'Phe', 'Leu', 'Pro', 'Ser', 'Ala', 'Thr', 'Ala', 'Ala', 'Pro', 'Ala', 'Asp', 'Cys', 'Pro', 'Ser', 'Leu', 'Leu', 'Pro', 'Leu', 'Ile', 'Pro', 'Glu', 'Lys', 'Gln', 'Val', 'Ser', 'Trp', 'Glu', 'Leu', 'Leu', 'Pro', 'Pro', 'Leu', 'Pro', 'Arg', 'Asp', 'Gln', 'Gln', 'Gly', 'Gln', 'Glu', 'Ala', 'Val', 'Thr', 'Asp', 'Pro', 'Glu', 'Thr', 'Phe', 'Ala', 'Ser', 'Cys', 'Thr', 'Ala', 'Arg', 'Asp', 'Pro', 'Leu', 'Leu', 'Lys', 'Ala', 'His', 'Cys', 'Trp', 'Phe', 'Leu', 'Leu', 'Ser', 'Ser', 'Leu', 'Leu', 'Ile', 'Gly', 'Val', 'Pro', 'Phe', 'Ser', 'Leu', 'Glu', 'Ala', 'Ser', '*', 'Glu', 'His', 'Ser', 'Gly', 'Ala', 'Gly', 'Trp', 'Val', 'Glu', 'Pro', 'Ser', 'Pro', 'His', 'Gly', 'Ala', 'Gln', 'Ala', 'Asp', 'Arg', 'Ser', 'Pro', 'Arg', 'Pro', 'Ser', 'Cys', 'Val', 'Ala', 'Ser', 'Ser', 'Gln', 'Pro', 'Ser', 'Ala', 'Pro', '*', 'Ser', 'Trp', 'Ser', 'Pro', 'His', 'Ser', 'Ala', 'Gly', 'Ser', 'Val', 'Thr', 'Pro', 'Ser', 'Gln', 'Gly', 'Ser', 'Arg', 'Ser', 'Glu', 'Gln', 'Leu', 'Val', 'Leu', 'Ala', 'Val', 'Ser', 'Met', 'Ser', 'Glu', 'Gln', 'Arg', 'Pro', 'Lys', 'Ser', 'Gly', 'Ser', 'Gly', 'Gly', 'Glu', 'Gly', 'Val', 'Met', 'Glu', 'Pro', 'Pro', 'Thr', 'Ile', 'Pro', 'Ser', 'Arg', 'Pro', 'Arg', 'Pro', 'Pro', 'Leu', 'Pro', 'Val', 'Ala', 'Ala', 'Ala', 'Val', 'Ala', 'Ala', 'Glu', 'Glu', 'Gly', 'Trp', 'Ser', 'Leu', 'Thr', 'Arg', 'Gly', 'Gln', 'Arg', 'Leu', 'Leu', 'Arg', 'Ala', 'Pro', 'His', 'Gln', 'Pro', 'Gln', 'Val', 'Leu', 'Ser', 'Gln', 'Arg', 'Cys', 'Leu', 'Glu', 'Gly', 'Lys', 'Gly', '*', 'Val', 'Arg', 'Val', 'Val', 'Gly', 'Gly', 'Lys', 'Pro', 'Trp', 'Phe', 'Pro', 'Gln', 'Pro', 'Pro', 'Glu', 'Thr', '*', 'Ile', 'Gln', 'Glu', 'Glu', 'Lys', 'Gly', 'Arg', 'Thr', 'Glu', 'Leu', 'Gln', 'Gly', 'Ala', 'Gly', 'Pro', 'Gly', 'Arg', 'Ala', 'Ala', 'Ala', 'Leu', 'Pro', 'Pro', 'Thr', 'Leu', 'Ala', 'Pro', 'His', 'Asp', 'Gln', 'Leu', 'Val', 'Glu', 'Glu', 'Ile', 'Arg', 'His', 'Gln', 'Val', 'Pro', 'Thr', 'Leu', 'Ala', 'Arg', 'Gly', 'Ser', 'His', 'Cys', 'Asn', 'Gly', 'Lys', 'Ala', 'Thr', 'Asp', 'Trp', 'Gly', 'Glu', 'Glu', 'Phe', 'Ser', 'His', 'Met', 'Arg', 'Pro', 'Val', 'Thr', 'Pro', 'Cys', 'Pro', 'His', 'Pro', 'His', 'Asp', 'Thr', 'Pro', 'Gln', 'Pro', 'Ser', 'Lys', 'Ala', 'Thr', 'Val', 'Phe', 'Pro', 'Ser', '*', 'Leu', 'Arg', 'Ala', 'Ser', 'Val', 'Asp', 'Pro', '*', 'Pro', 'Ser', 'Thr', 'Gly', 'His', '*', '*', 'Asp', 'Ser', 'Gly', 'Cys', 'Leu', 'Arg', 'Ser', 'His', 'Leu', 'Pro', 'Ala', 'Thr', 'Ser', 'Gly', 'Pro', 'Gly', 'Pro', 'Gly', 'Cys', 'Ala', 'Ala', 'Pro', 'Leu', 'Tyr', 'Asn', 'Gly', 'Glu', 'Thr', 'Gly', 'Pro', 'Glu', 'Arg', '*', 'Gly', 'Ser', 'Leu', 'Pro', 'Gly', 'Val', 'Thr', 'Glu', 'Gln', 'Gly', 'Lys', 'Ser', 'Ser', 'Ala', 'Gly', 'Tyr', 'Lys', 'Leu', 'Lys', 'Thr', 'Ile', 'Val', 'Pro', 'Arg', 'Ala', 'Leu', 'Pro', 'Leu', 'Gln', 'Ala', 'Gln', 'Ala', 'Ser', 'His', 'His', 'Thr', 'Ser', 'Val', 'Cys', 'Val', 'His', 'Ser', 'Arg', 'His', 'His', 'Gln', '*', 'Pro', 'Pro', 'Glu', 'Ala', 'Ser', 'Gly', 'Pro', 'Val', 'Ser', 'Lys', 'Asn', 'Ile', 'Ser', 'Gly', 'Gly', 'Cys', 'Ser', 'Gly', '*', 'Pro', 'Leu', 'Pro', 'Trp', 'Thr', 'Ala', 'Leu', 'Gly', 'Ser', 'Arg', 'Arg', 'Arg', 'Phe', 'Ser', 'Cys', 'Gln', 'Phe', 'Glu', 'Leu', 'Gly', 'Glu', 'Leu', 'Arg', 'Glu', 'Glu', 'Ser', 'Ser', 'Thr', 'Met', 'Ala', 'Pro', 'Lys', 'Pro', 'Gly', 'Arg', 'Ser', 'His', 'Ser', 'Pro', 'Gly', 'Arg', 'Arg', 'Ala', 'Glu', 'Asp', 'Leu', 'Trp', 'Trp', 'Arg', 'Pro', 'Arg', 'Ala', 'Ser', 'Ser', 'Met', 'Cys', 'Pro', 'Arg', 'Gly', 'Ser', 'Arg', 'Gly', 'Gln', 'Leu', 'Ala', 'Arg', 'Ala', 'Gly', 'Gly', 'Gly', 'Gln', 'Lys', 'Ala', 'Pro', 'Gly', 'Gly', 'Leu', 'Arg', 'Ala', 'Gly', 'Gly', 'Glu', 'Glu', 'Ala', 'Ile', 'Leu', 'Pro', 'Lys', 'Ala', 'Leu', 'Arg', 'Leu', 'Gln', 'Ala', 'Pro', 'Gly', 'Pro', 'Ala', 'His', 'Leu', 'Ala', 'Pro', 'Ala', 'Pro', 'Ser', 'Ala', 'Ala', 'Ala', 'Ser', 'Pro', 'Ala', 'Phe', 'Ala', 'Pro', 'Ser', 'Cys', 'Cys', 'Ala', 'Ala', 'Trp', 'Pro', 'Cys', 'Arg', 'Cys', 'Pro', 'Gln', 'Leu', 'Gly', 'Gly', 'Trp', 'Thr', 'Leu', 'Ala', 'Glu', 'Trp', 'Pro', 'Ala', 'Thr', 'Gly', 'Gly', 'Val', 'Asn', 'His', 'Phe', 'Pro', 'Gly', 'Ser', 'Ser', 'Leu', 'Asp', 'Trp', 'Ser', 'Arg', 'Glu', 'Val', 'Gly', 'Asn', 'Arg', 'Ala', 'Arg', 'Arg', 'Lys', 'Gly', 'Cys', 'Ser', 'Gly', 'Arg', 'Ala', 'Gly', 'Glu', 'Ala', 'Tyr', 'Cys', 'Val', 'Gln', 'Glu', 'Pro', 'Ala', 'Gly', 'Arg', 'Glu', 'Val', 'Thr', 'Ser', 'Pro', 'Gln', 'Thr', 'Arg', 'Ser', 'Pro', 'Ala', 'Leu', 'Gly', 'Arg', 'Pro', 'Asp', 'Leu', 'Trp', 'Arg', 'Leu', 'Cys', 'Val', 'Gly', 'Ala', 'Trp', 'Ala', 'Leu', 'Thr', 'Ser', 'Ala', 'Thr', 'Thr', '*', 'Ala', 'Arg', 'Ala', 'Ser', 'Cys', 'Val', 'Gln', 'Ile', 'Leu', 'Pro', 'Ala', 'Ser', 'Ser', 'Leu', 'Ala', 'Pro', 'Thr', 'Leu', 'Gln', 'Ser', 'Trp', 'Thr', 'Pro', 'Glu', 'Leu', 'Ala', 'Met', 'Leu', '*', 'Gln', 'Ser', 'Gln', 'Leu', 'His', 'Thr', 'Arg', 'Ala', 'Ser', 'Arg', 'Gly', 'Val', 'Leu', 'Cys', 'His', 'Phe', 'Trp', 'Met', 'Leu', 'Gly', 'Leu', 'His', 'Trp', 'Glu', 'Thr', 'Gln', 'Gln', '*', 'Ser', '*', 'Asn', 'Glu', 'Lys', 'Cys', 'Val', 'Ala', 'Val', 'Val', 'Cys', 'Tyr', '*', 'Thr', 'Pro', 'Ser', 'Phe', 'His', 'Trp', 'Phe', 'Asn', '*', 'Glu', 'Trp', 'Gly', 'Thr', 'Gln', 'Ser', 'Leu', 'Thr', 'Cys', 'Ser', 'Gly', 'Ser', 'Leu', 'Cys', 'Pro', 'Arg', 'Ser', 'Glu', 'Lys', 'Ser', 'Arg', 'Ala', 'Leu', 'Gln', 'Phe', 'Glu', 'Asn', 'His', 'Tyr', 'Phe', 'Met', 'Asn', 'Gln', 'Val', 'Glu', 'Gln', 'Asp', 'Ile', '*', 'Asn', 'Gly', 'Asn', 'Tyr', 'Ser', 'Lys', 'Asn', '*', 'Glu', 'Phe', 'Leu', 'Thr', 'Thr', '*', 'Gln', 'Thr', 'His', 'Arg', 'Lys', 'Ser', 'Thr', 'Arg', 'Val', 'His', '*', 'Ala', 'Arg', 'Gln', 'Lys', 'Ser', 'Gly', 'Gly', 'Leu', 'Lys', 'Glu', 'Leu', 'Leu', 'Pro', 'Pro', 'Glu', 'Gly', 'Asp', 'Ala', 'Leu', 'Leu', 'Leu', 'Leu', 'Ser', 'Ser', 'Cys', 'Leu', 'Ala', 'Pro', 'Trp', 'Pro', 'Thr', 'Gly', 'Ala', 'Ala', 'Val', 'Glu', 'Gly', 'Gly', 'Ser', 'Gly', 'Gly', 'Ala', 'Leu', 'Ala', 'Ser', 'Thr', 'Ser', 'Gly', 'Ala', 'Gly', 'Gly', 'Gly', 'Gly', 'Gly', 'Gly', 'Gly', 'Gly', 'Gly', 'Gly', 'Val', 'Ser', 'Thr', 'Pro', 'Ser', 'Cys', 'Arg', 'Ser', 'Glu', 'Thr', 'Gln', 'Ser', 'Val', 'Gly', 'Cys', 'Leu', 'Gly', 'Lys', 'Lys', 'Val', 'Cys', 'Asp', 'Gln', 'Gly', 'Gly', 'Pro', 'Arg', 'Pro', 'Ser', 'Ser', 'His', 'Pro', 'Arg', 'Thr', 'Gln', 'Leu', 'Thr', 'Tyr', 'Leu', 'Glu', 'Arg', 'Leu', 'Gly', 'Tyr', 'Leu', 'Ser', 'Val', 'Glu', 'Gly', 'Gly', 'Gln', 'Phe', 'Trp', 'Asn', 'Gly', 'Ala', 'Arg', 'Gly', 'Arg', 'Gly', 'Gly', 'Asn', 'Ala', 'Gly', 'Ala', 'Gln', 'Val', 'Gly', 'Asn', 'Val', 'His', 'Glu', 'Val', 'Val', 'Gly', 'Asn', 'Ala', 'Gly', 'Gln', 'Val', 'Arg', 'Gln', 'Val', 'Gly', 'Trp', 'Asn', 'Ile', 'Asn', 'Leu', 'Arg', 'His', 'Leu', 'Ala', 'Gln', 'Val', 'Trp', 'His', 'Ile', 'Glu', 'Val', 'Val', 'Leu', 'Trp', 'Asp', 'Leu', 'Gln', 'Asp', '*', 'Ala', 'Gly', 'Thr', 'Cys', 'Glu', 'Arg', '*', 'Gln', 'Gly', 'Pro', 'Ala', 'Gly', 'Ala', 'Ala', 'Asn', 'Lys', 'Thr', 'Leu', 'Cys', 'Ala', 'Pro', 'Pro', 'Met', 'Gly', 'Gly', 'Ile', 'Arg', 'Gly', 'Pro', 'Thr', 'Ala', 'Leu', 'Thr', 'Gly', 'Glu', 'Glu', 'Leu', 'Trp', 'Gln', 'Gly', 'Pro', 'Gly', 'Pro', 'Leu', 'His', 'Leu', 'Ser', 'Pro', 'Pro', 'Leu', 'Ser', 'His', 'Pro', 'Ser', 'His', 'Leu', 'Leu', 'Phe', 'Gln', 'Leu', 'Leu', 'Ser', 'Leu', 'Ala', 'Asp', 'Gly', 'Gln', 'Gly', 'Gly', 'Ile', 'Lys', 'Gln', 'Leu', 'Leu', 'Leu', 'Cys', 'Leu', 'Cys', 'Pro', 'Gln', 'His', 'His', 'Met', 'Gly', 'Leu', 'Cys', 'Tyr', 'Ser', 'Thr', 'Ser', 'Gln', 'Gly', 'Val', 'Gln', 'Glu', 'Asp', 'Ile', 'Leu', 'Leu', 'Leu', 'Pro', 'Thr', 'Glu', 'Ala', 'Thr', 'Trp', 'Gly', 'Ser', 'Gly', 'Lys', 'Leu', 'Thr', 'Pro', 'Ala', 'Val', 'Leu', 'Ser', 'Pro', 'Cys', 'Ser', 'Ser', 'Pro', 'Thr', 'Ser', 'Ser', 'Gly', 'Ala', 'Gln', 'Arg', 'Ala', 'Leu', 'Trp', 'Gly', 'Pro', 'Arg', 'Pro', 'Pro', 'Glu', 'Pro', 'Ser', 'His', 'Pro', 'Ser', 'Pro', 'Pro', 'Gly', 'Ser', 'Trp', 'Pro', 'Met', 'Cys', 'Cys', 'Thr', 'Cys', 'Val', '*', 'Cys', 'Pro', 'Gly', 'Ser', 'Pro', 'Leu', 'Ser', 'Gln', 'Ala', 'Gly', 'Pro', 'Pro', 'Ala', 'His', 'Thr', 'Pro', 'Arg', 'Pro', 'Cys', 'Pro', 'Leu', 'Ala', 'Ile', 'Gln', 'Val', 'Leu', 'Gly', 'Gly', 'Val', 'Glu', 'Glu', 'Gln', 'Gln', 'Gly', 'Ala', 'Asp', 'Arg', 'Ala', 'Asp', 'Val', 'Ala', 'Gly', 'Lys', 'Thr', 'Pro', 'Lys', 'Ser', 'Leu', 'Phe', 'Cys', 'Ile', 'Val', 'Leu', 'Gly', 'Leu', 'Arg', 'Leu', 'Gly', 'Ala', 'His', 'Ala', 'His', 'Arg', 'Lys', 'Val', 'Leu', 'Gln', 'Leu', 'Leu', 'Leu', 'Arg', 'Gly', 'Pro', 'Gly', 'Trp', 'Pro', 'Arg', 'Asp', 'Gly', 'Glu', 'Tyr', 'Leu', 'Val', 'Leu', 'Gly', 'Leu', 'Ile', 'Ser', 'Cys', 'His', 'Pro', 'Ile', 'Pro', 'Val', 'Ser', 'Leu', 'Leu', 'Trp', 'Gly', 'Thr', 'Glu', 'Pro', 'Tyr', 'Gly', 'Gly', 'Pro', 'Gly', 'Ser', 'Ser', 'Pro', 'Val', 'Ser', 'Ser', 'Pro', 'Pro', 'Gly', 'Val', '*', 'Gln', 'Ala', 'Ile', 'Cys', 'Ala', 'Ala', 'Ser', 'Arg', 'Pro', 'Ala', 'Gly', 'Pro', 'Ala', 'Arg', 'Pro', 'Gly', 'Gly', 'Gly', 'Ala', 'Cys', 'Ser', 'Gly', 'Ser', 'Cys', 'Gly', 'Gly', 'Gly', 'Val', 'Ser', 'Ala', 'Gly', 'Gln', 'Gly', 'Pro', 'Gly', 'Arg', 'Pro', '*', 'Arg', 'Trp', 'Ser', 'His', 'Ile', 'Pro', 'Ala', 'Gly', 'Ala', 'Leu', 'Glu', 'Gln', 'Gly', 'Thr', 'Trp', 'His', 'Trp', 'Arg', 'Thr', 'Pro', 'Val', 'Asp', 'Thr', 'Gly', 'Thr', 'Ser', 'Leu', 'Arg', 'Gly', 'Pro', 'Gln', 'Glu', 'Ala', 'Gln', 'Arg', 'Ala', 'Arg', 'Ile', 'Ala', 'Trp', 'Gln', 'Glu', 'Arg', 'Val', 'Glu', 'Leu', 'Glu', 'Ala', 'Trp', 'Ala', 'Arg', 'Arg', 'Lys', 'Leu', 'Lys', 'Val', 'Gln', 'Val', 'Gly', 'Ser', 'Arg', 'Ala', 'Glu', 'Thr', 'Gly', 'Gln', 'Pro', 'Gln', 'Arg', 'His', 'Gly', 'Glu', 'Met', 'Glu', 'Gly', 'Leu', 'Pro', 'Ser', 'Ser', 'Leu', 'Arg', 'Thr', 'Gln', 'Gly', 'Tyr', 'Gly', 'Asp', 'Tyr', 'Leu', 'Asp', 'Gly', 'Leu', 'Leu', 'Ala', 'Ala', 'Leu', 'Asp', 'Leu', 'Leu', 'Asn', 'Leu', 'Gly', 'Leu', 'Gly', 'Gln', 'Gly', 'Asp', 'Leu', 'Leu', 'Ser', 'Asn', 'Gly', 'Leu', 'His', 'Leu', 'Ala', 'Pro', 'Ala', 'Leu', 'Leu', 'Tyr', 'Leu', 'Leu', 'Gly', 'Asp', 'Pro', 'Ala', 'Met', 'Glu', 'Lys', 'Ile', 'Thr', 'Glu', 'Ala', 'Gly', 'Leu', 'Leu', 'Pro', 'Thr', 'Leu', 'Cys', 'Thr', 'Pro', 'Pro', 'Ala', 'Ser', 'Asn', 'Ser', 'Arg', 'Ala', 'Ala', 'Arg', 'Pro', 'Gly', 'Pro', 'Gln', 'Ala', 'Arg', 'Ala', 'Leu', 'Lys', 'Ser', 'Gly', 'Ser', 'Pro', 'Thr', 'Cys', 'Gln', 'Gly', 'Arg', 'Ser', 'Trp', 'Cys', 'His', 'Pro', 'Gly', 'Gly', 'Leu', 'Tyr', 'Lys', 'Asp', 'Asn', 'Leu', 'Thr', 'Cys', 'Arg', 'Val', 'Glu', 'Glu', 'Leu', 'Thr', 'Val', 'Leu', 'Ser', 'Ser', 'Leu', 'His', 'Ser', 'Gln', '*', 'Gly', 'Gln', 'Ala', 'Leu', 'Cys', 'Cys', 'His', 'Leu', 'Tyr', 'Met', 'Leu', 'Ser', 'Glu', 'Gly', 'Gln', 'Pro', 'Pro', 'Gly', 'His', 'Thr', 'Glu', 'Asp', 'Gly', 'Ile', 'Tyr', 'Thr', 'Cys', 'Thr', 'His', 'Gly', 'Tyr', '*', 'Trp', 'Gly', 'Lys', 'His', 'Phe', 'Thr', 'Thr', 'Pro', 'His', 'Asp', 'His', 'Val', 'Gln', 'Gln', 'Thr', 'Met', 'Trp', 'Pro', 'Leu', 'Gln', 'Arg', 'Gly', 'Asn', 'Gly', 'Asp', 'Arg', 'Arg', 'Leu', 'Arg', 'Leu', 'Ala', 'Arg', 'Leu', 'Asp', 'Leu', 'Ser', 'Val', 'Val', 'Thr', '*', 'Ile', 'Gln', 'Thr', 'Gly', 'Asn', 'Cys', 'Pro', 'Cys', 'Thr', 'Tyr', '*', 'Thr', 'Ala', 'His', '*', 'Ala', 'Asn', 'Pro', 'Glu', 'Ser', 'Arg', 'Pro', 'Pro', 'Pro', 'Gln', 'Cys', 'Gly', 'Leu', 'Ala', 'Pro', 'His', 'Leu', 'Leu', 'Pro', 'Ser', 'Ser', 'Leu', 'Val', 'Arg', 'Gly', 'Gly', 'Pro', 'Ser', 'Asp', 'Ile', 'Ser', 'Cys', 'Leu', 'Leu', 'Phe', 'Pro', 'Arg', 'Cys', 'Ala', 'Lys', 'Cys', 'Ile', 'Leu', 'Val', 'Cys', 'Leu', 'His', 'Leu', 'Met', 'Glu', 'Arg', 'His', 'Phe', 'Pro', 'Arg', 'His', 'Pro', 'Cys', 'Gly', 'Trp', 'Leu', 'Leu', 'Met', 'Pro', 'Glu', 'Ala', 'Gln', 'Val', 'Ser', 'Asp', 'Ala', 'Leu', 'Arg', 'His', 'Ile', 'Thr', 'Pro', 'Leu', 'Met', 'Leu', 'Phe', 'His', 'Val', 'Leu', 'Trp', 'Pro', 'Gln', 'Gln', 'Gly', 'Arg', 'Ser', 'His', 'Cys', 'Lys', 'Val', 'Asn', 'Ser', 'Asp', 'Ala', 'Cys', 'Val', 'Thr', 'Gln', 'His', 'Pro', 'Pro', 'Pro', 'Ser', 'Arg', 'Pro', 'Cys', 'Ser', 'Ser', 'Pro', 'Thr', 'Ser', 'Lys', 'Ser', 'Pro', 'Ala', 'Leu', 'Ala', 'His', 'Arg', 'Ala', 'Thr', 'Leu', 'His', 'Val', 'Gln', 'Ser', 'Ser', 'Leu', 'Ser', 'Thr', 'His', 'Arg', 'Ala', 'Arg', 'Ala', 'Ser', 'Leu', 'Cys', 'Gly', 'Ala', 'Gln', 'Gly', '*', 'Glu', 'Gly', 'Arg', 'Gly', 'Ala', 'Thr', 'Gly', 'Val', 'His', 'Glu', 'Glu', 'Gly', 'Gln', 'Glu', 'Glu', 'Gly', 'Val', 'Gly', 'Trp', 'Trp', 'Arg', 'Gly', 'Leu', 'Arg', 'Arg', 'Gln', 'Arg', 'Arg', 'Asp', 'Trp', 'Gly', 'Ser', '*', 'Gly', 'Lys', 'Gly', 'Gly', 'Gly', 'Gly', 'Cys', 'Gly', 'Met', 'Val', 'Glu', 'Gly', 'Leu', 'Gln', 'Thr', 'Leu', 'Gly', '*', 'Gly', 'Lys', 'Leu', 'Gly', 'Cys', 'Leu', '*', 'Arg', 'Leu', 'Glu', '*', 'Met', 'Ala', '*', 'Asn', 'Pro', 'Thr', 'Gln', '*', 'Ala', 'Lys', 'Ala', 'Thr', 'Ser', 'Thr', 'Asn', 'Val', 'Arg', 'Arg', 'Pro', 'Trp', 'Pro', 'Pro', 'Glu', 'Ser', 'Gln', 'Phe', 'His', 'Asn', 'Pro', 'Glu', 'Val', 'Pro', 'Val', 'Pro', '*', 'Arg', 'Val', 'Cys', 'Pro', 'Asp', 'Tyr', 'Ser', 'Trp', 'Leu', 'Leu', 'Val', 'Cys', 'Arg', 'Gly', 'Leu', 'Arg', 'His', 'Gly', 'Arg', 'Ala', 'Gly', 'Ser', 'Thr', 'Ser', 'Arg', 'His', 'Ser', 'Ser', 'Gly', 'Leu', 'Ser', 'Val', 'Pro', '*', 'Gln', 'Thr', 'Gly', 'Met', 'Lys', 'Val', 'Ala', 'Thr', 'Ile', 'Gln', 'Lys', 'Glu', 'Lys', 'Arg', 'Arg', 'Ala', 'Pro', 'Ser', 'Pro', 'Ser', 'Ser', 'Glu', 'Glu', 'Ala', 'Gly', 'Pro', 'Pro', 'Pro', 'Ser', 'Val', 'Cys', 'Ser', 'Ile', 'Phe', 'Ser', 'Gly', 'Trp', 'Gly', 'Glu', 'Ala', 'Phe', 'Ile', 'Cys', 'Cys', 'Lys', 'Gly', 'Ser', 'Ser', 'Ser', 'Thr', 'Ser', 'Cys', 'Leu', 'Asn', '*', 'Pro', '*', 'Phe', 'Pro', 'Gly', 'Gln', 'Pro', 'Arg', 'Ser', 'Ala', 'Leu', 'Gly', 'Ala', 'Asp', 'Thr', 'Thr', 'Phe', 'Gly', 'Arg', 'Cys', 'Ile', 'Ser', 'Ser', 'Ala', 'Phe', 'Glu', 'Val', 'His', 'Arg', 'Gly', 'Ser', 'Gly', 'Arg', 'Glu', 'Leu', 'Arg', 'Leu', 'Gly', 'Arg', 'Asp', 'Lys', 'Gly', 'Cys', 'Ser', 'Val', 'Leu', 'Val', 'Leu', 'Pro', 'Gln', 'Arg', 'Arg', 'Arg', 'Ala', 'Asp', 'His', 'Ser', 'Lys', 'Leu', 'Arg', 'Thr', 'Pro', 'Ser', 'Ser', 'Thr', 'Met', 'Ser', 'Pro', 'Gly', 'Lys', 'Phe', 'Leu', 'Glu', 'Trp', 'Ile', 'Ile', 'Lys', 'Gln', 'Arg', 'Val', 'Cys', 'Lys', 'His', 'Leu', 'Glu', 'Lys', 'Ala', 'Ala', 'Val', 'Ser', 'Pro', 'Arg', 'Gly', 'Gln', 'His', 'Cys', 'Ser', 'Lys', 'Cys', 'Thr', 'Ala', 'Phe', 'Leu', 'Phe', 'Val', 'Thr', 'Gly', 'Leu', 'Leu', 'Ala', 'Cys', 'Cys', 'Ala', 'Arg', 'Gly', 'Lys', 'His', 'Ala', 'Ala', 'Gln', 'Cys', 'Ile', 'Ser', 'Ser', 'Gln', 'Gln', 'Asp', 'Phe', 'Asp', 'Gly', 'Phe', '*', 'Gln', 'Asn', 'Leu', 'Val', 'Asp', 'Lys', 'Met', 'Glu', 'Leu', 'Trp', 'Gly', 'Leu', 'Glu', 'Glu', 'Arg', 'Thr', 'Tyr', 'Arg', 'Lys', 'Asn', 'Gln', 'Ser', 'Gln', 'Met', 'Asn', 'His', 'Ser', 'Pro', 'Lys', 'Gly', 'His', 'Ser', '*', 'Thr', 'Met', 'Asp', '*', 'Phe', 'Gln', 'Pro', 'Cys', 'Thr', 'Glu', 'Gly', 'Ser', 'Gly', 'Arg', 'Val', 'His', 'Pro', 'Val', 'His', 'Ser', 'Thr', 'Pro', 'Gly', '*', 'Lys', 'Thr', 'Gly', 'Ala', 'Ser', 'Thr', 'Gln', 'Gly', 'Lys', 'Gly', 'Lys', 'Leu', 'Val', 'Ser', '*', 'Ser', 'Asn', 'Gln', 'Gly', 'Ser', 'Asp', 'Asn', 'Phe', '*', 'Arg', 'Pro', 'Glu', 'Gly', 'Arg', 'Leu', 'Gln', 'Ser', 'Pro', 'Arg', '*', 'Asn', 'Leu', 'Gln', 'Gly', 'Thr', 'Asn', 'Val', 'Lys', 'Pro', 'Asn', 'Ile', '*', 'Val', 'Leu', 'Lys', 'Ile', 'Lys', 'Arg', 'Ile', 'Asn', 'Thr', 'Glu', 'Gly', 'Gly', 'Gly', 'Asn', 'Leu', 'Leu', '*', 'Thr', 'Gln', 'Phe', 'Arg', '*', 'Arg', 'Lys', 'Thr', 'Trp', 'Lys', 'Leu', 'Leu', 'Leu', 'Thr', 'Ile', 'Ser', 'Ser', 'Val', 'Gly', 'Ala', 'Lys', 'Ser', 'Met', 'Leu', 'Ile', 'Gly', 'Ile', 'Lys', 'Arg', 'Gln', '*', 'Asp', 'Leu', 'Arg', 'Ala', 'His', 'Ser', 'Ser', 'Pro', 'Pro', 'Leu', 'Phe', 'Cys', 'Pro', 'Ser', 'Ser', 'Phe', 'Phe', 'Gln', 'Ser', 'Ala', 'Gly', 'Thr', 'Val', 'His', 'Ser', 'Leu', 'Gly', 'Ala', 'Thr', 'Thr', 'Glu', 'Asn', 'Arg', 'Gly', 'Ala', 'Ser', 'Ser', 'Thr', 'Thr', 'Glu', 'Asn', 'Arg', 'Ala', 'Thr', 'Thr', 'Glu', 'Asn', 'Arg', 'Gly', '*', 'Leu', 'Ser', 'Ser', 'Pro', 'Pro', 'Val', 'Ser', 'Ala', 'His', 'Ser', 'Gln', 'Leu', 'Gln', 'Gln', 'Ser', 'Arg', 'Arg', 'Arg', 'Glu', 'His', 'Ser', 'Leu', 'Gln', 'Cys', '*', 'Phe', 'Ala', 'Arg', 'Ser', 'Ser', 'Pro', 'Ala', 'Cys', 'Val', 'Thr', 'Gly', 'His', 'Arg', 'Arg', 'Gln', '*', 'Gly', 'Gln', 'Arg', 'Pro', 'Gly', 'Cys', 'Ala', 'Gly', 'Ala', '*', 'Ala', 'Gly', 'Trp', 'Trp', 'Gly', 'Glu', 'Ser', 'Leu', 'Ser', 'Pro', 'Ala', 'Pro', 'Val', 'Ser', 'Ser', 'Val', 'Gln', 'Glu', 'Glu', 'His', 'Val', '*', 'Gly', 'Asp', 'Gly', 'Phe', 'Lys', 'Ala', 'Gly', 'His', 'Ile', 'Pro', 'Thr', 'Glu', 'Lys', 'Ala', 'His', 'Gly', 'Gln', 'Arg', 'Lys', 'Ala', 'His', '*', 'Leu', 'Val', 'Gln', 'Cys', 'His', 'Arg', 'Arg', 'Gly', 'Lys', 'Trp', 'Arg', 'Arg', 'Arg', 'Gly', 'Gly', 'Gly', 'Ala', 'Pro', 'His', 'Ser', 'Thr', 'Ala', 'Ser', 'Arg', 'His', 'Trp', 'Leu', 'Ser', 'Leu', 'Pro', 'Phe', 'Ile', 'Leu', 'Val', 'Pro', 'Tyr', 'Leu', 'Ser', 'Pro', 'Phe', 'Pro', 'Val', 'Val', 'Val', 'Ser', 'Ser', 'Glu', 'Cys', 'Leu', 'Thr', 'Leu', 'Pro', 'Ser', 'Leu', 'Leu', 'Ala', 'Ser', 'Pro', 'Leu', 'Ser', 'Val', 'Ala', 'Ser', 'Pro', 'Leu', 'Ser', 'Tyr', 'Pro', 'Asp', 'Tyr', 'Asn', 'Asn', 'Ser', 'Phe', 'Trp', 'Val', 'Ser', 'Leu', 'Ala', 'Ser', 'Thr', 'Leu', 'Ser', 'Pro', 'Phe', 'Leu', 'Ser', 'Leu', 'Pro', '*', 'Arg', 'Met', 'Pro', 'Glu', 'Glu', 'Pro', 'Ser', 'Pro', 'Asn', 'Ser', 'Ser', 'Val', 'Pro', 'Ser', 'Leu', 'Pro', 'Cys', 'Ser', 'Lys', 'Ser', 'Asn', 'His', 'Ser', 'Ser', 'Leu', 'Thr', 'Arg', 'Leu', 'Asn', 'Gln', 'Leu', 'Glu', 'Val', 'Leu', 'Ser', '*', 'Val', 'Ile', 'Arg', 'Gly', 'Pro', '*', 'Leu', 'Thr', 'His', 'Pro', 'Asn', 'Ser', 'Ser', 'Leu', 'Thr', 'Ala', 'Leu', 'Pro', 'His', 'Thr', 'Leu', 'Pro', 'Gly', 'Ser', 'Leu', 'Pro', 'Trp', 'His', 'Arg', 'Gly', 'Asp', 'Thr', 'Lys', 'Glu', 'Pro', 'Gly', 'Gln', 'Ser', 'Ser', 'Leu', 'Ser', 'Pro', 'Ile', 'Gln', 'Arg', 'Gly', 'Leu', 'Ala', 'His', 'Arg', 'Leu', 'Thr', 'Glu', 'Ser', 'Gln', 'Pro', 'Leu', 'Met', 'Pro', 'Arg', 'Glu', 'Leu', 'Ser', 'Ala', 'Arg', 'Glu', 'Arg', 'Gln', 'Arg', 'Cys', 'Leu', 'Cys', 'Phe', 'Pro', 'Cys', 'Arg', 'Ser', 'Thr', 'Pro', 'Leu', 'Pro', 'Pro', 'Leu', 'Cys', 'Arg', 'Pro', 'Ala', 'Phe', 'Ala', 'Ala', 'Asp', 'His', 'His', 'Thr', 'Pro', 'Arg', 'Ser', 'Lys', 'Pro', 'His', '*', 'Gly', 'Leu', 'Pro', 'Pro', 'Ser', 'Leu', 'Gln', 'Pro', 'Pro', 'Phe', 'Pro', 'Asp', 'Pro', 'Ala', 'Arg', 'Ala', 'Thr', 'Cys', 'Ile', 'Ser', 'Thr', 'Ser', 'Leu', 'Pro', 'Cys', 'Pro', 'Pro', 'Leu', 'Pro', 'Gly', 'Val', 'Cys', 'Pro', 'Met', 'Trp', 'Ser', 'Lys', 'His', 'Val', 'Val', 'Phe', 'Leu', 'Phe', 'Ser', 'Asn', 'Tyr', 'Phe', 'Leu', 'Phe', 'Thr', 'Gln', 'Ala', 'Met', 'Ala', 'Pro', 'Phe', 'Pro', 'Leu', 'Gly', 'Asn', 'Pro', 'Ser', 'Leu', 'Ser', 'Gln', 'Ala', '*', 'Ser', 'Gln', 'Ser', 'Phe', 'Arg', 'Trp', 'Gly', 'Cys', 'Pro', 'Gln', 'Ser', 'Ser', 'Ser', 'Val', '*', 'Ala', 'Lys', 'Trp', 'Cys', 'Val', 'Ile', 'Val', 'Pro', 'Trp', 'Pro', 'His', '*', 'Trp', 'Ile', 'Leu', 'Gly', '*', 'Thr', '*', 'Gly', 'Pro', 'Ser', 'Gln', 'Val', 'Gly', '*', 'Val', 'Ser', 'Val', 'Ala', 'Ser', 'Gly', 'Gly', 'Ser', 'Gly', 'Asp', 'Thr', 'Gly', 'Gln', 'His', 'Ser', 'Phe', 'Leu', 'Leu', 'Asp', 'Leu', 'Thr', 'Leu', 'Cys', 'His', 'Val', 'Thr', 'Leu', 'Leu', 'Pro', 'Arg', 'Glu', 'His', 'Gly', 'Leu', 'Ser', 'Gly', 'Asn', 'Ala', 'Ala', 'Arg', 'Pro', 'Lys', 'Glu', 'Ala', 'Asn', '*', 'His', 'Gly', 'Arg', 'Lys', 'Ala', 'Lys', 'Pro', 'Gly', 'Pro', 'Glu', 'Asp', 'Ile', 'Ile', 'Leu', 'Ala', 'Leu', 'Thr', 'Pro', 'Lys', 'Ala', 'Ala', 'Leu', 'Leu', 'Ile', 'Gly', '*', 'Phe', 'Leu', 'Leu', 'Ser', 'Leu', 'Val', 'Trp', 'Gly', 'Val', 'Leu', 'Thr', 'Gly', 'Val', 'Pro', 'Pro', 'Ile', 'Leu', 'Thr', 'Asp', 'Phe', 'Ser', 'Pro', 'Leu', '*', 'Thr', 'Leu', 'Arg', 'Ser', 'Pro', 'Arg', 'Gly', 'Ser', 'Cys', '*', 'Gln', 'Leu', 'Thr', 'Ile', 'Asn', 'Leu', 'Ala', 'Leu', 'Cys', 'Val', 'Pro', 'Ile', 'Pro', 'Ala', 'Ser', 'Arg', 'Thr', 'Gln', 'Trp', 'Gln', 'Pro', 'His', 'Asn', 'Trp', 'Tyr', 'Leu', 'Leu', 'Arg', 'Ser', 'Ser', 'Thr', 'Arg', 'Trp', 'Ser', 'Thr', 'Trp', 'Trp', 'Arg', 'Asp', 'Arg', 'Cys', 'Ser', 'Asp', 'Leu', 'Glu', 'Pro', 'Arg', 'Ser', 'Glu', 'Gly', 'Ala', 'Arg', 'Thr', 'Gln', 'Ala', 'Gln', 'Gly', 'Ser', '*', 'Glu', 'Ala', 'Ser', 'Gly', 'Pro', 'Pro', 'Cys', 'Ala', 'Val', 'Pro', 'Ala', 'Ala', 'Trp', 'Arg', 'Thr', 'His', 'Thr', 'Gln', '*', 'Thr', 'Gln', 'His', 'Ser', 'Thr', 'Thr', 'Gln', 'Glu', 'Met', 'Pro', 'Ser', 'Cys', 'Pro', 'Leu', 'Leu', 'Ile', 'Pro', 'Ser', 'Leu', 'Gly', 'Arg', 'Gly', 'His', 'Ala', 'Thr', 'Val', 'Tyr', 'Lys', 'Val', 'Pro', 'Ser', 'Thr', 'Arg', 'Thr', 'Gly', 'Lys', 'Glu', 'Arg', 'Arg', 'Gln', 'Lys', 'Ser', 'Ser', 'Ala', 'Ala', 'Leu', 'Arg', 'Glu', 'Gly', 'Gln', 'Pro', 'Arg', 'Ser', 'Pro', 'His', 'Leu', 'Gly', 'Lys', 'Glu', 'Thr', 'Gln', 'Phe', 'Pro', 'Arg', 'Glu', 'Trp', 'Phe', 'Trp', 'Pro', 'Pro', 'Phe', '*', 'Val', 'Leu', 'Asp', 'Met', 'Gly', 'Trp', 'Pro', '*', 'Ser', 'Gly', 'Ala', 'Asp', 'Gly', 'Ser', '*', 'Arg', 'Pro', 'Ala', 'Ser', 'Ser', 'Ser', 'Leu', 'Gly', 'Val', 'Pro', 'Arg', 'Ala', 'His', 'Leu', 'Ala', 'Gln', 'Arg', '*', 'Ala', 'Gln', 'Lys', 'Val', 'His', 'Pro', 'Ala', 'Leu', 'Cys', 'Tyr', 'Tyr', 'Trp', 'Trp', 'Gln', 'Val', 'Tyr', 'Glu', 'Trp', 'Gln', 'Pro', 'Lys', 'Ala', 'Val', 'Tyr', 'Gly', 'Ser', 'Arg', 'Leu', 'Ser', 'Thr', 'Gly', 'Lys', 'Arg', '*', 'His', 'Phe', 'Leu', 'Lys', 'Ala', 'Ser', '*', 'Val', 'Pro', 'Gly', 'Thr', 'Val', 'Pro', 'Phe', 'Leu', 'Cys', 'Met', 'Phe', '*', 'Leu', 'Ile', '*', 'Tyr', 'Leu', 'Lys', '*', 'Phe', 'Tyr', 'Gln', 'Glu', 'Ala', 'Thr', 'Ile', 'Ile', 'Thr', 'Thr', 'Thr', 'Ser', 'Gln', 'Met', 'Arg', 'Thr', 'Pro', 'Arg', 'Leu', 'Arg', 'Gly', 'Val', 'Gly', 'Leu', 'Pro', 'Lys', 'Val', 'Thr', 'Glu', 'Glu', 'Glu', 'Asn', 'Arg', 'Gly', 'Ala', 'Gly', 'Ser', 'Glu', 'Pro', 'Arg', 'His', 'Gln', 'Leu', 'Gln', 'Gly', 'Asn', 'Pro', 'Ser', 'Val', 'Thr', 'Ser', 'Leu', 'Cys', 'Val', 'Pro', 'Trp', 'Leu', 'Leu', 'Gly', 'His', 'Ser', '*', 'Gln', 'Thr', 'Arg', 'Gly', 'Lys', 'Pro', 'Val', 'Ser', 'Gln', 'Trp', 'Gly', 'Arg', 'Thr', 'Phe', 'Arg', 'Lys', 'Arg', 'Trp', 'Val', 'Pro', 'Ser', 'Trp', '*', 'Gln', 'Lys', 'Arg', 'Arg', 'Leu', 'Gln', 'Ser', 'Glu', 'Gly', 'Ala', 'Gly', 'Ala', 'Pro', 'Gly', 'Leu', 'Ala', 'Thr', 'Thr', 'Arg', 'Glu', 'Gly', 'Thr', 'Gly', 'Gln', 'Gly', 'Trp', 'Leu', 'Gly', 'Pro', 'Arg', 'Glu', 'Ala', 'Pro', 'Glu', 'Ser', 'Gly', 'Ser', 'His', 'Ile', 'Leu', 'Pro', 'Thr', 'Gly', 'Val', 'Tyr', 'His', 'Val', 'Arg', 'His', 'Gly', 'Val', 'Gly', 'Ser', 'Trp', 'Glu', 'Gly', 'Asp', 'Gln', 'Ala', 'Ser', 'Phe', 'Gln', 'Phe', 'Ala', 'Tyr', 'Gly', 'Gln', 'Arg', 'Gln', 'Asp', 'Leu', 'Cys', 'Thr', 'Arg', 'Gln', 'Pro', 'Leu', 'Gly', 'Pro', 'Leu', 'Pro', 'Lys', 'Lys', 'Glu', 'Gln', 'Thr', 'Pro', 'Phe', 'Thr', 'His', 'Ser', 'Cys', '*', 'Ile', 'Asn', 'Thr', 'Glu', '*', 'Ser', 'His', 'Trp', 'Ser', 'Pro', 'Arg', 'Thr', 'Val', 'Arg', 'Gly', 'Gln', 'His', 'Cys', 'Gln', 'Tyr']
    """obs = ['Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Gln', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Asn', 'Pro', 'Lys', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', 'Gln', 'Pro', 'Gln', 'Pro', 'Gln', 'Pro', 'Gln', 'Pro', 'Gln', 'Pro', 'Gln', 'Pro', '*', 'Pro', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Thr', 'Leu', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', 'Leu', 'Thr', 'Pro', 'Asn', 'Pro', 'Asn', 'Pro', 'Asn', 'Pro', 'Asn', 'Pro', 'Asn', 'Pro', 'Asn', 'Pro', 'Asn', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', '*', 'Pro', 'Ser', 'Arg', 'Tyr', 'Pro', 'Gln', 'Pro', 'Ala', 'Arg', 'Pro', 'Pro', 'Gly', 'Ser', 'Asp', 'Leu', 'Arg', 'Arg', 'Thr']
    model2 = HMM(transitions, emissions)
    model2.train(obs)
    print("Model transmission probabilities:\n{}".format(model2.transmission_prob))
    print("Model emission probabilities:\n{}".format(model2.emission_prob))
    # Probability of a new sequence
    new_seq = ['Thr', 'Leu', 'Thr', '*']
    print("Finding likelihood for {}".format(new_seq))
    likelihood = model2.likelihood(new_seq)
    print("Likelihood: {}".format(likelihood))"""

    # Example inputs from Jason Eisner's Ice Cream and Baltimore Summer example
    # http://www.cs.jhu.edu/~jason/papers/#eisner-2002-tnlp
    emission = np.array([[0.7, 0], [0.2, 0.3], [0.1, 0.7]])
    transmission = np.array([ [0, 0, 0, 0], [0.5, 0.8, 0.2, 0], [0.5, 0.1, 0.7, 0], [0, 0.1, 0.1, 0]])
    observations = ['2','3','3','2','3','2','3','2','2','3','1','3','3','1','1',
                    '1','2','1','1','1','3','1','2','1','1','1','2','3','3','2',
                    '3','2','2']
    model = HMM(transmission, emission)
    model.train(observations)
    print("Model transmission probabilities:\n{}".format(model.transmission_prob))
    print("Model emission probabilities:\n{}".format(model.emission_prob))
    # Probability of a new sequence
    new_seq = ['1', '2', '3']
    print("Finding likelihood for {}".format(new_seq))
    likelihood = model.likelihood(new_seq)
    print("Likelihood: {}".format(likelihood))
