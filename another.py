import pandas as pd
import numpy as np
 
 
def forward(V, a, b, initial_distribution):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    print(b[:,2])
    alpha[0,:] = initial_distribution * b[:,]
 
    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]
 
    return alpha
 
 
def backward(V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0]))
 
    # setting beta(T) = 1
    beta[V.shape[0] - 1] = np.ones((a.shape[0]))
 
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])
 
    return beta
 
 
def baum_welch(V, a, b, initial_distribution, n_iter=100):
    M = a.shape[0]
    T = len(V)
 
    for n in range(n_iter):
        alpha = forward(V, a, b, initial_distribution)
        beta = backward(V, a, b)
 
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator
 
        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
 
        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
 
        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1)
 
        b = np.divide(b, denominator.reshape((-1, 1)))
 
    return (a, b)
 

def viterbi(V, a, b, initial_distribution):
    T = V.shape[0]
    M = a.shape[0]
    
    omega = np.zeros((21, 4))
    
    omega = np.log(initial_distribution*b)
    #omega[0,:] = np.log(initial_distribution * b[:,V[0]])
    print(omega)
    prev = np.zeros((T - 1, M))

    for t in range(0,21):
        for j in range(4):
            # Same as Forward Probability
            probability = omega[t - 1] + np.log(a[:,j]) + np.log(b[j,:])
 
            # This is our most probable state given previous state at time t (1)
            prev[t - 1, j] = np.argmax(probability)
 
            # This is the probability of the most probable state (2)
            omega[t, j] = np.max(probability)
 
    # Path Array
    S = np.zeros(T)
 
    # Find the most probable last hidden state
    last_state = np.argmax(omega[21 - 1, :])
 
    S[0] = last_state
 
    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1
 
    # Flip the path array since we were backtracking
    S = np.flip(S, axis=0)
 
    # Convert numeric values to actual hidden states
    result = []
    for s in S:
        if s == 0:
            result.append("A")
        else:
            result.append("B")
 
    return result
 
 
data = open('./data_scaffolds.txt','r')

import ast
with open('data_scaffolds.txt', 'r') as f:
	x = ast.literal_eval(f.read())
V = []

for i in x:
    V.append(x[1])
V = np.array(V)
#V = data['Sequence'].values
 
# Transition Probabilities
a = np.array([[0.23971024, 0.24234442, 0.37141916, 0.14652618],
       [0.27374847, 0.35262515, 0.09084249, 0.28278388],
       [0.20955596, 0.28901584, 0.33575695, 0.16567125],
       [0.1350871 , 0.28546036, 0.37611091, 0.20334163]])
a = a / np.sum(a, axis=1)
 
# Emission Probabilities
b = np.array([[0.        , 0.        , 0.        , 0.15991238],
       [0.        , 0.        , 0.278327  , 0.        ],
       [0.22910217, 0.08630528, 0.        , 0.        ],
       [0.08978328, 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.06692015, 0.        ],
       [0.        , 0.        , 0.        , 0.18181818],
       [0.        , 0.17475036, 0.        , 0.        ],
       [0.        , 0.        , 0.14448669, 0.        ],
       [0.        , 0.        , 0.33840304, 0.        ],
       [0.        , 0.11840228, 0.        , 0.        ],
       [0.09287926, 0.        , 0.        , 0.        ],
       [0.        , 0.2831669 , 0.        , 0.07776561],
       [0.124871  , 0.        , 0.        , 0.        ],
       [0.04747162, 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.11938664],
       [0.        , 0.33737518, 0.        , 0.        ],
       [0.16718266, 0.        , 0.        , 0.27491785],
       [0.24871001, 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.13253012],
       [0.        , 0.        , 0.        , 0.05366922],
       [0.        , 0.        , 0.17186312, 0.        ]])
b = b / np.sum(b, axis=1).reshape((-1, 1))
 
# Equal Probabilities for the initial distribution
initial_distribution = np.array([0.0476])
 
#a, b = baum_welch(V, a, b, initial_distribution, n_iter=100)
 
print(viterbi(V, a, b, initial_distribution))
