import simplehmm # Define HMM state list and observation list

test_hmm_states = ['title', 'givenname', 'surname']
test_hmm_observ = ['TI', 'GM', 'GF', 'SN', 'UN']

# Some example training records (one per line) with state/tag pairs

train_data = [[('title','TI'),('givenname','GF'),('surname','SN')],
              [('givenname','GM'),('surname','UN')],
              [('title','UN'),('givenname','GM'),('surname','UN')],
              [('title','TI'),('givenname','SN'),('surname','SN')],
              [('givenname','GM'),('surname','SN')],
              [('title','TI'),('givenname','GF'),('surname','SN')],
              [('title','TI'),('surname','SN'),('givenname','GM')],
              [('surname','UN'),('givenname','UN')],
              [('givenname','GF'),('surname','GF'),('surname','SN')]]

# Some test examples (observation (tag) sequences), one per line

test_data = [['TI','GM','SN'],
             ['UN','SN'],
             ['TI','UN','UN'],
             ['TI','GF','UN'],
             ['UN','UN','UN','UN'],
             ['TI','GM','UN','SN'],
             ['GF','UN']]

# Initialise a new HMM and train it

test_hmm = simplehmm.hmm('Test HMM', test_hmm_states, test_hmm_observ)
test_hmm.train(train_data)  # Train the HMM

test_hmm.check_prob()  # Check its probabilities
test_hmm.print_hmm()   # Print it out

# Apply the Viterbi algorithm to each sequence of the test data

for test_rec in test_data:
  [state_sequence, sequence_probability] = test_hmm.viterbi(test_rec)

# Initialise and train a second HMM using the same training data and
# applying Laplace smoothing

test_hmm2 = simplehmm.hmm('Test HMM 2', test_states, test_observ)
test_hmm2.train(train_data, smoothing='laplace')

# Save the second  HMM into a text file

test_hmm2.save_hmm('testhmm2.hmm')

# Initialise a third HMM, then load the previously saved HMM into it

test_hmm3 = simplehmm.hmm('Test HMM 3',  ['dummy'], ['dummy'])
test_hmm3.load_hmm('testhmm2.hmm')
test_hmm3.print_hmm()  # Print it out
