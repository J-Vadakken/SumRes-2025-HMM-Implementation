import pandas as pd
import numpy as np
from scipy.special import logsumexp
class MarkovModel:
    def __init__(self, 
                 data_file_path, 
                 params: list = [0.7, 0.7, 0.7, 0.3, 0.6, 0.4] 
                 ):
        """
        Initialize the Markov Model with a data file path and parameters.
        :param data_file_path: Path to the CSV file containing the data.
        :param params: List of parameters to initialize the model.
        """
        self.data_file_path = data_file_path
        self.transition_matrix_log = {}
        self.state_probabilities_log = {}
        self.initial_probabilities_log = {}
        

        self.df = pd.read_csv(self.data_file_path)
        self.initialize_params(params)

    def data_preprocessing(self):
        """
        Preprocess the data to extract sequences of observations.
        This function should be implemented based on the specific structure 
        of the CSV file.
        """
        # Sort by timestamp for each user
        self.df = self.df.sort_values(['user_id', 'created_at'])

        # Create sequences of responses (1 for correct, 0 for incorrect)
        self.df_organized = self.df.groupby('user_id')['disagree'].agg(
            lambda x: list(x*-1 + 1)
        )

        # Convert the sequences to 'correct'/'wrong' format
        self.df_organized = self.df_organized.apply(
            lambda x: ['correct' if i else 'wrong' for i in x]
        )


    def initialize_params(self, params: list = [0.7, 0.7, 0.7, 0.3, 0.6, 0.4]):
        """
        Initialize the model parameters.
        :param params: List of parameters to initialize.
        """
        self.transition_matrix_log["good"] = np.log(params[0])
        self.transition_matrix_log["bad"] = np.log(params[1])
        self.state_probabilities_log["good"] = np.log(params[2])
        self.state_probabilities_log["bad"] = np.log(params[3])
        self.initial_probabilities_log["good"] = np.log(params[4])
        self.initial_probabilities_log["bad"] = np.log(params[5])

        """
        self.transition_matrix["good"] represents the log of the probability 
        of transitioning to a good state from a good state. Denote this as 
        P(G|G). Note P(B|G) = 1 - P(G|G).
        self.transition_matrix["bad"] represents the log of the probability 
        of transitioning to a bad state from a bad state. Denote this as 
        P(B|B). Note P(G|B) = 1 - P(B|B).
        self.state_probabilities["good"] represents the log of the probability 
        of getting a label correct from a good state. Denote this as P(C|G). 
        Note P(W|B) = 1 - P(C|G). (Where C is correct and W is wrong)
        self.state_probabilities["bad"] represents the log of the probability 
        of getting a label correct from a bad state. Denote this as P(C|B). 
        Note P(W|B) = 1 - P(C|B).
        self.initial_probabilities["good"] represents the log of the 
        probability of starting in a good state. Denote this as P(G). 
        Note P(B) = 1 - P(G).
        self.initial_probabilities["bad"] represents the log of the probability
        of starting in a bad state. Denote this as P(B). Note P(G) = 1 - P(B).
        """
    
    # Some helper functions
    def trans_mat_g_to_b_log(self):
        """Calculate the log probability of transitioning from a good state to a bad state."""
        return np.log(1 - np.exp(self.transition_matrix_log["good"]))
    def trans_mat_b_to_g_log(self):
        """Calculate the log probability of transitioning from a bad state to a good state."""
        return np.log(1 - np.exp(self.transition_matrix_log["bad"]))
    def state_prob_g_to_w_log(self):
        """Calculate the log probability of transitioning from a good state to a wrong observation."""
        return np.log(1 - np.exp(self.state_probabilities_log["good"]))
    def state_prob_b_to_w_log(self):
        """Calculate the log probability of transitioning from a bad state to a wrong observation."""
        return np.log(1 - np.exp(self.state_probabilities_log["bad"]))
    def emission_prob_log(self, sequence, t, state):
        """
        Calculate the emission probability for a given state and observation.
        :param sequence: List of observations ('correct' or 'wrong')
        :param t: Time step in the sequence
        :param state: Current state ('good' or 'bad')
        :return: Log probability of the emission
        """
        if sequence[t] == 'correct':
            return self.state_probabilities_log[state]
        else:
            return np.log(1 - np.exp(self.state_probabilities_log[state]))

    def forward_backward_log(self, sequence):
        """
        Implements the forward-backward algorithm for a sequence of observations.
        :param sequence: List of observations ('correct' or 'wrong')
        :return: log of Forward probabilities, log of backward probabilities
        """
        N = len(sequence)
        forward_log = np.zeros((N, 2))  # 2 states: good (0) and bad (1)
        backward_log = np.zeros((N, 2))

        # Forward pass
        # Initialize first step
        forward_log[0, 0] = self.initial_probabilities_log["good"] + \
            self.emission_prob_log(sequence, 0, "good")
        forward_log[0, 1] = self.initial_probabilities_log["bad"] + \
            self.emission_prob_log(sequence, 0, "bad") 

        # Recursion step
        for t in range(1, N):
            # Good state
            forward_log[t, 0] = logsumexp([forward_log[t-1, 0] + 
                                          self.transition_matrix_log["good"],
                                          forward_log[t-1, 1] + 
                                          self.trans_mat_b_to_g_log()]) + \
                self.emission_prob_log(sequence, t, "good")
            # Bad state
            forward_log[t, 1] = logsumexp([forward_log[t-1, 0] + 
                                          self.trans_mat_g_to_b_log(),
                                          forward_log[t-1, 1] +
                                          self.transition_matrix_log["bad"]]) + \
                self.emission_prob_log(sequence, t, "bad")

        # Backward pass
        # Initialize last step
        backward_log[N-1] = 1.0

        # Recursion step
        for t in range(N-2, -1, -1):
            # Good state
            backward_log[t, 0] = logsumexp([backward_log[t+1, 0] + 
                                           self.transition_matrix_log["good"] + 
                                           self.emission_prob_log(sequence, t+1, "good"),
                                           backward_log[t+1, 1] +
                                           self.trans_mat_g_to_b_log() +
                                           self.emission_prob_log(sequence, t+1, "bad")])
            # Bad state
            backward_log[t, 1] = logsumexp([backward_log[t+1, 0] +
                                           self.trans_mat_b_to_g_log() + 
                                           self.emission_prob_log(sequence, t+1, "good"),
                                           backward_log[t+1, 1] +
                                           self.transition_matrix_log["bad"] +
                                           self.emission_prob_log(sequence, t+1, "bad")])

        return forward_log, backward_log

    def baum_welch(self, max_iterations=100, tolerance=1e-6):
        """
        Implements the Baum-Welch algorithm for parameter estimation.
        :param max_iterations: Maximum number of iterations
        :param tolerance: Convergence tolerance
        """
        # Use the preprocessed sequences from df_organized
        sequences = self.df_organized.tolist()
        
        for iteration in range(max_iterations):
            old_params = self.get_params()
            
            # Accumulators for new parameters
            new_transition_log = {"good": 0, "bad": 0}
            new_emission_log = {"good": 0, "bad": 0}
            new_initial_log = {"good": 0, "bad": 0}
            denominators_log = {"good": 0, "bad": 0}
            denominators_transition_log = {"good": 0, "bad": 0}
            
            # Process each sequence
            for sequence in sequences:
                forward_log, backward_log = self.forward_backward_log(sequence)
                
                # Check gamma probabilities
                gamma_log = forward_log + backward_log - \
                           logsumexp(forward_log + backward_log, axis=1, keepdims=True)
                
                # Update initial probabilities
                new_initial_log["good"] = logsumexp([new_initial_log["good"], gamma_log[0, 0]])
                new_initial_log["bad"] += logsumexp([new_initial_log["bad"], gamma_log[0, 1]])
                
                # Update transition and emission probabilities
                for t in range(len(sequence)):
                    if t < len(sequence) - 1:
                        # Update transition probabilities
                        next_emission_good_log = self.emission_prob_log(sequence, t+1, "good")
                        next_emission_bad_log = self.emission_prob_log(sequence, t+1, "bad")# new_transition["good"] += (gamma[t, 0] * self.transition_matrix["good"] * next_emission_good)
                        # new_transition["bad"] += gamma[t, 1] * self.transition_matrix["bad"] * next_emission_bad 
                        # new_transition["good"] += (gamma[t, 0] * self.transition_matrix["good"] * next_emission_good) / (gamma[t,0] + gamma[t,1])
                        # new_transition["bad"] += gamma[t, 1] * self.transition_matrix["bad"] * next_emission_bad / (gamma[t,0] + gamma[t,1])
                        new_transition_log["good"] = logsumexp([new_transition_log["good"], 
                                                               forward_log[t, 0] + backward_log[t+1, 0] + 
                                                               self.transition_matrix_log["good"] + next_emission_good_log - 
                                                               logsumexp([gamma_log[t, 0], gamma_log[t, 1]])])
                        new_transition_log["bad"] = logsumexp([new_transition_log["bad"], 
                                                              forward_log[t, 1] + backward_log[t+1, 1] + 
                                                              self.transition_matrix_log["bad"] + next_emission_bad_log - 
                                                              logsumexp([gamma_log[t, 0], gamma_log[t, 1]])])
                        denominators_transition_log["good"] = logsumexp([denominators_transition_log["good"], gamma_log[t, 0]])
                        denominators_transition_log["bad"] = logsumexp([denominators_transition_log["bad"], gamma_log[t, 1]])
                    # Update emission probabilities
                    if sequence[t] == 'correct':
                        new_emission_log["good"] = logsumexp([new_emission_log["good"], gamma_log[t, 0]])
                        new_emission_log["bad"] = logsumexp([new_emission_log["bad"], gamma_log[t, 1]])
                    

                    denominators_log["good"] = logsumexp([denominators_log["good"], gamma_log[t, 0]])
                    denominators_log["bad"] = logsumexp([denominators_log["bad"], gamma_log[t, 1]])
            
            # Update parameters
            self.transition_matrix_log["good"] = (new_transition_log["good"] - denominators_transition_log["good"]) #if denominators_transition["good"] > 0 else self.transition_matrix["good"]
            self.transition_matrix_log["bad"] = (new_transition_log["bad"] / denominators_transition_log["bad"]) #if denominators_transition["bad"] > 0 else self.transition_matrix["bad"]
            self.state_probabilities_log["good"] = (new_emission_log["good"] / denominators_log["good"]) #if denominators_log["good"] > 0 else self.state_probabilities["good"]
            self.state_probabilities_log["bad"] = (new_emission_log["bad"] / denominators_log["bad"]) #if denominators["bad"] > 0 else self.state_probabilities["bad"]
            self.initial_probabilities_log["good"] = new_initial_log["good"] - np.log(len(sequences))
            self.initial_probabilities_log["bad"] = new_initial_log["bad"] - np.log(len(sequences))
            
            print(f"Iteration {iteration + 1}: Transition Matrix: {self.dict_str(self.transition_matrix_log)}, \
                  State Probabilities: {self.dict_str(self.state_probabilities_log)}, \
                  Initial Probabilities: {self.dict_str(self.initial_probabilities_log)}")
        
            # Check convergence
            if np.allclose(old_params, self.get_params(), rtol=tolerance):
                break

    def get_params(self):
        """Helper function to get all parameters as a numpy array"""
        return np.array([
            np.exp(self.transition_matrix_log["good"]),
            np.exp(self.transition_matrix_log["bad"]),
            np.exp(self.state_probabilities_log["good"]),
            np.exp(self.state_probabilities_log["bad"]),
            np.exp(self.initial_probabilities_log["good"]),
            np.exp(self.initial_probabilities_log["bad"])
        ])

    def dict_str(self, d):
        """Helper function to print dictionary in a readable format"""
        return_string = ""
        for key, value in d.items():
            return_string += f"{key}: {np.exp(value):.3f} "
        return return_string[:-1]  # Remove trailing space

if __name__ == "__main__":
    data_file_path = "/home/jgv555/CS/ResSum2025/drive-download-20250502T210721Z-1-001/ECPD/answers_revised2.csv"
    data_file_path = "/home/jgv555/CS/ResSum2025/drive-download-20250502T210721Z-1-001/ECPD/test.csv"
    model = MarkovModel(data_file_path)
    model.data_preprocessing()
    print(model.df_organized.head(3))  # Display the first few sequences
    model.baum_welch()
