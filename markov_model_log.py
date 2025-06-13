import pandas as pd
import numpy as np
from scipy.special import logsumexp
import time

class MarkovModel:
    def __init__(self, 
                 data_file_path,
                 save_data_verbose_path,
                 save_data_summary_path,
                 params: list = [0.7, 0.7, 0.7, 0.3, 0.6, 0.4] 
                 ):
        """
        Initialize the Markov Model with a data file path and parameters.
        :param data_file_path: Path to the CSV file containing the data.
        :param params: List of parameters to initialize the model.
        :param save_data_verbose_path: Path to save verbose data.
        :param save_data_summary_path: Path to save summary data.
        """
        self.data_file_path = data_file_path
        self.transition_matrix_log = {}
        self.state_probabilities_log = {}
        self.initial_probabilities_log = {}

        self.df = pd.read_csv(self.data_file_path)
        self.initialize_params(params)

        self.save_data_verbose_path = save_data_verbose_path
        self.save_data_summary_path = save_data_summary_path

    def data_preprocessing(self, verbose=False, worker_id=None, date=None):
        """
        Preprocess the data to extract sequences of observations.
        :param verbose: If True, prints preprocessing information.
        :param worker_id: Optional worker ID that will filter the data.
        :param date: Optional date to filter the data.
        """
        # Sort by timestamp for each user
        df_sorted = self.df.sort_values(['user_id', 'created_at'])

        self.df_organized = df_sorted.copy()
        if worker_id is not None:
            self.df_organized = self.df_organized[
                (self.df_organized['user_id'] == worker_id) & 
                (self.df_organized['date'] == date)
            ]
        # Create sequences of responses (1 for correct, 0 for incorrect)
        self.df_organized = self.df_organized.groupby('user_id')['disagree'].agg(
            lambda x: list(x*-1 + 1)
        )

        # Convert the sequences to 'correct'/'wrong' format
        self.df_organized = self.df_organized.apply(
            lambda x: ['correct' if i else 'wrong' for i in x]
        )
        print(self.df_organized.shape)
        print(self.df_organized)
        if verbose:
            print("Data Preprocessing Complete.")
            print(f"Number of points: {len(self.df_organized)}")
            # print(f"Example sequence: {self.df_organized.iloc[0]}")


        
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
    
    def save_data_to_summary_file(self, message="", newline=False):
        """ 
        Save the current model parameters to a summary file.
        :param message: Optional message to include in the summary file.
        :param newline: If True, adds a newline after the message.
        """
        with open(self.save_data_summary_path, 'a') as f:
            if (message != ""):
                f.write(f"{message}\n")
            f.write(f"Transition Matrix: {self.dict_str(self.transition_matrix_log, 8)}\n")
            f.write(f"State Probabilities: {self.dict_str(self.state_probabilities_log, 8)}\n")
            f.write(f"Initial Probabilities: {self.dict_str(self.initial_probabilities_log, 8)}\n")
            if (newline): f.write("\n")

    def save_data_to_verbose_file(self, iteration, time, message="", newline=False):
        """ 
        Save the current model parameters to a verbose file.
        :iteration: Current iteration number.
        :time: Time taken for the current iteration.
        :param message: Optional message to include in the verbose file.
        """
        with open(self.save_data_verbose_path, 'a') as f:
            if (message != ""):
                f.write(f"{message}\n")

            f.write(f"Iteration {iteration}, " +
                    f"Time: {time:.2f} s, " +
                    f"Trans Mat: {self.dict_str(self.transition_matrix_log, 8)}, " +
                    f"State Probs: {self.dict_str(self.state_probabilities_log, 8)}, " +
                    f"Init Probs: {self.dict_str(self.initial_probabilities_log, 8)}\n")
            if (newline): f.write("\n")
        

    # Some helper functions
    def trans_mat_g_to_b_log(self):
        """Calculate the log probability of transitioning from a good state to a bad state."""
        return np.log(1 - np.exp(self.transition_matrix_log["good"]))
    def trans_mat_b_to_g_log(self):
        """Calculate the log probability of transitioning from a bad state to a good state."""
        return np.log(1 - np.exp(self.transition_matrix_log["bad"]))
    def state_prob_g_to_w_log(self):
        """Calculate the log probability of a wrong observation in a good state."""
        return np.log(1 - np.exp(self.state_probabilities_log["good"]))
    def state_prob_b_to_w_log(self):
        """Calculate the log probability of a wrong observation in a bad state."""
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
        backward_log[N-1] = 0

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

    def baum_welch(self, save_data, max_iterations=100, tolerance=1e-6):
        """
        Implements the Baum-Welch algorithm for parameter estimation.
        :param max_iterations: Maximum number of iterations
        :param tolerance: Convergence tolerance
        """
        # Use the preprocessed sequences from df_organized
        sequences = self.df_organized.tolist()
        
        print(f"Iteration 0: Transition Matrix: {self.dict_str(self.transition_matrix_log)}",
        f"State Probabilities: {self.dict_str(self.state_probabilities_log)}",
        f"Initial Probabilities: {self.dict_str(self.initial_probabilities_log)}")

        if save_data:
            self.save_data_to_summary_file(message=f"Iteration 0, tol: {tolerance}")
            self.save_data_to_verbose_file(0, 0)
        

        for iteration in range(max_iterations):
            start_time = time.time()
            old_params = self.get_params_exp()
            
            # Accumulators for new parameters
            new_transition_log = {"good": -np.inf, "bad": -np.inf}
            new_emission_log = {"good": -np.inf, "bad": -np.inf}
            new_initial_log = {"good": -np.inf, "bad": -np.inf}
            denominators_log = {"good": -np.inf, "bad": -np.inf}
            denominators_transition_log = {"good": -np.inf, "bad": -np.inf}
            
            # Process each sequence
            for sequence in sequences:
                forward_log, backward_log = self.forward_backward_log(sequence)
                
                # Check gamma probabilities
                gamma_denom = logsumexp(forward_log + backward_log, axis=1, keepdims=True)
                gamma_log = forward_log + backward_log - \
                           gamma_denom
                
                # Update initial probabilities
                new_initial_log["good"] = logsumexp([new_initial_log["good"], gamma_log[0, 0]])
                new_initial_log["bad"] = logsumexp([new_initial_log["bad"], gamma_log[0, 1]])
                
                # Update transition and emission probabilities
                for t in range(len(sequence)):
                    if t < len(sequence) - 1:
                        # Update transition probabilities
                        next_emission_good_log = self.emission_prob_log(sequence, t+1, "good")
                        next_emission_bad_log = self.emission_prob_log(sequence, t+1, "bad")
                        new_transition_log["good"] = logsumexp([new_transition_log["good"], 
                                                               forward_log[t, 0] +
                                                                backward_log[t+1, 0] + 
                                                               self.transition_matrix_log["good"] + 
                                                               next_emission_good_log - 
                                                               gamma_denom[t, 0]])
                        new_transition_log["bad"] = logsumexp([new_transition_log["bad"], 
                                                              forward_log[t, 1] + 
                                                              backward_log[t+1, 1] + 
                                                              self.transition_matrix_log["bad"] + 
                                                              next_emission_bad_log - 
                                                              gamma_denom[t,0]])
                        denominators_transition_log["good"] = logsumexp([denominators_transition_log["good"], 
                                                                         gamma_log[t, 0]])
                        denominators_transition_log["bad"] = logsumexp([denominators_transition_log["bad"], 
                                                                        gamma_log[t, 1]])
                    # Update emission probabilities
                    if sequence[t] == 'correct':
                        new_emission_log["good"] = logsumexp([new_emission_log["good"], 
                                                              gamma_log[t, 0]])
                        new_emission_log["bad"] = logsumexp([new_emission_log["bad"], 
                                                             gamma_log[t, 1]])
                    

                    denominators_log["good"] = logsumexp([denominators_log["good"], 
                                                          gamma_log[t, 0]])
                    denominators_log["bad"] = logsumexp([denominators_log["bad"], 
                                                         gamma_log[t, 1]])
            
            # Update parameters
            self.transition_matrix_log["good"] = (new_transition_log["good"] - 
                                                  denominators_transition_log["good"])
            self.transition_matrix_log["bad"] = (new_transition_log["bad"] - 
                                                 denominators_transition_log["bad"]) 
            self.state_probabilities_log["good"] = (new_emission_log["good"] - 
                                                    denominators_log["good"]) 
            self.state_probabilities_log["bad"] = (new_emission_log["bad"] - 
                                                   denominators_log["bad"])
            self.initial_probabilities_log["good"] = new_initial_log["good"] - \
                np.log(len(sequences))
            self.initial_probabilities_log["bad"] = new_initial_log["bad"] - \
                np.log(len(sequences))
            
            print(f"Iteration {iteration + 1}:",
                  f"Time: {time.time() - start_time:.2f} s",
                  f"Trans Mat: {self.dict_str(self.transition_matrix_log)}",
                  f"State Probs: {self.dict_str(self.state_probabilities_log)}",
                  f"Init Probs: {self.dict_str(self.initial_probabilities_log)}")
            
            if save_data:
                self.save_data_to_verbose_file(iteration + 1, time.time() - start_time)
        
            # Check convergence
            if np.allclose(old_params, self.get_params_exp(), rtol=tolerance):
                break
        if save_data:
            self.save_data_to_summary_file(
                message=f"Iteration {iteration + 1}, tol: {tolerance}", 
                newline=True)
            with open(self.save_data_verbose_path, 'a') as f:
                f.write(f"\n")
        
    def viterbi_log(self, sequence, params=None):
        """
        Implements the Viterbi algorithm to find the most likely sequence of states.
        :param sequence: List of observations ('correct' or 'wrong')
        :param params: Optional List of parameters to use for the Viterbi algorithm.
        :type params: list
        :return: Most likely sequence of states
        """
        N = len(sequence)
        viterbi_log = np.zeros((N, 2))
        backpointer = np.zeros((N, 2), dtype=int) # To store backpointers.

        # Initialize the transition matrix and state probabilities
        if params != None:
            self.initialize_params(params)

        # Initialization step
        viterbi_log[0, 0] = self.initial_probabilities_log["good"] + \
            self.emission_prob_log(sequence, 0, "good")
        viterbi_log[0, 1] = self.initial_probabilities_log["bad"] + \
            self.emission_prob_log(sequence, 0, "bad")
        backpointer[0, 0] = 0
        backpointer[0, 1] = 1

        # Recursion step
        for t in range(1, N):
            # Good state
            backpointer[t,0] = np.argmax([
                viterbi_log[t-1, 0] + self.transition_matrix_log["good"],
                viterbi_log[t-1, 1] + self.trans_mat_b_to_g_log()
            ])
            backpointer[t, 1] = np.argmax([
                viterbi_log[t-1, 0] + self.trans_mat_g_to_b_log(),
                viterbi_log[t-1, 1] + self.transition_matrix_log["bad"]
            ])
            viterbi_log[t, 0] = (viterbi_log[t-1, backpointer[t, 0]] +
                                 self.emission_prob_log(sequence, t, "good"))
            viterbi_log[t, 1] = (viterbi_log[t-1, backpointer[t, 1]] +
                                 self.emission_prob_log(sequence, t, "bad"))
        # Backtrack to find the most likely sequence of states
        best_path = np.zeros(N, dtype=int)
        best_path[N-1] = np.argmax(viterbi_log[N-1])
        for t in range(N-2, -1, -1):
            best_path[t] = backpointer[t+1, best_path[t+1]]
        # Convert the best path to 'good'/'bad' states
        best_path_states = ['good' if state == 0 else 'bad' for state in best_path]
        return best_path_states


    def get_params_exp(self):
        """Helper function to get all parameters as a numpy array"""
        return np.array([
            np.exp(self.transition_matrix_log["good"]),
            np.exp(self.transition_matrix_log["bad"]),
            np.exp(self.state_probabilities_log["good"]),
            np.exp(self.state_probabilities_log["bad"]),
            np.exp(self.initial_probabilities_log["good"]),
            np.exp(self.initial_probabilities_log["bad"])
        ])

    def dict_str(self, d, n=6):
        """Helper function to print dictionary in a readable format
        :param d: Dictionary to convert to string
        :param n: Number of decimal places to round to
        :return: String representation of the dictionary
        """

        return_string = ""
        for key, value in d.items():
            return_string += f"{key}: {np.exp(value):.{n}f} "
        return return_string[:-1]  # Remove trailing space

if __name__ == "__main__":
    # data_file_path = "/home/jgv555/CS/ResSum2025/drive-download-20250502T210721Z-1-001/ECPD/answers_revised2.csv"
    save_data_summary_path = "/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/DataSummary/ECPD_full_summary.csv"
    save_data_verbose_path = "/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/DataSummary/ECPD_full_verbose.csv"
    # data_file_path = "/home/jgv555/CS/ResSum2025/drive-download-20250502T210721Z-1-001/ECPD/test.csv"
    data_file_path = "/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/fake_data.csv"
    # model = MarkovModel(data_file_path, [0.7, 0.3, 0.7, 0.3, 0.6, 0.4])
    # model = MarkovModel(data_file_path, [0.9, 0.7, 0.6, 0.4, 0.5, 0.5])

    # Normal Training

    # model = MarkovModel(data_file_path=data_file_path, 
    #                     save_data_verbose_path=save_data_verbose_path,
    #                     save_data_summary_path=save_data_summary_path, 
    #                     params=[0.99849304, 0.99607914, 0.97798713, 0.88902691, 0, 1]) # Iteration 68: Time: 242.63 s Trans Mat: good: 0.99582 bad: 0.94148 State Probs: good: 0.98495 bad: 0.58401 Init Probs: good: 0.77121 bad: 0.22879, 
    # model.data_preprocessing(verbose=True, 
                            #  worker_id="955eb227-5421-470a-ae87-1b210a94bcfb",
                            #  date="2023-01-19")
    # model.data_preprocessing(verbose=True)
    
    # model.baum_welch(save_data=True)

    # Testing the Viterbi algorithm
    model = MarkovModel(data_file_path=data_file_path,
                        save_data_verbose_path=save_data_verbose_path,
                        save_data_summary_path=save_data_summary_path,
                        params = [0]*6)  # Initialize with zeros for testing
    model.data_preprocessing(verbose=True)
    data_sequence = model.viterbi_log(model.df_organized.iloc[0],
                      params = [0.9995, 0.998, 0.95, 0.6, 0.6, 0.4],
                      )  # Use the same parameters as in the data generation
    
    # Compare data_sequence
    # Create a new column in the original DataFrame for predicted states
    model.df['predicted_state'] = None  # Initialize the new column
    
    # Get the user_id for the first sequence
    first_user_id = model.df['user_id'].iloc[0]
    
    # Add predicted states for this user
    mask = model.df['user_id'] == first_user_id
    model.df.loc[mask, 'predicted_state'] = data_sequence
    
    # Save the updated DataFrame to CSV
    model.df.to_csv('predictions.csv', index=False)
    
    print("Predicted states:", data_sequence)
    print("Actual observations:", model.df_organized.iloc[0])