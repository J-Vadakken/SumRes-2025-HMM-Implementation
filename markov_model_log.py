import pandas as pd
import numpy as np
from scipy.special import logsumexp
import time
import matplotlib.pyplot as plt
import matplotlib.dates

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
        self.worker_id = None
        self.date = None

        self.converged = False # Flag to indicate if the model has converged

    def preprocess_data(self, verbose=False, worker_id=None, date=None):
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

        self.worker_id = worker_id
        self.date = date

    def preprocess_data_continuous_activity(self, min_sequence_length=5, max_time_gap=60, verbose=False, worker_id=None, date=None):
        """
        Process and filter data based on time gaps and minimum sequence length.
        
        :param min_sequence_length: Minimum length of sequences to keep
        :param max_time_gap: Maximum time gap (in seconds) between annotations
        :param verbose: If True, prints preprocessing information
        :param worker_id: Optional worker ID that will filter the data
        :param date: Optional date to filter the data
        """
        # Sort by timestamp for each user
        df_sorted = self.df.sort_values(['user_id', 'created_at'])
        
        # Filter by worker_id and date if specified
        df_filtered = df_sorted.copy()
        if worker_id is not None:
            df_filtered = df_filtered[
                (df_filtered['user_id'] == worker_id) & 
                (df_filtered['date'] == date)
            ]

        # Convert timestamps to datetime
        df_filtered['created_at'] = pd.to_datetime(df_filtered['created_at'])
        
        # Initialize list to store sequences
        sequences = []
        
        # Process each user's data
        for user_id in df_filtered['user_id'].unique():
            user_data = df_filtered[df_filtered['user_id'] == user_id].copy()
            
            # Calculate time differences between consecutive annotations
            time_diffs = user_data['created_at'].diff().dt.total_seconds()
            
            # Find breaks in continuous activity
            break_points = time_diffs[time_diffs > max_time_gap].index.tolist()
            
            # Add start and end points to break points
            all_points = [user_data.index[0]] + break_points + [user_data.index[-1] + 1]
            
            # Split into sequences based on break points
            for i in range(len(all_points) - 1):
                start = all_points[i]
                end = all_points[i + 1]
                
                # Only keep sequences that meet minimum length requirement
                if end - start < min_sequence_length:
                    continue

                # Get sequence between break points
                sequence = user_data.loc[start:end-1]['disagree'].tolist()
                sequence = [1 if x == 0 else 0 for x in sequence]  # Convert to 1 for correct, 0 for wrong
                
                sequences.append(['correct' if x == 1 else 'wrong' for x in sequence])
        
        # Store processed sequences
        self.df_organized = pd.Series(sequences)
        
        if verbose:
            print("Data Processing Complete")
            if len(sequences) == 0:
                print("No sequences found after filtering.")
                return False
            print(f"Number of sequences: {len(sequences)}")
            print(f"Average sequence length: {np.mean([len(seq) for seq in sequences]):.2f}")
            print(f"Max sequence length: {max([len(seq) for seq in sequences])}")
            print(f"Min sequence length: {min([len(seq) for seq in sequences])}")

        return len(sequences) > 0
        
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

    def baum_welch(self, save_data, max_iterations=300, tolerance=1e-6):
        """
        Implements the Baum-Welch algorithm for parameter estimation.
        :param max_iterations: Maximum number of iterations
        :param tolerance: Convergence tolerance
        """
        # Use the preprocessed sequences from df_organized
        sequences = self.df_organized.tolist()
        self.converged = False
        
        print(f"Iteration 0: Transition Matrix: {self.dict_str(self.transition_matrix_log)}",
        f"State Probabilities: {self.dict_str(self.state_probabilities_log)}",
        f"Initial Probabilities: {self.dict_str(self.initial_probabilities_log)}")

        if save_data:
            if self.worker_id is not None and self.date is not None:
                message = f"Worker ID: {self.worker_id}, Date: {self.date}"
            else:
                message = "No Worker ID or Date provided."
            self.save_data_to_verbose_file(0, 0, message=message)
            message += f" Iteration 0, tol: {tolerance}"
            self.save_data_to_summary_file(message=message)
        

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
                self.converged = True
                break
        if save_data:
            self.save_data_to_summary_file(
                message=f"Iteration {iteration + 1}, tol: {tolerance}", 
                newline=True)
            with open(self.save_data_verbose_path, 'a') as f:
                f.write(f"\n")
        return self.get_params_exp()
        
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

    def generate_state_graph(self, output_dir, user_id, date, params, timebucket="15s"):
        """
        Generate a state graph for a specific user on a specific day.
        :param user_id: ID of the user. Make sure data is already preprocessed for 
        this user.
        :param output_dir: Directory to save the state graph
        :param day: Day for which to generate the state graph
        :param date: Date for which to generate the state graph
        :param params: Parameters to use for the Viterbi algorithm
        :return: None. Will Generate a plot the state graph.
        """
        # Filter the data for the specific user and day
        user_df = self.df[(self.df['user_id'] == user_id) & 
                          (self.df['date'] == date)].copy()
        
        seq = self.df_organized.iloc[0]
        state_sequence = self.viterbi_log(
            seq,
            params=params
        )
        # Merge the state sequence with the user_df
        user_df['pred_state'] = state_sequence

        # user_df['created_at'] = pd.to_datetime(user_df['created_at'])

        # # Convert time window to rolling average based on actual timestamps
        # user_df['disagree_rate'] = user_df.set_index('created_at')['disagree'].rolling(window=timebucket, min_periods=1).mean()

        rolling_values = user_df['disagree'].rolling(window=15, min_periods=1).mean()

        # Assign to DataFrame with explicit indexing
        user_df.loc[:, 'disagree_rate'] = rolling_values

        # Calculate average duration
        user_df['duration_ms'] = pd.to_numeric(user_df['duration_ms'])
        avg_duration = user_df['duration_ms'].mean() /1000 # Convert to seconds

        # Calculate time differences between timestamps
        user_df['pd_date'] = pd.to_datetime(user_df['created_at'])
        time_diffs = user_df['pd_date'].diff().dt.total_seconds()
        avg_time_diff = time_diffs.mean()

        # Expected time in good state: 
        exp_g_state = 1/(1-np.exp(self.transition_matrix_log["good"]))
        # Expected time in bad state:
        exp_b_state = 1/(1-np.exp(self.transition_matrix_log["bad"]))
        disag_rate_g = 1 - np.exp(self.state_probabilities_log["good"])
        disag_rate_b = 1 - np.exp(self.state_probabilities_log["bad"])


        # Create a plot of the disagreement rate, with the predicted states as a background
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(15, 6))


        
        # Prepare data
        x = user_df['created_at'].apply(lambda x: x.split()[1]).to_numpy()
        y = user_df['disagree_rate'].to_numpy()
        print(y)
        
        # Calculate y-axis limits with padding
        ymin, ymax = min(y), max(y)
        padding = (ymax - ymin) * 0.1
        ymin -= padding
        ymax += padding
        
        # Fill background states
        ax.fill_between(x, ymin, ymax,
                        where=(user_df['pred_state'].to_numpy() == 'good'),
                        color='#90EE90',  # light green
                        alpha=0.3,
                        label='Good State')
        ax.fill_between(x, ymin, ymax,
                        where=(user_df['pred_state'].to_numpy() == 'bad'),
                        color='#FFB6C1',  # light pink
                        alpha=0.3,
                        label='Bad State')
        
        # Plot the disagreement rate line
        ax.plot(x, y, 
                label='Disagreement Rate',
                color='#000080',  # navy blue
                linewidth=2,
                alpha=0.7)
        
        # Set x-axis ticks (show 10 evenly spaced labels)
        n_ticks = 10
        total_points = len(x)
        tick_indices = np.linspace(0, total_points-1, n_ticks, dtype=int)
        ax.set_xticks(x[tick_indices])
        
        # Format datetime labels
        # date_format = matplotlib.dates.DateFormatter('%H:%M:%S')
        # ax.xaxis.set_major_formatter(date_format)
        plt.xticks(rotation=30, ha='right')
        
        # Customize the plot
        ax.set_title(f'State Analysis for User {user_id}\n{date}\n_{timebucket}',
                    pad=20,
                    fontsize=12,
                    fontweight='bold')
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Disagreement Rate', fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.02, 1),
                loc='upper left',
                borderaxespad=0)
        
        # Add text annotations for average duration and time difference
        # and also some interpretation of the parameters.
        textstr = ( f'Avg Duration: {avg_duration:.2f} seconds\n',
                    f'Avg Time Diff: {avg_time_diff:.2f} seconds\n',
                    f'Expected Time in Good State: {exp_g_state:.2f} periods\n',
                    f'Expected Time in Bad State: {exp_b_state:.2f} periods\n',
                    f'Disagreement Rate in Good State: {disag_rate_g:.2f}\n',
                    f'Disagreement Rate in Bad State: {disag_rate_b:.2f}\n')
        # Create a text box in the upper left corner of the plot
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, ''.join(textstr),
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)


        
        # Add margins and adjust layout
        plt.margins(x=0.02)
        plt.tight_layout()
        
        # Save plot
        output_path = f"{output_dir}/state_graph_{user_id}_{date}_{timebucket}.png"
        plt.savefig(output_path, 
                    dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close()

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
    data_file_path = "/home/jgv555/CS/ResSum2025/drive-download-20250502T210721Z-1-001/ECPD/answers_revised2.csv"
    save_data_summary_path = "/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/DataSummary/ECPD_full_summary.csv"
    save_data_verbose_path = "/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/DataSummary/ECPD_full_verbose.csv"
    # data_file_path = "/home/jgv555/CS/ResSum2025/drive-download-20250502T210721Z-1-001/ECPD/test.csv"
    # data_file_path = "/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/fake_data.csv"
    # data_file_path = "/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/test2.csv"
    # model = MarkovModel(data_file_path, [0.7, 0.3, 0.7, 0.3, 0.6, 0.4])
    # model = MarkovModel(data_file_path, [0.9, 0.7, 0.6, 0.4, 0.5, 0.5])

    # Normal Training

    # model = MarkovModel(data_file_path=data_file_path, 
                        # save_data_verbose_path=save_data_verbose_path,
                        # save_data_summary_path=save_data_summary_path, 
                        # params=[0.99849304, 0.99607914, 0.97798713, 0.88902691, 0, 1]) # Iteration 68: Time: 242.63 s Trans Mat: good: 0.99582 bad: 0.94148 State Probs: good: 0.98495 bad: 0.58401 Init Probs: good: 0.77121 bad: 0.22879, 
    # model.preprocess_data(verbose=True, 
                            #  worker_id="955eb227-5421-470a-ae87-1b210a94bcfb",
                            #  date="2023-01-19")
    # model.preprocess_data(verbose=True)
    
    # model.baum_welch(save_data=True)

    # Testing the Viterbi algorithm
    # model = MarkovModel(data_file_path=data_file_path,
    #                     save_data_verbose_path=save_data_verbose_path,
    #                     save_data_summary_path=save_data_summary_path,
    #                     params = [0]*6)  # Initialize with zeros for testing
    # model.preprocess_data(verbose=True)
    # data_sequence = model.viterbi_log(model.df_organized.iloc[0],
    #                   params = [0.9995, 0.998, 0.95, 0.6, 0.6, 0.4],
    #                   )  # Use the same parameters as in the data generation
    
    # Compare data_sequence
    # Create a new column in the original DataFrame for predicted states
    # model.df['predicted_state'] = None  # Initialize the new column
    
    # # Get the user_id for the first sequence
    # first_user_id = model.df['user_id'].iloc[0]
    
    # # Add predicted states for this user
    # mask = model.df['user_id'] == first_user_id
    # model.df.loc[mask, 'predicted_state'] = data_sequence
    
    # # Save the updated DataFrame to CSV
    # model.df.to_csv('predictions.csv', index=False)
    
    # print("Predicted states:", data_sequence)
    # print("Actual observations:", model.df_organized.iloc[0])


    # Testing
    # model = MarkovModel(data_file_path=data_file_path,
    #                     save_data_verbose_path=save_data_verbose_path,
    #                     save_data_summary_path=save_data_summary_path,
    #                     params = [0.9995, 0.998, 0.95, 0.6, 0.6, 0.4])
    # model.preprocess_data(verbose=True, 
    #                       worker_id="USER00000",
    #                       date="2023-01-19")
    # model.generate_state_graph(
    #     output_dir="/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation",
    #     user_id="USER00000",
    #     date="2023-01-19",
    #     params=[0.9995, 0.998, 0.95, 0.6, 0.6, 0.4],
    #     timebucket=15
    # )



    # Testing viterbi_log on real worker data:
    # model = MarkovModel(data_file_path=data_file_path,
    #                     save_data_verbose_path=save_data_verbose_path,
    #                     save_data_summary_path=save_data_summary_path,
    #                     params = [0.9995, 0.998, 0.95, 0.6, 0.6, 0.4])
    # model.preprocess_data(verbose=True,
    #                         worker_id="955eb227-5421-470a-ae87-1b210a94bcfb",
    #                         date="2023-01-19")
    # model.generate_state_graph(
    #     output_dir="/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation",
    #     user_id="955eb227-5421-470a-ae87-1b210a94bcfb",
    #     date="2023-01-19",
    #     params=[0.99881183, 0.99676310, 0.97708480, 0.88847795, 0.00000000, 1.00000000],
    #     timebucket=15
    # )


    # Training Params on a few users.
    # init_params = [0.99881183, 0.99676310, 0.97708480, 0.88847795, 0.05, 0.95]
    # train_list = [
    #     ["955eb227-5421-470a-ae87-1b210a94bcfb", "2023-01-19"],
    #     ["229addc1-f8db-430d-887c-6b6719718978", "2023-01-23"],
    #     ["194db731-0e6e-450e-bfe4-aba328b25861", "2023-01-21"],
    #     ["45b63e66-62e4-4156-ae45-60832ab4bd73", "2023-01-21"],
    #     ["15d5a6ba-813e-42c0-b3c6-c260a0a358c6", "2023-01-19"],
    #     ["9d50aa2e-3d04-49cf-a000-0f0b92e8592e", "2023-01-21"],
    #     ["7e672082-824d-49b2-ae51-1a0e0ac09dd3", "2023-01-22"],
    #     ["4c709b89-7a88-4bed-8ee2-fe08c96ee83f", "2023-01-21"],
    #     ["1cdd9682-4160-44aa-b4fb-6cebdbb63af0", "2023-01-21"],
    #     ["0ee6d700-8332-4147-9ef1-cf23169fe958", "2023-01-19"],
    #     ["0b3438ad-cf01-46fb-8b65-eeb47d0fcad9", "2023-01-19"],
    #               ] 
    
    # for user_id, date in train_list:
    #     model = MarkovModel(data_file_path=data_file_path,
    #                         save_data_verbose_path=save_data_verbose_path,
    #                         save_data_summary_path=save_data_summary_path,
    #                         params = init_params)
    #     model.preprocess_data(verbose=True,
    #                           worker_id=user_id,
    #                           date=date)
    #     model.baum_welch(save_data=True)


    # train_list = [
    # ["955eb227-5421-470a-ae87-1b210a94bcfb", "2023-01-19", [0.99881243, 0.99676431, 0.97708267, 0.88847533, 0.00000000, 1.00000000]],
    # ["229addc1-f8db-430d-887c-6b6719718978", "2023-01-23", [0.99872311, 0.99862214, 0.98199381, 0.89386636, 0.00000000, 1.00000000]],
    # ["194db731-0e6e-450e-bfe4-aba328b25861", "2023-01-21", [0.97098767, 0.94486824, 0.97609808, 0.90344657, 1.00000000, 0.00000000]],
    # ["45b63e66-62e4-4156-ae45-60832ab4bd73", "2023-01-21", [0.98099109, 0.64208946, 0.95156533, 0.28076786, 1.00000000, 0.00000000]],
    # ["15d5a6ba-813e-42c0-b3c6-c260a0a358c6", "2023-01-19", [0.99984841, 0.99687240, 0.97275188, 0.87637805, 0.00000000, 1.00000000]],
    # ["9d50aa2e-3d04-49cf-a000-0f0b92e8592e", "2023-01-21", [0.99687528, 0.98357774, 0.96068858, 0.57871568, 0.00000000, 1.00000000]],
    # ["7e672082-824d-49b2-ae51-1a0e0ac09dd3", "2023-01-22", [0.99811104, 0.99925745, 0.98349474, 0.92041379, 1.00000000, 0.00000000]],
    # ["4c709b89-7a88-4bed-8ee2-fe08c96ee83f", "2023-01-21", [0.99644558, 0.95152804, 0.97267302, 0.63075011, 1.00000000, 0.00000000]],
    # ["1cdd9682-4160-44aa-b4fb-6cebdbb63af0", "2023-01-21", [0.99896960, 0.98848887, 0.96816190, 0.76310725, 1.00000000, 0.00000000]],
    # ["0ee6d700-8332-4147-9ef1-cf23169fe958", "2023-01-19", [0.99124663, 0.93245263, 0.96252469, 0.12904887, 0.00000000, 1.00000000]],
    # ["0b3438ad-cf01-46fb-8b65-eeb47d0fcad9", "2023-01-19", [0.99733764, 0.96880579, 0.97655913, 0.22445456, 1.00000000, 0.00000000]]
    # ]

    # global_params = [0.99582, 0.94148, 0.98495, 0.58401, 0.77121, 0.22879]
    # output_dir_ind = "/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/plots/individual_params"
    # output_dir_glob = "/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/plots/global_params"

    # for user_id, date, params in train_list:
    #     # print(f"Training for user {user_id} on date {date} with params {params}")
    #     model = MarkovModel(data_file_path=data_file_path,
    #                         save_data_verbose_path=save_data_verbose_path,
    #                         save_data_summary_path=save_data_summary_path,
    #                         params = global_params)
    #     model.preprocess_data(verbose=True,
    #                             worker_id=user_id,
    #                             date=date)
    #     model.generate_state_graph(
    #         output_dir=output_dir_ind,
    #         user_id=user_id,
    #         date=date,
    #         params=params,
    #         timebucket=15
    #     )
    #     model.generate_state_graph(
    #         output_dir=output_dir_glob,
    #         user_id=user_id,
    #         date=date,
    #         params=global_params,
    #         timebucket=15
    #     )

    # Testing the preprocess data function
    # model = MarkovModel(data_file_path=data_file_path,
    #                     save_data_verbose_path=save_data_verbose_path,
    #                     save_data_summary_path=save_data_summary_path,
    #                     params = [0.99881183, 0.99676310, 0.97708480, 0.88847795, 0.00000000, 1.00000000])
    # model.preprocess_data_continuous_activity(
    #     verbose=True,
    #     worker_id="955eb227-5421-470a-ae87-1b210a94bcfb",
    #     date="2023-01-19"
    # )
    # model.preprocess_data_continuous_activity(verbose=True,
    #                       min_sequence_length=4,
    #                       worker_id="01423e11-1fdc-41df-bd9c-a2f0d719a5cf",
    #                       date="2023-01-19")
    # print(model.df_organized)
    # model.df_organized.to_csv("prediction.csv")


    # Getting parameters for each user on each day.
    glob_params = [0.99881183, 0.99676310, 0.97708480, 0.88847795, 0.4, 0.6]
    model = MarkovModel(data_file_path=data_file_path,
                        save_data_verbose_path=save_data_verbose_path,
                        save_data_summary_path=save_data_summary_path,
                        params = glob_params)
    # Get unique combinations of user_id and date
    user_date_pairs = model.df[['user_id', 'date']].drop_duplicates()
    # Initialize DataFrame to store results
    results_df = pd.DataFrame(columns=['user_id', 'date', 'params', 'conv'])
    # Iterate over each user and date combination
    for _, row in user_date_pairs.iterrows():
        user_id = row['user_id']
        date = row['date']
        print(f"Processing user {user_id} on date {date}")
        
        # Preprocess data for this user and date
        model.preprocess_data_continuous_activity(
            verbose=True,
            min_sequence_length=60,
            max_time_gap=60,
            worker_id=user_id,
            date=date
        )

        model.initialize_params(glob_params)
        
        # Run Baum-Welch algorithm
        if len(model.df_organized) > 0:  # Only if we have sequences for this user/date
            params = model.baum_welch(save_data=True)
            print(f"Params for {user_id} on {date}: {params}")
            # Check convergence
            converged = model.converged
            # Append results to DataFrame
            results_df = results_df._append({
                'user_id': user_id,
                'date': date,
                'params': params,
                'conv': converged
            }, ignore_index=True)
    # Save results to CSV
    results_df.to_csv('/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/DataSummary/user_date_params.csv', index=False)
