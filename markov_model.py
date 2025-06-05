import pandas as pd
import numpy as np

class MarkovModel:
    def __init__(self, data_file_path, params: list = [0.7, 0.7, 0.7, 0.3, 0.6, 0.4]):
        """
        Initialize the Markov Model with a data file path and parameters.
        :param data_file_path: Path to the CSV file containing the data.
        :param params: List of parameters to initialize the model.
        """
        self.data_file_path = data_file_path
        self.transition_matrix = {}
        self.state_probabilities = {}
        self.initial_probabilities = {}
        

        self.df = pd.read_csv(self.data_file_path)
        self.initialize_params(params)

    def data_preprocessing(self):
        """
        Preprocess the data to extract sequences of observations.
        This function should be implemented based on the specific structure of the CSV file.
        """
        # Sort by timestamp for each user
        self.df = self.df.sort_values(['user_id', 'created_at'])

        # Create sequences of responses (1 for correct, 0 for incorrect)
        self.df_organized = self.df.groupby('user_id')['disagree'].agg(lambda x: list(x*-1 + 1))

        # Convert the sequences to 'correct'/'wrong' format
        self.df_organized = self.df_organized.apply(lambda x: ['correct' if i else 'wrong' for i in x])


    def initialize_params(self, params: list = [0.7, 0.7, 0.7, 0.3, 0.6, 0.4]):
        """
        Initialize the model parameters.
        :param params: List of parameters to initialize.
        """
        self.transition_matrix["good"] = params[0]
        self.transition_matrix["bad"] = params[1]
        self.state_probabilities["good"] = params[2]
        self.state_probabilities["bad"] = params[3]
        self.initial_probabilities["good"] = params[4]
        self.initial_probabilities["bad"] = params[5]

        """
        self.transition_matrix["good"] represents the probability of transitioning 
        to a good state from a good state. Denote this as P(G|G). Note 
        P(B|G) = 1 - P(G|G).
        self.transition_matrix["bad"] represents the probability of transitioning
        to a bad state from a bad state. Denote this as P(B|B). Note 
        P(G|B) = 1 - P(B|B).
        self.state_probabilities["good"] represents the probability of getting a label 
        correct from a good state. Denote this as P(C|G). Note P(W|B) = 1 - P(C|G). 
        (Where C is correct and W is wrong)
        self.state_probabilities["bad"] represents the probability of getting a label
        correct from a bad state. Denote this as P(C|B). Note P(W|B) = 1 - P(C|B).
        self.initial_probabilities["good"] represents the probability of starting in a
        good state. Denote this as P(G). Note P(B) = 1 - P(G).
        self.initial_probabilities["bad"] represents the probability of starting in a
        bad state. Denote this as P(B). Note P(G) = 1 - P(B).
        """

    def forward_backward(self, sequence):
        """
        Implements the forward-backward algorithm for a sequence of observations.
        :param sequence: List of observations ('correct' or 'wrong')
        :return: Forward probabilities, backward probabilities
        """
        N = len(sequence)
        forward = np.zeros((N, 2))  # 2 states: good (0) and bad (1)
        backward = np.zeros((N, 2))

        # Forward pass
        # Initialize first step
        forward[0, 0] = self.initial_probabilities["good"] * (self.state_probabilities["good"] if sequence[0] == 'correct' else 1 - self.state_probabilities["good"])
        forward[0, 1] = self.initial_probabilities["bad"] * (self.state_probabilities["bad"] if sequence[0] == 'correct' else 1 - self.state_probabilities["bad"])

        # Recursion step
        for t in range(1, N):
            # Good state
            forward[t, 0] = (forward[t-1, 0] * self.transition_matrix["good"] + 
                            forward[t-1, 1] * (1 - self.transition_matrix["bad"])) * \
                            (self.state_probabilities["good"] if sequence[t] == 'correct' else 1 - self.state_probabilities["good"])
            # Bad state
            forward[t, 1] = (forward[t-1, 0] * (1 - self.transition_matrix["good"]) + 
                            forward[t-1, 1] * self.transition_matrix["bad"]) * \
                            (self.state_probabilities["bad"] if sequence[t] == 'correct' else 1 - self.state_probabilities["bad"])

        # Backward pass
        # Initialize last step
        backward[N-1] = 1.0

        # Recursion step
        for t in range(N-2, -1, -1):
            # Good state
            backward[t, 0] = backward[t+1, 0] * self.transition_matrix["good"] * \
                            (self.state_probabilities["good"] if sequence[t+1] == 'correct' else 1 - self.state_probabilities["good"]) + \
                            backward[t+1, 1] * (1 - self.transition_matrix["good"]) * \
                            (self.state_probabilities["bad"] if sequence[t+1] == 'correct' else 1 - self.state_probabilities["bad"])
            # Bad state
            backward[t, 1] = backward[t+1, 0] * (1 - self.transition_matrix["bad"]) * \
                            (self.state_probabilities["good"] if sequence[t+1] == 'correct' else 1 - self.state_probabilities["good"]) + \
                            backward[t+1, 1] * self.transition_matrix["bad"] * \
                            (self.state_probabilities["bad"] if sequence[t+1] == 'correct' else 1 - self.state_probabilities["bad"])

        return forward, backward

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
            new_transition = {"good": 0, "bad": 0}
            new_emission = {"good": 0, "bad": 0}
            new_initial = {"good": 0, "bad": 0}
            denominators = {"good": 0, "bad": 0}
            denominators_transition = {"good": 0, "bad": 0}
            
            # Process each sequence
            for sequence in sequences:
                forward, backward = self.forward_backward(sequence)
                

                # Check gamma probabilities
                gamma = forward * backward / np.sum(forward * backward, axis=1, keepdims=True)
                print(type(gamma))
                print(type(forward))
                print(gamma.dtype, forward.dtype, backward.dtype)
                
                # Update initial probabilities
                new_initial["good"] += gamma[0, 0]
                new_initial["bad"] += gamma[0, 1]
                
                # Update transition and emission probabilities
                for t in range(len(sequence)):
                    if t < len(sequence) - 1:
                        # Update transition probabilities
                        next_emission_good = self.state_probabilities["good"] if sequence[t+1] == 'correct' else 1 - self.state_probabilities["good"]
                        next_emission_bad = self.state_probabilities["bad"] if sequence[t+1] == 'correct' else 1 - self.state_probabilities["bad"]
                        # new_transition["good"] += (gamma[t, 0] * self.transition_matrix["good"] * next_emission_good)
                        # new_transition["bad"] += gamma[t, 1] * self.transition_matrix["bad"] * next_emission_bad 
                        # new_transition["good"] += (gamma[t, 0] * self.transition_matrix["good"] * next_emission_good) / (gamma[t,0] + gamma[t,1])
                        # new_transition["bad"] += gamma[t, 1] * self.transition_matrix["bad"] * next_emission_bad / (gamma[t,0] + gamma[t,1])
                        new_transition["good"] += (forward[t, 0] * backward[t+1, 0] * self.transition_matrix["good"] * next_emission_good) / (gamma[t, 0] + gamma[t, 1])
                        new_transition["bad"] += (forward[t, 1] * backward[t+1, 1] * self.transition_matrix["bad"] * next_emission_bad) / (gamma[t, 0] + gamma[t, 1])
                        denominators_transition["good"] += gamma[t, 0]
                        denominators_transition["bad"] += gamma[t, 1]
                    # Update emission probabilities
                    if sequence[t] == 'correct':
                        new_emission["good"] += gamma[t, 0]
                        new_emission["bad"] += gamma[t, 1]
                    

                    denominators["good"] += gamma[t, 0]
                    denominators["bad"] += gamma[t, 1]
            
            # Update parameters
            self.transition_matrix["good"] = (new_transition["good"] / denominators_transition["good"]) if denominators_transition["good"] > 0 else self.transition_matrix["good"]
            self.transition_matrix["bad"] = (new_transition["bad"] / denominators_transition["bad"]) if denominators_transition["bad"] > 0 else self.transition_matrix["bad"]
            self.state_probabilities["good"] = (new_emission["good"] / denominators["good"]) if denominators["good"] > 0 else self.state_probabilities["good"]
            self.state_probabilities["bad"] = (new_emission["bad"] / denominators["bad"]) if denominators["bad"] > 0 else self.state_probabilities["bad"]
            self.initial_probabilities["good"] = new_initial["good"] / len(sequences)
            self.initial_probabilities["bad"] = new_initial["bad"] / len(sequences)
            
            print(f"Iteration {iteration + 1}: Transition Matrix: {self.dict_str(self.transition_matrix)}, State Probabilities: {self.dict_str(self.state_probabilities)}, Initial Probabilities: {self.dict_str(self.initial_probabilities)}")
        
            # Check convergence
            if np.allclose(old_params, self.get_params(), rtol=tolerance):
                break

    def get_params(self):
        """Helper function to get all parameters as a numpy array"""
        return np.array([
            self.transition_matrix["good"],
            self.transition_matrix["bad"],
            self.state_probabilities["good"],
            self.state_probabilities["bad"],
            self.initial_probabilities["good"],
            self.initial_probabilities["bad"]
        ])

    def dict_str(self, d):
        """Helper function to print dictionary in a readable format"""
        return_string = ""
        for key, value in d.items():
            return_string += f"{key}: {value:.3f} "
        return return_string[:-1]  # Remove trailing space

if __name__ == "__main__":
    data_file_path = "/home/jgv555/CS/ResSum2025/drive-download-20250502T210721Z-1-001/ECPD/answers_revised2.csv"
    # data_file_path = "/home/jgv555/CS/ResSum2025/drive-download-20250502T210721Z-1-001/ECPD/test.csv"
    model = MarkovModel(data_file_path)
    model.data_preprocessing()
    print(model.df_organized.head(3))  # Display the first few sequences
    model.baum_welch()
