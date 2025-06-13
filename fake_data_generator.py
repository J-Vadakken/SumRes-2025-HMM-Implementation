import random
import pandas as pd

def two_state_hmm_data_gen(filename, n, n_range, params: list = [0.7, 0.7, 0.7, 0.3, 0.6, 0.4]):
    """
    Generate data based on a two state HMM.
    :n: Number of user sequences to generate.
    :n_range: Range of the number of labels per user sequence.
    :param params: List of parameters to initialize.
    """

    transition_matrix = {
        "good": params[0],  # P(G|G)
        "bad": params[1],   # P(B|B)
    }
    state_probabilities = {
        "good": params[2],  # P(C|G)
        "bad": params[3],   # P(C|B)
    }
    initial_probabilities = {
        "good": params[4],  # P(G)
        "bad": params[5],   # P(B)
    }
    """
    self.transition_matrix["good"] represents the probability 
    of transitioning to a good state from a good state. Denote this as 
    P(G|G). Note P(B|G) = 1 - P(G|G).
    self.transition_matrix["bad"] represents the probability 
    of transitioning to a bad state from a bad state. Denote this as 
    P(B|B). Note P(G|B) = 1 - P(B|B).
    self.state_probabilities["good"] represents the probability 
    of getting a label correct from a good state. Denote this as P(C|G). 
    Note P(W|B) = 1 - P(C|G). (Where C is correct and W is wrong)
    self.state_probabilities["bad"] represents the probability 
    of getting a label correct from a bad state. Denote this as P(C|B). 
    Note P(W|B) = 1 - P(C|B).
    self.initial_probabilities["good"] represents the 
    probability of starting in a good state. Denote this as P(G). 
    Note P(B) = 1 - P(G).
    self.initial_probabilities["bad"] represents the probability
    of starting in a bad state. Denote this as P(B). Note P(G) = 1 - P(B).
    """
    with open(filename, "w") as f:
        f.write("id,user_id,disagree,created_at,state\n")
    for i in range(n):
        seq_length = random.randint(n_range[0], n_range[1])
        state = "good" if random.random() < initial_probabilities["good"] else "bad"
        sequence = []
        state_list = [state]
        for _ in range(seq_length):
            if state == "good":
                label = 0 if random.random() < state_probabilities["good"] else 1
                sequence.append(label)
                state_list.append(state)
                if random.random() < transition_matrix["good"]:
                    state = "good"
                else:
                    state = "bad"
            else:
                label = 0 if random.random() < state_probabilities["bad"] else 1
                sequence.append(label)
                state_list.append(state)
                if random.random() < transition_matrix["bad"]:
                    state = "bad"
                else:
                    state = "good"

        with open(filename, "a") as f:
            for index, label in enumerate(sequence):
                f.write(f"label{i:05d}{index:05d},USER{i:05d},{label},{i:05d}{index:05d},{state_list[index]}\n")

    exp_state = {}
    g = transition_matrix["good"]
    b = transition_matrix["bad"]
    exp_state["good"] = (1-b) / (2-g-b)
    exp_state["bad"] = (1-g) / (2-b-g)
    
    exp_correct = exp_state["good"]* state_probabilities["good"] + \
                  exp_state["bad"] * state_probabilities["bad"]
    print(f"Expected state distribution: {exp_state}")
    print(f"Expected correct label distribution: {exp_correct}")

def data_gen_verification(filename):
    """
    Verify the generated data.
    :param filename: The name of the file to verify.
    """
    df = pd.read_csv(filename)
    # print(df.head())
    # print(f"Total sequences: {len(df['id'].unique())}")
    # print(f"Total labels: {len(df)}")
    print("State distribution:")
    print(df['state'].value_counts(normalize=True))
    print("Label distribution:")
    print(df['disagree'].value_counts(normalize=True))
        

if __name__ == "__main__":
    num_labels = 1*10**4
    two_state_hmm_data_gen("fake_data.csv", 1, (num_labels, num_labels), [0.9995, 0.998, 0.95, 0.6, 0.6, 0.4])
    print("Data generation complete.")
    data_gen_verification("fake_data.csv")
