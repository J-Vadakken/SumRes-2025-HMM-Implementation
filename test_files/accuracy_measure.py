import pandas as pd
import numpy as np

# A file for testing the rolling average calculation

class TestObject: 
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        return None
    
    def rolling_average(self, window_size="15S"):
        """
        Calculate the rolling average of the data in the file.
        :param window_size: The size of the rolling window.
        :return: A pandas Series with the rolling average.
        """
        
        df = pd.read_csv(self.data_file_path)
        for user_id in df["user_id"].unique():
            user_df = df[(df['user_id'] == user_id)].copy()
            if "disagree" not in user_df.columns:
                raise ValueError("The 'disagree' column is missing from the data file.")
            
            # Ensure 'created_at' is in datetime format
            user_df['created_at'] = pd.to_datetime(user_df['created_at'], errors='coerce')
            if user_df['created_at'].isnull().any():
                raise ValueError("Some 'created_at' values could not be converted to datetime.")
            
            # Set the index to 'created_at'
            user_df_sorted = user_df.sort_values('created_at')
            user_df_sorted.set_index('created_at', inplace=True)
            
            # Calculate the rolling average
            rolling_avg = user_df_sorted['disagree'].rolling(window=window_size, center=True).mean()

            # Create a new column to hold this data
            user_df['disagree_rate'] = user_df['created_at'].map(rolling_avg)

            # Map rolling averages back to the main DataFrame using created_at timestamps
            user_mask = df['user_id'] == user_id
            df.loc[user_mask, 'disagree_rate'] = df.loc[user_mask, 'created_at'].map(rolling_avg)
            return df
        
        return df

if __name__ == "__main__":
    # data_file_path = "/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/test2.csv"
    data_file_path = "/home/jgv555/CS/ResSum2025/drive-download-20250502T210721Z-1-001/ECPD/answers_revised2.csv"
    print("Testing rolling average calculation...")
    test_obj = TestObject(data_file_path)
    rolling_avg = test_obj.rolling_average(window_size="20min1S")
    print(rolling_avg)