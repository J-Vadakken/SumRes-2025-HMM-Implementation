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
    # data_file_path = "/home/jgv555/CS/ResSum2025/drive-download-20250502T210721Z-1-001/ECPD/answers_revised2.csv"
    # print("Testing rolling average calculation...")
    # test_obj = TestObject(data_file_path)
    # rolling_avg = test_obj.rolling_average(window_size="20min1S")
    # print(rolling_avg)

    def compare_two_param_files():
        data_file_path1 = "/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/DataSummary/user_date_params_600.csv"
        data_file_path2 = "/home/jgv555/CS/ResSum2025/model/SumRes-2025-HMM-Implementation/DataSummary/user_date_params_60.csv"

        df1 = pd.read_csv(data_file_path1)
        df2 = pd.read_csv(data_file_path2)

        # Compare the two DataFrames
        df1["params"] = df1["params"].to_numpy()
        df2["params"] = df2["params"].to_numpy()

        diff_count = 0
        count_above_10 = 0
        tot = 0
        for _, row in df1.iterrows():
            user_id = df1.iloc[_]["user_id"]
            date = df1.iloc[_]["date"]
            matching_row = df2[(df2["user_id"] == user_id) & (df2["date"] == date)]
            if matching_row.empty:
                print(f"No matching row found for user {user_id} on {date}")
            else:
                df1params = np.fromstring(row["params"].strip("'\"[]"), sep=" ")
                df2params = np.fromstring(matching_row["params"].iloc[0].strip("'\"[]"), sep=" ")
                # print(matching_row["params"].iloc[0])
                
                diff = df1params - df2params
                if diff.sum() == 0:
                    tot += 1
                else:
                    diff_count += 1
                    # print(f"Difference for user {user_id} on {date}: {diff}")
                    
                    num1 = 4
                    num2 = 6
                    print(np.round(diff[num1:num2]*100, 1))
                    if diff[num1]*100 >= 99 or diff[num2-1]*100 >= 99:
                        count_above_10 += 1

                    # print(np.max([diff[num1:num2]/ df1params[num1:num2], diff[num1:num2] / df2params[num1:num2]],axis=1 ))

                
                    
                    # tg1 = 1/(1-df1params[0])
                    # tb1 = 1/(1-df1params[1])
                    # tg2 = 1/(1-df2params[0])
                    # tb2 = 1/(1-df2params[1])

                    # print(f"tg1: {tg1} vs tg2: {tg2} \n tb1: {tb1} vs tb2: {tb2} \n ____")
                    # arr = np.max([[tg2 / tg1, tg1 / tg2], [tb2 / tb1, tb1 / tb2]], axis=0)
                    # print(arr)
                    # if arr[0] > 10000 or arr[1] > 10000:
                    #     count_above_10 += 1


                    

                    # print(np.round(diff[num1:num2] / df1params[num1:num2] * 100, 1))
                    # print("*")
                    # print(df1params[num1:num2])
                    # print("G")
                    # print(1/(1-df1params[0]) - 1/(1-df2params[0]))

                    print("____________________________")
            
        print(f"Total matches: {tot}, Total differences: {diff_count}")
        print("Count of differences above 10%: ", count_above_10)
    compare_two_param_files()