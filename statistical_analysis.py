import pandas as pd

def calculate_mean(file_path: str, column_name: str) -> float:
    """
    Calculate the mean of a specified column in a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.
    column_name (str): The name of the column to calculate the mean for.

    Returns:
    float: The mean of the specified column.
    """
    data = pd.read_csv(file_path)
    mean_value = data[column_name].mean()
    return mean_value

print(calculate_mean('/home/bytefuse/batsi/SB-MCL/experiments/STD-offline/results.csv', 'best_test_acc'))