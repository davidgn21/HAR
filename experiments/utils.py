import pandas as pd
import numpy as np
import scipy.signal as signal
import os
from sklearn.preprocessing import MinMaxScaler



def make_dataframe(path):
    """
    This function recives a path of a csv file, transforming it into a dataframe, 
    then the dataframe is separated into gravity and movement components and 
    calculations of stadistics such as mean, std are performed, considering a 
    window size of 50 samples and concatenates all of the dataframes into a single 
    df with all the stadistics calculated

    Parameters:
        path (str): path to the csv file

    Returns:
        pandas.DataFrame: The function returns a single dataframe 
        with all the stadistics calculated of the window size
    """
    
    df = pd.read_csv(path)
    df['vector_mag_back'] = np.linalg.norm(df[['back_x', 'back_y', 'back_z']], axis=1)
    df['vector_mag_thigh'] = np.linalg.norm(df[['thigh_x', 'thigh_y', 'thigh_z']], axis=1)

    ### GRAVITY DF
    # Create a copy of the original dataframe
    gravity_df = df.copy()
    # Define the filter parameters
    order = 4  # Filter order
    cutoff = 1.0  # Cutoff frequency in Hz
    fs = 50  # Your actual sampling frequency
    # Compute the filter coefficients
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    # Apply the filter to the desired columns
    columns_to_filter = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z', 'vector_mag_back', 'vector_mag_thigh']
    for column in columns_to_filter:
        gravity_df[column] = signal.lfilter(b, a, df[column])

    ## Orientation information
    window_size = 50
    columns_to_calculate = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z', 'vector_mag_back', 'vector_mag_thigh']
    # Calculate the stats for each window
    label_mode = gravity_df.groupby(gravity_df.index // window_size)['label'].transform(lambda x: x.mode().iat[0])
    gravity_mean = gravity_df[columns_to_calculate].rolling(window=window_size).mean().add_suffix('_mean')
    gravity_median = gravity_df[columns_to_calculate].rolling(window=window_size).median().add_suffix('_median')
    gravity_std = gravity_df[columns_to_calculate].rolling(window=window_size).std().add_suffix('_std')
    gravity_25_p = gravity_df[columns_to_calculate].rolling(window=window_size).quantile(0.25).add_suffix('_25_p')
    gravity_75_p = gravity_df[columns_to_calculate].rolling(window=window_size).quantile(0.75).add_suffix('_75_p')
    gravity_min = gravity_df[columns_to_calculate].rolling(window=window_size).min().add_suffix('_min')
    gravity_max = gravity_df[columns_to_calculate].rolling(window=window_size).max().add_suffix('_max')
    res_gravity_df = pd.concat([label_mode, gravity_mean, gravity_median, gravity_std, gravity_25_p, gravity_75_p, gravity_min, gravity_max], axis=1)
    res_gravity_df = res_gravity_df[window_size-1::window_size].reset_index(drop=True)
    # Calculate the coefficient of variation
    for column in columns_to_calculate:
        mean_col = column + '_mean'
        std_col = column + '_std'
        cv_col = column + '_cv'
        res_gravity_df[cv_col] = (res_gravity_df[std_col] / res_gravity_df[mean_col]) * 100


    ### MOVEMENT DF
    movement_df = df.copy()
    movement_df.iloc[:, 1:7] -= gravity_df.iloc[:, 1:7]
    # Frequency-domain features
    # Apply Fourier Transform to each axis separately
    for column in columns_to_calculate:
        # Get the time-domain signal values from the DataFrame
        time_domain_signal = movement_df[column].values
        # Apply Fast Fourier Transform (FFT)
        frequency_domain_signal = np.fft.fft(time_domain_signal)
        # Optionally, you can calculate the magnitude spectrum using absolute values
        magnitude_spectrum = np.abs(frequency_domain_signal)
        # Store the frequency-domain representation back into the DataFrame
        movement_df[column + '_freq'] = magnitude_spectrum
    # The DataFrame 'movement_df' now contains the frequency-domain representation of each axis in separate columns

    ## Movement information
    window_size = 50
    columns_to_calculate = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z', 'vector_mag_back', 'vector_mag_thigh']
    # Calculate the stats for each window
    movement_skew = movement_df[columns_to_calculate].rolling(window=window_size).skew().add_suffix('_skew')
    res_movement_df = pd.concat([movement_skew], axis=1)
    res_movement_df = res_movement_df[window_size-1::window_size].reset_index(drop=True)
    signal_energies = {}
    for column in columns_to_calculate:
        column_values = movement_df[column].values
        windows = [column_values[i:i+window_size] for i in range(0, len(column_values), window_size)]
        energy_values = [np.sum(np.square(window)) for window in windows]
        signal_energies[column + '_se'] = energy_values
    energy_df = pd.DataFrame(signal_energies)
    res_movement_df = pd.concat([res_movement_df, energy_df], axis=1)

    # Frequency-domain features 
    columns_to_calculate = ['back_x_freq', 'back_y_freq', 'back_z_freq', 'thigh_x_freq', 'thigh_y_freq', 'thigh_z_freq', 'vector_mag_back_freq', 'vector_mag_thigh_freq']
    movement_freq_mean = movement_df[columns_to_calculate].rolling(window=window_size).mean().add_suffix('_mean')
    movement_freq_std = movement_df[columns_to_calculate].rolling(window=window_size).std().add_suffix('_std')
    res_freq_movement_df = pd.concat([movement_freq_mean, movement_freq_std], axis=1)
    res_freq_movement_df = res_freq_movement_df[window_size-1::window_size].reset_index(drop=True)
    
    res_df = pd.concat([res_gravity_df, res_movement_df, res_freq_movement_df], axis=1).dropna()
    res_df['label'] = res_df['label'].astype(int)


    ### Min-Max scaling of the resulting dataframe
    # Create an instance of the MinMaxScaler
    scaler = MinMaxScaler()
    # Select the columns
    column_to_exclude = 'label'
    numerical_columns = res_df.select_dtypes(include='number').columns.drop(column_to_exclude).tolist()

    # Apply min-max scaling to the numerical columns
    res_df[numerical_columns] = scaler.fit_transform(res_df[numerical_columns])
    
    return res_df





def read_files(path_name):
    file_list = os.listdir(path_name)
    df_list = []
    final_df = None
    for file_name in file_list:
        file_path = os.path.join(path_name, file_name)
        df = make_dataframe(file_path)
        df_list.append(df)   
    final_df = pd.concat(df_list, axis=0)
    return final_df




# df_final = read_files('/home/david/Documents/HAR/data/harth')
# print(df_final)




