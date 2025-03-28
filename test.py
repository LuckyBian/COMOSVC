import numpy as np

# Replace 'file_path.npy' with the path to your .npy file
file_path = '/aifs4su/data/weizhen/data/emo/spk/795.npy'

# Load the numpy array from the .npy file
data = np.load(file_path)

# Print the size of the array
print("Size of the array:", data.size)

# Optionally, you can print the shape of the array to understand its dimensions
print("Shape of the array:", data.shape)
