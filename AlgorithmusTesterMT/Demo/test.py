import string
import random

def randomString(length):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

# Define the number of random strings you want to generate
num_strings = 10000

# Define the length of each random string
string_length = 10

# Generate random strings and save them to a list
random_strings = [randomString(string_length) for _ in range(num_strings)]

# Define the file path where you want to save the random strings
file_path = "random_strings.txt"

# Write the random strings to the file
with open(file_path, 'w') as file:
    for rand_str in random_strings:
        file.write(rand_str + '\n')
