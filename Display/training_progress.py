import matplotlib.pyplot as plt

import pickle as pkl

import numpy as np


training_losses = []
validation_losses = []

with open("../Training Progress/training_log_to_14.txt", "r") as file:

    print("Opened file")

    line = file.readline()

    while line != '':

        if(line.startswith("\n")):
            line = file.readline()
            continue

        if(line.startswith("{")):
            if(line.split()[0] == "{'eval_runtime':"):
                line = file.readline()
                continue

            training_losses.append(float(line.split()[1][:-1]))

        if(line.startswith("E")):
            validation_losses.append(float(line.split()[3]))

        line = file.readline()

# There are 14 epochs, and validation occurs 14 times, after training is reported
# during the whole epoch

epoch_length = len(training_losses) // 14

# Make the distances between each of the validation epochs the same as the epoch
# length in training

linspaced_validation_losses = np.array([])

# Pad the new array using linspace
for idx in range(len(validation_losses) - 1):
    linspace = np.linspace(start=validation_losses[idx], 
                           stop=validation_losses[idx+1], 
                           num=epoch_length)

    print(linspace)

    # # Add the real validation loss
    # linspaced_validation_losses = np.append(linspaced_validation_losses, 
    #                                         validation_losses[idx])

    # Add interpolated values
    linspaced_validation_losses = np.append(linspaced_validation_losses, 
                                            linspace)

# Add last real validation loss figure
# linspaced_validation_losses = np.append(linspaced_validation_losses, validation_losses[-1])

x1 = np.arange(len(linspaced_validation_losses)) + epoch_length
y1 = linspaced_validation_losses

x2 = np.arange(len(training_losses))
y2 = np.array(training_losses)

plt.plot(x1, y1, label = "Validation Loss")
plt.plot(x2, y2, label = "Training Loss", ls="--")

plt.xlabel("Step")

plt.ylabel("Loss")

plt.legend()

plt.grid('True')

plt.title("Training vs Validation Loss")

plt.show()