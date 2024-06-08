import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import pandas as pd

# Load the data
df = pd.read_excel(r"probabilities0.xlsx")

# Extract true class probabilities and true class labels
true_class_probs = df.iloc[:, 1:12].values
true_class_labels = df.iloc[:, 12:].values

# Get class names from the column headers
class_names = df.columns[1:12].tolist()

# Initialize arrays to store the empirical accuracy and predicted probabilities
n_classes = true_class_probs.shape[1]
prob_true = np.zeros((n_classes, 10))
prob_pred = np.zeros((n_classes, 10))

# Calculate the empirical accuracy and predicted probabilities for each class
for i in range(n_classes):
    prob_true[i], prob_pred[i] = calibration_curve(true_class_labels[:, i], true_class_probs[:, i], n_bins=10)

# Define a list of distinctive colors for each class
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'olive']

# Plot the reliability diagram for each class
plt.figure(figsize=(15, 10))
for i in range(n_classes):
    plt.plot(prob_pred[i], prob_true[i], marker='o', label=f'{class_names[i]}', color=colors[i])
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('Predicted Probability')
plt.ylabel('Expected Accuracy')
plt.title('Calibration Plot For Protein Subcellular Localization')
plt.legend()
plt.grid(True)
plt.savefig('Calibration_plot.png')

plt.show()
