"""
In the following code I am experimenting with statistical and data visualization techniques using
the classic small dataset from Fisher, 1936, on Iris flower measuremnets. Although I am not a flower expert, 
I hope to be able to work with the data in the files and understand it better through the following computational
techniques using pandas, seaborn, and matplotlib.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
filepath = "/Users/hamzahussain/Desktop/compsci/projects/data_projects/IRIS/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_df = pd.read_csv(filepath, header=None, names=columns)

# Statistics
statistics = iris_df.describe()
print(statistics)


# Correlations
correlation_matrix = iris_df.drop('class', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title('Correlation Matrix')
plt.show()

# Visualize sepal length distribution by species
plt.figure(figsize=(10, 6))
sns.boxplot(x='class', y='sepal_length', data=iris_df)
plt.title('Sepal Length Distribution by Species')
plt.show()