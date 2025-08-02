# ==============================================================================

# Step 1: Import Libraries and Load Data

# ==============================================================================

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Load the dataset from a reliable URL

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

df = pd.read_csv(url)



print("--- Initial Data ---")

print("First 5 rows of the dataset:")

print(df.head())

print("\n")





# ==============================================================================

# Step 2: Data Cleaning

# ==============================================================================

print("--- Data Cleaning Process ---")

# Get a summary of the dataset before cleaning

print("Dataset Info Before Cleaning:")

df.info()



# Check for missing values

print("\nMissing Values Before Cleaning:")

print(df.isnull().sum())



# --- Handle Missing Values ---

# 1. Fill missing 'Age' values with the median age.

df['Age'].fillna(df['Age'].median(), inplace=True)



# 2. Fill missing 'Embarked' values with the most frequent value (mode).

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)



# 3. Drop the 'Cabin' column due to too many missing values.

df.drop('Cabin', axis=1)



# --- Verify Cleaning ---

print("\nMissing Values After Cleaning:")

print(df.isnull().sum())

print("\nData cleaning complete.\n")





# ==============================================================================

# Step 3: Exploratory Data Analysis (EDA)

# ==============================================================================

print("--- Starting Exploratory Data Analysis ---")



# Set the visual style for plots

sns.set_style("whitegrid")



# --- Analysis 1: Overall Survival Rate ---

plt.figure(figsize=(6, 4))

sns.countplot(x='Survived', data=df)

plt.title('Overall Survival Count (0 = No, 1 = Yes)')

plt.show()



survival_rate = df['Survived'].mean() * 100

print(f"Overall Survival Rate: {survival_rate:.2f}%")





# --- Analysis 2: Survival by Gender ---

plt.figure(figsize=(6, 4))

sns.countplot(x='Sex', hue='Survived', data=df)

plt.title('Survival Count by Gender')

plt.show()



print("\nSurvival Rate by Gender:")

print(df.groupby('Sex')['Survived'].mean() * 100)





# --- Analysis 3: Survival by Passenger Class (Pclass) ---

plt.figure(figsize=(8, 5))

sns.countplot(x='Pclass', hue='Survived', data=df)

plt.title('Survival Count by Passenger Class')

plt.show()



print("\nSurvival Rate by Passenger Class:")

print(df.groupby('Pclass')['Survived'].mean() * 100)





# --- Analysis 4: Survival by Age Distribution ---

plt.figure(figsize=(10, 6))

sns.kdeplot(df[df['Survived'] == 1]['Age'], shade=True, label='Survived')

sns.kdeplot(df[df['Survived'] == 0]['Age'], shade=True, label='Did not Survive')

plt.title('Age Distribution of Passengers by Survival Status')

plt.xlabel('Age')
plt.legend()
plt.show()
# --- Analysis 5: Correlation Heatmap for Numerical Features ---
# Select only numeric columns for the correlation matrix
numeric_df = df.select_dtypes(include=[np.number])
# Create a correlation matrix
correlation_matrix = numeric_df.corr()
# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()
print("\n--- EDA Complete ---")