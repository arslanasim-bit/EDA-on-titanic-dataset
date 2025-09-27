# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset (use train.csv for EDA)
df = pd.read_csv("train.csv")

# 3. Quick Look at Data
print(df.head())   # first 5 rows
print(df.isnull().sum())   # check missing values

# 4. Handle Missing Values
df['Age'] = df['Age'].fillna(df['Age'].median())   # fill missing ages with median
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])   # fill embarked with mode
df = df.drop(columns=['Cabin'])   # drop cabin (too many missing)

# 5. Convert Data Types
df['Pclass'] = df['Pclass'].astype('category')  # passenger class can be category

# ✅ Create a new readable column for reporting (Yes/No)
df['Survived_label'] = df['Survived'].map({0: 'No', 1: 'Yes'})

# 6. Summary Statistics
print(df.describe(include='all'))   # summary stats
print("\nOverall Survival Rate (%):")
print(df['Survived_label'].value_counts(normalize=True) * 100)

# 7. Group-based Insights
print("\nSurvival by Gender (numeric):")
print(df.groupby('Sex')['Survived'].mean())

print("\nSurvival by Gender (Yes/No %):")
print(df.groupby('Sex')['Survived_label'].value_counts(normalize=True) * 100)

print("\nSurvival by Class (numeric):")
print(df.groupby('Pclass')['Survived'].mean())

print("\nSurvival by Gender & Class (numeric):")
print(df.groupby(['Sex','Pclass'])['Survived'].mean())

# 8. Visualizations
sns.countplot(x='Sex', hue='Survived_label', data=df)
plt.title("Survival by Gender")
plt.show()

sns.countplot(x='Pclass', hue='Survived_label', data=df)
plt.title("Survival by Class")
plt.show()

sns.histplot(data=df, x='Age', hue='Survived_label', bins=20, kde=True)
plt.title("Age Distribution by Survival")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
