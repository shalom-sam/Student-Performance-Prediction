

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("C:/Users/Shalo/OneDrive/Desktop/python/student_data.csv")

print("Dataset Preview:")
print(data.head())
print("\nColumns:", data.columns)


data['Previous_Result'] = data['Previous_Result'].map({'Pass': 1, 'Fail': 0})
data['Final_Result'] = data['Final_Result'].map({'Pass': 1, 'Fail': 0})


sns.countplot(x='Final_Result', data=data)
plt.title("Final Result Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='Final_Result', y='Attendance', data=data)
plt.title("Attendance vs Final Result")
plt.xlabel("Final Result (0 = Fail, 1 = Pass)")
plt.ylabel("Attendance (%)")
plt.show()


plt.figure(figsize=(6,4))
sns.boxplot(x='Final_Result', y='Study_Hours', data=data)
plt.title("Study Hours vs Final Result")
plt.xlabel("Final Result (0 = Fail, 1 = Pass)")
plt.ylabel("Study Hours per Day")
plt.show()


plt.figure(figsize=(6,4))
sns.histplot(data['Internal_Marks'], kde=True)
plt.title("Internal Marks Distribution")
plt.xlabel("Internal Marks")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(6,4))
sns.boxplot(x='Final_Result', y='Assignment_Score', data=data)
plt.title("Assignment Score vs Final Result")
plt.xlabel("Final Result (0 = Fail, 1 = Pass)")
plt.ylabel("Assignment Score")
plt.show()


plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()



X = data.drop('Final_Result', axis=1)
y = data['Final_Result']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


new_student = pd.DataFrame(
    [[80, 3, 65, 70, 1]],
    columns=X.columns
)

prediction = model.predict(new_student)

print("\nPrediction Result:")
if prediction[0] == 1:
    print("Student will PASS")
else:
    print("Student will FAIL")

