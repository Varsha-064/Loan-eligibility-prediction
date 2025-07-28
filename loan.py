# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Load the dataset

data_path = "Traindata.csv"
data = pd.read_csv(data_path)

# Preprocess the data

data['LoanAmount_log'] = np.log(data['LoanAmount'] + 1)
data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
data['TotalIncome_log'] = np.log(data['TotalIncome'] + 1)

# Fill missing values

data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
data['Married'].fillna(data['Married'].mode()[0], inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)
data['LoanAmount_log'].fillna(data['LoanAmount_log'].mean(), inplace=True)

# Select features and target variable

X = data.iloc[:, np.r_[1:5, 9:11, 13:15]].values
y = data.iloc[:, 12].values

# Encode categorical variables before splitting

label_encoder = LabelEncoder()
for i in range(X.shape[1]):
    if data.dtypes[i] == 'object':
        X[:, i] = label_encoder.fit_transform(X[:, i])

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale features

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encode target variable

y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Train a classifier (Decision Tree in this case)

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)


# Evaluate the model

accuracy = metrics.accuracy_score(y_test, predictions)
print(f"Decision Tree Classifier Accuracy: {accuracy}")

# Visualize the results

unique, counts = np.unique(predictions, return_counts=True)
results = dict(zip(unique, counts))

# Bar chart for loan approval

labels = ['Not Approved', 'Approved']
values = [results.get(0, 0), results.get(1, 0)]

plt.bar(labels, values, color=['red', 'green'])
plt.xlabel('Loan Status')
plt.ylabel('Number of Applications')
plt.title('Loan Approval Status')
plt.show()

# Bar chart for loan approval by Self_Employed status

self_employed_counts = data[data['Loan_Status'] == 'Y']['Self_Employed'].value_counts()
plt.bar(self_employed_counts.index, self_employed_counts.values, color='blue')
plt.xlabel(' Employed Status')
plt.ylabel('UnEmolyed Status')
plt.title('Approved Loans by Self Employed Status')
plt.show()

# Extract details of approved loans

approved_loans = data[data['Loan_Status'] == 'Y'][['Loan_ID', 'LoanAmount']]

# Display approved loan details

print("\nDetails of Approved Loans:")
print(approved_loans)