import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
import streamlit as st
import warnings

warnings.filterwarnings('ignore')

@st.cache_data
def load_data():
    df = pd.read_csv('Heart_Disease_Prediction.csv')
    return df

df = load_data()

st.title("Heart Disease Prediction App")
st.write("### Dataset Overview")
st.write("Missing values:\n", df.isnull().sum())
st.write("\nDataset Info:")
st.write(df.info())
st.write("\nBasic Statistics:")
st.write(df.describe())

df['Sex'] = df['Sex'].map({0: 'Female', 1: 'Male'})
df['Heart Disease'] = df['Heart Disease'].map({'Absence': 0, 'Presence': 1})

st.write("### Visualizations")
fig, ax = plt.subplots(figsize=(15, 10))
sns.histplot(data=df, x='Age', hue='Heart Disease', multiple="stack", ax=ax)
plt.title('Age Distribution by Heart Disease')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(data=df, x='Sex', hue='Heart Disease', ax=ax)
plt.title('Gender Distribution by Heart Disease')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(12, 8))
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
plt.title('Correlation Matrix')
st.pyplot(fig)

prevalence = df['Heart Disease'].value_counts(normalize=True) * 100
st.write("### Prevalence of Heart Disease:")
st.bar_chart(prevalence)

X = df.drop(['Heart Disease'], axis=1)
y = df['Heart Disease']
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

st.write("### Model Classification Report:")
st.text(classification_report(y_test, y_pred_svm))

fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(fig)

st.write("Model Accuracy:", svm_model.score(X_test_scaled, y_test))

dt_model = DecisionTreeClassifier(random_state=42, max_depth=5) 
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)

st.write("### Decision Tree Model Classification Report:")
st.text(classification_report(y_test, dt_pred))

fig, ax = plt.subplots(figsize=(8, 6))
dt_cm = confusion_matrix(y_test, dt_pred)
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt_model, feature_names=X_train.columns, class_names=['Absence', 'Presence'], 
          filled=True, rounded=True, ax=ax)
plt.title('Decision Tree Visualization')
st.pyplot(fig)

rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

st.write("### Random Forest Model Classification Report:")
st.text(classification_report(y_test, rf_pred))

fig, ax = plt.subplots(figsize=(8, 6))
rf_cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(fig)

st.write("Model Accuracy (Random Forest):", rf_model.score(X_test_scaled, y_test))

st.write("### Model Comparison")
st.write("SVM Accuracy:", svm_model.score(X_test_scaled, y_test))
st.write("Decision Tree Accuracy:", dt_model.score(X_test_scaled, y_test))
st.write("Random Forest Accuracy:", rf_model.score(X_test_scaled, y_test))