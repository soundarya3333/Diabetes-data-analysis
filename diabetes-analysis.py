import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_diabetes_data(file_path):
    """
    Comprehensive analysis of diabetes dataset including statistical analysis,
    visualization, and predictive modeling.
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # 1. Basic Statistics
    print("=== Dataset Overview ===")
    print(f"Total records: {len(df)}")
    print(f"Diabetic cases: {len(df[df['Outcome'] == 1])}")
    print(f"Non-diabetic cases: {len(df[df['Outcome'] == 0])}")
    
    # 2. Statistical Summary
    print("\n=== Statistical Summary ===")
    print(df.describe())
    
    # 3. Check for missing values and zeros
    print("\n=== Missing Values Analysis ===")
    print("Number of zeros in each column:")
    for column in df.columns:
        zero_count = len(df[df[column] == 0])
        print(f"{column}: {zero_count} zeros")
    
    # 4. Correlation Analysis
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Diabetes Features')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # 5. Feature Distribution by Outcome
    features_to_plot = ['Glucose', 'BMI', 'Age', 'BloodPressure']
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features_to_plot, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x='Outcome', y=feature, data=df)
        plt.title(f'{feature} Distribution by Diabetes Outcome')
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()
    
    # 6. Predictive Modeling
    print("\n=== Predictive Modeling ===")
    
    # Prepare the data
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Print classification report
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # 7. Risk Analysis
    def calculate_risk_score(row):
        """Calculate a simple risk score based on key factors"""
        risk_score = 0
        if row['Glucose'] > 140: risk_score += 2
        if row['BMI'] > 30: risk_score += 1.5
        if row['Age'] > 40: risk_score += 1
        if row['BloodPressure'] > 90: risk_score += 1
        return risk_score
    
    df['RiskScore'] = df.apply(calculate_risk_score, axis=1)
    print("\n=== Risk Analysis ===")
    print("Risk Score Statistics:")
    print(df['RiskScore'].describe())
    
    return df, rf_model

if __name__ == "__main__":
    # Usage example
    df, model = analyze_diabetes_data('diabetes.csv')
