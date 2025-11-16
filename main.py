import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    #  Load your data (change the filename if needed!)
    df = pd.read_csv('HousingData.csv')  # or 'Boston.csv', etc.

    #  Handle missing data (drop or fill)
    if df.isnull().values.any():
        print('\nMissing values found. Filling with column means...')
        df = df.fillna(df.mean())

    #  Inspect data
    print("\nFirst five rows:")
    print(df.head())
    print("\nData info:")
    print(df.info())
    print("\nSummary statistics:")
    print(df.describe())

    #  Visualizations
    plt.figure(figsize=(8,5))
    sns.histplot(df['MEDV'], bins=30, kde=True)
    plt.title('Distribution of Median House Prices')
    plt.xlabel('MEDV ($1000s)')
    plt.ylabel('Frequency')
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,5))
    if 'RM' in df.columns:
        sns.scatterplot(x='RM', y='MEDV', data=df)
        plt.title('Number of Rooms vs. House Price')
        plt.xlabel('Average Number of Rooms (RM)')
        plt.ylabel('Median Value (MEDV $1000s)')
        plt.tight_layout(); plt.show()

    plt.figure(figsize=(12,10))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout(); plt.show()

    #  Prepare features and target
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    #  Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Train shape:', X_train.shape, y_train.shape)
    print('Test shape:', X_test.shape, y_test.shape)

    #  Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #  Train & Evaluate Multiple Models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest (10 trees)": RandomForestRegressor(n_estimators=10, random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"\n{name} Results:")
        print(f"  Mean Squared Error: {mse:.2f}")
        print(f"  R2 Score: {r2:.3f}")

    #  Fine-Tune Decision Tree with GridSearchCV (small grid, fast CV)
    print('\nStarting fine-tuning with GridSearchCV for Decision Tree...')
    param_grid = {
        'max_depth': [3, 5],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)
    print("Best Decision Tree Parameters:", grid_search.best_params_)
    dt_best = grid_search.best_estimator_
    y_pred_grid = dt_best.predict(X_test_scaled)
    print("Fine-tuned Decision Tree Results:")
    print("  Mean Squared Error: {:.2f}".format(mean_squared_error(y_test, y_pred_grid)))
    print("  R2 Score: {:.3f}".format(r2_score(y_test, y_pred_grid)))

    #  Visualize Actual vs. Predicted for Fine-tuned Decision Tree
    plt.figure(figsize=(8,5))
    plt.scatter(y_test, y_pred_grid, alpha=0.7)
    plt.xlabel('Actual MEDV ($1000s)')
    plt.ylabel('Predicted MEDV ($1000s)')
    plt.title('Actual vs. Predicted (Fine-tuned Decision Tree)')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()

