import pandas as pd
from preprocessing import load_csv, interpolate_timeseries, train_test_split_time_series
from feature_engineering import add_lag_features, add_moving_averages, add_cyclical_time_features
from models import train_decision_tree, train_random_forest, train_xgboost
from evaluation import evaluate, plot_feature_importance

RAW_PATH = "../data/raw/aqi.csv"   # put your dataset here
TARGET = "PM2.5"

def main():
    print("🔹 Loading dataset...")
    df = load_csv(RAW_PATH, datetime_col='datetime')
    df = interpolate_timeseries(df, freq='H')

    # Split into train/test
    train_df, test_df = train_test_split_time_series(df)

    # Feature engineering
    cols = [c for c in df.columns if c != TARGET]
    df = add_lag_features(df, cols)
    df = add_moving_averages(df, cols)
    df = add_cyclical_time_features(df)
    df = df.dropna()

    train_df, test_df = train_test_split_time_series(df)

    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

    # Train models
    print("🔹 Training Decision Tree...")
    dt = train_decision_tree(X_train, y_train)
    print("DT:", evaluate(dt, X_test, y_test))

    print("🔹 Training Random Forest...")
    rf = train_random_forest(X_train, y_train)
    print("RF:", evaluate(rf, X_test, y_test))
    plot_feature_importance(rf, X_train.columns, "../results/figures/rf_importance.png")

    print("🔹 Training XGBoost...")
    xgb = train_xgboost(X_train, y_train)
    print("XGB:", evaluate(xgb, X_test, y_test))
    plot_feature_importance(xgb, X_train.columns, "../results/figures/xgb_importance.png")

if __name__ == "__main__":
    main()
