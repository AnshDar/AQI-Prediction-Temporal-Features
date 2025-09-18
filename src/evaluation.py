import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model, X, y_true):
    y_pred = model.predict(X)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "r2": r2_score(y_true, y_pred)
    }

def plot_feature_importance(model, features, outpath):
    importance = model.feature_importances_
    imp_df = pd.Series(importance, index=features).sort_values(ascending=True).tail(20)
    imp_df.plot(kind='barh')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
