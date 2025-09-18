from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def train_decision_tree(X, y):
    model = DecisionTreeRegressor(random_state=42)
    return model.fit(X, y)

def train_random_forest(X, y):
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    return model.fit(X, y)

def train_xgboost(X, y):
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, learning_rate=0.05)
    return model.fit(X, y)
