import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

def train_models(data):

    X = data[['day']]
    y = data['sales']

    models = {}

    lr = LinearRegression()
    lr.fit(X, y)
    models["linear_regression"] = lr

    rf = RandomForestRegressor()
    rf.fit(X, y)
    models["random_forest"] = rf

    dt = DecisionTreeRegressor()
    dt.fit(X, y)
    models["decision_tree"] = dt

    ridge = Ridge()
    ridge.fit(X, y)
    models["ridge"] = ridge

    return models


def predict_sales(models, model_name, day):

    model = models.get(model_name)

    if model is None:
        return None

    prediction = model.predict([[day]])
    return prediction[0]