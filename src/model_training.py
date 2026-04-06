from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def train_models(X, y):

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if y.nunique() < 20:
        # Classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier

        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Random Forest": RandomForestClassifier()
        }

    else:
        # Regression
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor()
        }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models, X_test, y_test






#"Decision Tree": DecisionTreeClassifier(),
#"Gradient Boosting": GradientBoostingClassifier()