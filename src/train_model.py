import pathlib
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def guess_from_mean(X, y):
    """Create a baseline estimate for mean square error if we just guess the mean rating of the training set."""
    mses = []
    for seed_num in range(10):
        cv = KFold(n_splits=5, shuffle=True, random_state=seed_num)
        for train_index, test_index in cv.split(X, y):
            y_train, y_test = y.loc[train_index], y.loc[test_index]
            y_mean = y_train.mean()
            y_pred = y_test.copy()
            y_pred.loc[:] = y_mean
            mses.append(mean_squared_error(y_test, y_pred))
    return mses


def split_features_labels(data):
    X = data[[col for col in data.columns if col != 'Rating']]
    y = data['Rating']
    return X, y


def plot_errors(errors, name):
    plot_folder = pathlib.Path("../plots")
    plot_folder.mkdir(exist_ok=True)
    histogram = sns.histplot(data=errors)
    fig = histogram.get_figure()
    fig.savefig(plot_folder / (name + ".png"))


def preprocess_for_random_forest(X_train, X_test):
    X_train_processed = pd.get_dummies(X_train)
    X_test_processed = pd.get_dummies(X_test)
    # Get columns in the training set that are missing in test
    # This arises due to the separate one-hot encoding processes.
    missing_cols = set(X_train_processed.columns) - set(X_test_processed.columns)
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        X_test_processed[c] = 0
    X_train_processed, X_test_processed = X_train_processed.align(X_test_processed,
                                                                  join='inner',
                                                                  axis=1)  # inner join removes unshared columns
    return X_train_processed, X_test_processed


def nested_cross_validation(X_data, y_data):
    p_grid = {"n_estimators": [50, 100, 200], 'max_depth': [None, 3, 4]}
    X_data = pd.get_dummies(X_data)
    rf = RandomForestRegressor()
    NUM_TRIALS = 5
    nested_scores = []
    # Loop for each trial
    for i in range(NUM_TRIALS):
        # Choose cross-validation techniques for the inner and outer loops,
        # independently of the dataset.
        # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
        inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

        # Non_nested parameter search and scoring
        clf = GridSearchCV(estimator=rf, param_grid=p_grid, cv=inner_cv, n_jobs=4)
        clf.fit(X_data, y_data)

        # Nested CV with parameter optimization
        nested_score = cross_val_score(clf, X=X_data, y=y_data, cv=outer_cv, n_jobs=4)
        nested_scores.append(nested_score)
    return nested_scores


if __name__ == "__main__":
    data = pd.read_pickle("../data/data_processed.pickle")
    X, y = split_features_labels(data)
    baseline_errors = pd.DataFrame(guess_from_mean(X, y))
    random_forest_scores = pd.DataFrame(nested_cross_validation(X, y))
    random_forest_scores.to_pickle("../data/random_forest_scores.pickle")
    plot_errors(baseline_errors, "mean_guessing")
    plot_errors(pd.concat([random_forest_scores[col] for col in random_forest_scores.columns]).reset_index().drop(columns=['index']),
                name='random_forest_scores')