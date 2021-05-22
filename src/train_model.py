import pandas as pd
import pathlib
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import seaborn as sns


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


if __name__ == "__main__":
    data = pd.read_pickle("../data/data_processed.pickle")
    X, y = split_features_labels(data)
    baseline_errors = guess_from_mean(X, y)
    plot_errors(baseline_errors, "mean_guessing")

