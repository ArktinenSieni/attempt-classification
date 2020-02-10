import pandas as pd
import os

from sklearn.model_selection import train_test_split, cross_validate
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from pathlib import Path
from joblib import dump, load
from data_helper import load_data, upload_data


def _perform_cross_validation_and_save(clf, X, y, name, output_path):
    print(f'--- cross validating {name}             ---')
    result = cross_validate(clf, X, y, n_jobs=5,
                            return_estimator=True, cv=5)

    print(f"{name} fit times: {result['fit_time']}")
    print(f"{name} test scores: {result['score_time']}")
    print(f'--- cross validation for {name} ended   ---')

    for i, estimator in enumerate(result['estimator']):
        upload_data(estimator, f'{name}_{i}', output_path)

    return result['estimator']


def _upload_classifier(clf, clf_name, output_path):
    # p = Path(f'{clf_name}.joblib')
    p = os.path.join(output_path, f'{clf_name}.joblib')
    dump(clf, p)


def _validate_classifier(clf, X, y):
    predictions = pd.Series(clf.predict(X)).apply(int)
    expected_int = y.apply(lambda exp: int(exp) if pd.notna(exp) else 0)

    score = accuracy_score(expected_int, predictions)

    return score


def _validate_in_location(path, X, y):
    clf = load(path)

    print(clf)

    return _validate_classifier(clf, X, y)


def main():
    INPUTS_DIR = os.getenv('VH_INPUTS_DIR', '../data')
    OUTPUTS_DIR = os.getenv('VH_OUTPUTS_DIR', './models')
    data_name = 'sql_trainer_filtered_attempts.csv'
    data_path = os.path.join(INPUTS_DIR,
                             'filtered-data',
                             data_name)

    X_train, X_val, y_train, y_val = load_data(data_path)

    classifiers = {
        # "linear SVM": svm.SVC(kernel='linear', gamma='scale'),
        "polynomialSVM": svm.SVC(kernel='poly', gamma='scale'),
        # "radialSVM": svm.SVC(kernel='rbf', gamma='scale'),
        # "sigmoidSVM": svm.SVC(kernel='sigmoid'),
        # "GaussianNB": GaussianNB(),
        # "MultinomialNB": MultinomialNB(),
        # "ComplementNB": ComplementNB(),
        # "BernoulliNB": BernoulliNB(),
        # "LDA": LinearDiscriminantAnalysis(),
        # "Randomforest": RandomForestClassifier(n_estimators=100, criterion="gini", bootstrap=True, oob_score=True),
        # "Linearregression": LinearRegression(normalize=True)
    }

    clf_estimators = dict(map(
        lambda name_and_clf: (
            name_and_clf[0], _perform_cross_validation_and_save(
                name_and_clf[1], X_train, y_train, name_and_clf[0], OUTPUTS_DIR)
        ),
        classifiers.items()
    ))


if __name__ == "__main__":
    main()
