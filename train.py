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


def _load_data(data_url):
    data = pd.read_csv(data_url, header=0)[:10]

    data_selected_columns = [
        # 'user_id', # Don't want to fit on basis of users
        'topic_id',
        # 'topic_name', # already in `topic_id`
        'exercise_id',
        # 'exercise_name', # already in `exercise_id`
        # 'model_solution', # not relevant for now: would require some complicated stuff
        'exercise_created_by_user',
        'attempt_id',
        # 'attempt_time', # not readable by models. Already in `week_number`, `period`, `period_week` and `attempt_number`
        # 'attempt_code', # not relevant for now: would require some complicated stuff
        # 'attempt_correct', # The value which is being inspected
        'week_number',
        'period',
        'period_week',
        'attempt_number',
        'topic_1_count',
        'topic_4_count',
        'topic_2_count',
        'topic_3_count',
        'topic_5_count',
        'topic_8_count',
        'topic_9_count',
        'topic_10_count',
        'topic_11_count',
        'topic_12_count',
        'topic_13_count',
        'topic_14_count'
    ]

    X = data[data_selected_columns]
    y = data['attempt_correct']

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.1)

    return X_train, X_validation, y_train, y_validation


def _perform_cross_validation(clf, X, y, name):
    print(f'--- cross validating {name}             ---')
    result = cross_validate(clf, X, y, n_jobs=5,
                            return_estimator=True)

    print(result['fit_time'])
    print(result['score_time'])
    print(f'--- cross validation for {name} ended   ---')

    return result['estimator']


def _upload_classifier(clf, clf_name):
    p = Path(f'{clf_name}.joblib')
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
    data = 'sql_trainer_filtered_attempts.csv'

    INPUTS_DIR = os.getenv('VH_INPUTS_DIR', '../data')
    OUTPUTS_DIR = os.getenv('VH_OUTPUTS_DIR', './models')

    X_train, X_val, y_train, y_val = _load_data(f'{INPUTS_DIR}/{data}')

    classifiers = {
        "polynomialSVM": svm.SVC(kernel='poly'),
        "radialSVM": svm.SVC(kernel='rbf'),
        "sigmoidSVM": svm.SVC(kernel='sigmoid'),
        "GaussianNB": GaussianNB(),
        "MultinomialNB": MultinomialNB(),
        "ComplementNB": ComplementNB(),
        "BernoulliNB": BernoulliNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "Randomforest": RandomForestClassifier(n_estimators=100, criterion="gini", bootstrap=True, oob_score=True),
        "Linearregression": LinearRegression(normalize=True)
    }

    clf_estimators = dict(map(
        lambda name_and_clf: (
            name_and_clf[0], _perform_cross_validation(
                name_and_clf[1], X_train, y_train, name_and_clf[0])
        ),
        classifiers.items()
    ))

    for clf_name in clf_estimators:
        estimators = clf_estimators[clf_name]
        for i, estimator in enumerate(estimators):
            clf_i_name = f'{clf_name}_{i}'

            _upload_classifier(
                estimator, f'{OUTPUTS_DIR}/{clf_i_name}')

            print(
                f'{clf_i_name} score: {_validate_classifier(estimator, X_val, y_val)}')


if __name__ == "__main__":
    main()
