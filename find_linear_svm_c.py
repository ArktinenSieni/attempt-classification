import os
import time
import pandas as pd
from data_helper import load_data
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from thundersvm import SVC


def validate_classifier(estimator, data, expected):
    predictions = pd.Series(estimator.predict(data)).apply(int)

    if expected.dtypes != 'int':
        expected_int = expected.apply(
            lambda exp: int(exp) if pd.notna(exp) else 0)
    else:
        expected_int = expected

    score = accuracy_score(expected_int, predictions)

    return score


def timed_function(func, print_prefix=""):
    start_time = time.time()
    start_time_print = time.asctime(time.localtime(start_time))
    print('%s starting fitting at %s' % (print_prefix, start_time_print))
    result = func()

    end_time = time.time()
    time_delta = end_time - start_time
    end_time_print = time.asctime(time.localtime(end_time))
    print('%s Time spent fitting: %s seconds. Finished at %s' %
          (print_prefix, time_delta, end_time_print))

    return result


def fit_and_test_model(clf, clf_name, X_trn, X_tst, y_trn, y_tst):
    estimator = timed_function(lambda: clf.fit(X_trn, y_trn), clf_name)
    score = validate_classifier(estimator, X_tst, y_tst)

    return {'name': clf_name, 'estimator': estimator, 'score': score}


def fit_and_test_split(clf, clf_name, split, split_index, data, expected):
    train_index, test_index = split[split_index]

    data_train, data_test = data.iloc[train_index], data.iloc[test_index]
    exp_train, exp_test = expected.iloc[train_index], expected[test_index]

    print(data_train.shape)

    name_with_index = '%s_%s' % (clf_name, split_index)

    result = fit_and_test_model(
        clf, name_with_index, data_train, data_test, exp_train, exp_test)

    return result


def upload_tsvm_classifier(clf, clf_name, output_path):
    dump_name = os.path.join(output_path, '%s.tsvm.sav' % clf_name)

    clf.save_to_file(dump_name)


def train_classifier_with_split(splits, index, clf, X, y, name, upload_func, output_path):
    result = fit_and_test_split(clf, name, splits, index, X, y)
    print(result['score'])

    upload_func(result['estimator'], result['name'], output_path)

    return result


def main():
    INPUTS_DIR = os.getenv('VH_INPUTS_DIR', '../data')
    OUTPUTS_DIR = os.getenv('VH_OUTPUTS_DIR', './models')
    data_name = 'sql_trainer_filtered_attempts.csv'
    data_path = os.path.join(INPUTS_DIR,
                             'filtered-data',
                             data_name)

    X_train, X_val, y_train, y_val = load_data(data_path)

    linear_scores = dict()

    split_amount = 5  # 5 is default
    kf_splits = list(KFold(n_splits=split_amount).split(X_train))

    C_key = 1

    for i in range(0, len(kf_splits)):
        clf = SVC(kernel='linear', C=1)
        res = train_classifier_with_split(
            kf_splits, i, clf, X_train[:10], y_train[:10], 'linearSVM', upload_tsvm_classifier, OUTPUTS_DIR)
        linear_scores.setdefault(C_key, []).append(res['score'])

    for key, val in linear_scores.values():
        print('%s: %s' % (key, val))


if __name__ == "__main__":
    main()
