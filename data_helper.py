import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split


def load_data(data_url):
    data = pd.read_csv(data_url, header=0)

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


def upload_data(data, name, output_path):
    p = os.path.join(output_path, f'{name}.joblib')
    joblib.dump(data, p)
