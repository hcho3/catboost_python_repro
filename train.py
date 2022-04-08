#!/usr/bin/env python
import pandas as pd
from catboost import CatBoostClassifier, Pool
from load_adult import process_df, load_adult_train, load_adult_test, get_categorical_features


def main():
    pd.set_option("display.max_columns", None)

    X_train, y_train, sample_weight = load_adult_train("adult.data")
    print(X_train.head(10))
    X_test, y_test, test_sample_weight = load_adult_test("adult.test")

    clf = CatBoostClassifier(n_estimators=30,
                             loss_function="Logloss",
                             learning_rate=0.1,
                             depth=3, task_type="CPU",
                             simple_ctr=["Borders"],  # Other CTR modes not yet supported
                             random_state=1,
                             one_hot_max_size=0,
                             thread_count=1,
                             verbose=True,
                             allow_writing_files=False)

    cat_features = get_categorical_features()
    pool_train = Pool(X_train, y_train, cat_features=cat_features, weight=sample_weight)

    model_path = "adult_model.json"

    clf.fit(pool_train)
    clf.save_model(model_path, format="json")

    clf2 = CatBoostClassifier()
    clf2.load_model(model_path, format="json")

    print(X_test.head(2))
    pool_test = Pool(X_test.head(2), cat_features=cat_features)
    y_leaf = clf2.calc_leaf_indexes(pool_test, thread_count=1)
    print(f"y_leaf =\n{y_leaf}")


if __name__ == "__main__":
    main()
