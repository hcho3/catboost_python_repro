import json
import itertools
import pandas as pd
import numpy as np
from preprocess import load_ctr_data, load_categorical_features_info, \
    load_used_model_ctrs, hash_categorical_columns, calc_ctr_features
from hashes import _hash_string_cat
from load_adult import load_adult_train, get_categorical_features


def wait_for_keypress():
    input("Press Enter to continue...")


def main():
    pd.set_option("display.max_rows", 6)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.expand_frame_repr", False)
    np.set_printoptions(threshold=100)

    cat_features = get_categorical_features()
    X_train, _, _ = load_adult_train("adult.data")
    print(f"X_train =\n{X_train.head(10)}")
    wait_for_keypress()

    for col_name in X_train:
        if col_name in cat_features:
            print(f"{col_name:=^40}")
            X_train[col_name] = X_train[col_name].astype("category")
            for cat_code, category in enumerate(X_train[col_name].cat.categories):
                encoded_category = _hash_string_cat(str(category))
                print(f"{cat_code: <2}  {category: <23}  {encoded_category: >10}")
    wait_for_keypress()

    with open("adult_model.json", "r") as f:
        model = json.load(f)

    cat_features_info = load_categorical_features_info(model)
    ctr_data = load_ctr_data(model)
    used_model_ctrs = load_used_model_ctrs(model)

    used_cat_features_flat_index = set(
        cat_features_info[x].position.flat_index
        for model_ctr in used_model_ctrs
        for x in model_ctr.ctr.base.projection.cat_features
    )
    used_cat_features_flat_index = list(used_cat_features_flat_index)
    cartesian_product = list(itertools.product(*[
        X_train.iloc[:, col_id].cat.categories for col_id in used_cat_features_flat_index
    ]))
    n_combo = len(cartesian_product)

    print("Iterating all combination of categorical features...")
    ref_row = X_train.iloc[3:4, :]
    df = pd.concat([ref_row] * n_combo, ignore_index=True)
    for row_id, item in enumerate(cartesian_product):
        df.iloc[row_id, used_cat_features_flat_index] = item
    print(f"df =\n{df}")
    wait_for_keypress()

    converted_df = hash_categorical_columns(df, categorical_features=cat_features)
    print(f"converted_df =\n{converted_df}")
    wait_for_keypress()
    ctr_features = calc_ctr_features(converted_df=converted_df,
                                     ctr_data=ctr_data,
                                     used_model_ctrs=used_model_ctrs,
                                     cat_features_info=cat_features_info)
    print(f"ctr_features =\n{ctr_features}")

    # Compare against correct CTR features from Catboost. The reference CTR features were
    # computed by running the following code snippet:
    #   clf = CatBoostClassifier()
    #   clf.load_model("adult_model.json", format="json")
    #   pool_test = Pool(df, cat_features=cat_features)
    #   y_leaf = clf.calc_leaf_indexes(pool_test, thread_count=1)
    ref = [
        2.00699, 2.00699, 2.00699, 2.00699, 2.00699, 2.00699, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 3.37881,
        3.37881, 3.37881, 3.37881, 3.37881, 3.37881, 7.27648, 7.27648, 7.27648, 7.27648, 7.27648,
        7.27648, 1.74242, 1.74242, 1.74242, 1.74242, 1.74242, 1.74242, 0.92154, 0.92154, 0.92154,
        0.92154, 0.92154, 0.92154, 1.86833, 1.86833, 1.86833, 1.86833, 1.86833, 1.86833, 0.616247,
        0.616247, 0.616247, 0.616247, 0.616247, 0.616247, 0.104167, 0.104167, 0.104167, 0.104167,
        0.104167, 0.104167, 6.72567, 6.72567, 6.72567, 6.72567, 6.72567, 6.72567, 4.88372, 4.88372,
        4.88372, 4.88372, 4.88372, 4.88372, 4.05858, 4.05858, 4.05858, 4.05858, 4.05858, 4.05858,
        4.56736, 4.56736, 4.56736, 4.56736, 4.56736, 4.56736, 3.04196, 3.04196, 3.04196, 3.04196,
        3.04196, 3.04196, 2.009, 2.009, 2.009, 2.009, 2.009, 2.009, 2.25, 2.25, 2.25, 2.25, 2.25,
        2.25, 3.38067, 3.38067, 3.38067, 3.38067, 3.38067, 3.38067, 7.27836, 7.27836, 7.27836,
        7.27836, 7.27836, 7.27836, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 0.927091, 0.927091, 0.927091,
        0.927091, 0.927091, 0.927091, 1.87214, 1.87214, 1.87214, 1.87214, 1.87214, 1.87214,
        0.618581, 0.618581, 0.618581, 0.618581, 0.618581, 0.618581, 0.15625, 0.15625, 0.15625,
        0.15625, 0.15625, 0.15625, 6.72753, 6.72753, 6.72753, 6.72753, 6.72753, 6.72753, 4.89535,
        4.89535, 4.89535, 4.89535, 4.89535, 4.89535, 4.06067, 4.06067, 4.06067, 4.06067, 4.06067,
        4.06067, 4.57557, 4.57557, 4.57557, 4.57557, 4.57557, 4.57557, 3.04673, 3.04673, 3.04673,
        3.04673, 3.04673, 3.04673, 2.01102, 2.01102, 2.01102, 2.01102, 2.01102, 2.01102, 3, 3, 3, 3,
        3, 3, 3.38254, 3.38254, 3.38254, 3.38254, 3.38254, 3.38254, 7.28024, 7.28024, 7.28024,
        7.28024, 7.28024, 7.28024, 1.75758, 1.75758, 1.75758, 1.75758, 1.75758, 1.75758, 0.932642,
        0.932642, 0.932642, 0.932642, 0.932642, 0.932642, 1.87595, 1.87595, 1.87595, 1.87595,
        1.87595, 1.87595, 0.620915, 0.620915, 0.620915, 0.620915, 0.620915, 0.620915, 0.208333,
        0.208333, 0.208333, 0.208333, 0.208333, 0.208333, 6.72939, 6.72939, 6.72939, 6.72939,
        6.72939, 6.72939, 4.90698, 4.90698, 4.90698, 4.90698, 4.90698, 4.90698, 4.06276, 4.06276,
        4.06276, 4.06276, 4.06276, 4.06276, 4.58379, 4.58379, 4.58379, 4.58379, 4.58379, 4.58379,
        3.05149, 3.05149, 3.05149, 3.05149, 3.05149, 3.05149, 6.83448, 1.59764, 0.589888, 0.214909,
        0.994398, 7.39872, 6.83448, 1.59764, 0.589888, 0.214909, 0.994398, 7.39872, 6.83448,
        1.59764, 0.589888, 0.214909, 0.994398, 7.39872, 6.83448, 1.59764, 0.589888, 0.214909,
        0.994398, 7.39872, 6.83448, 1.59764, 0.589888, 0.214909, 0.994398, 7.39872, 6.83448,
        1.59764, 0.589888, 0.214909, 0.994398, 7.39872, 6.83448, 1.59764, 0.589888, 0.214909,
        0.994398, 7.39872, 6.83448, 1.59764, 0.589888, 0.214909, 0.994398, 7.39872, 6.83448,
        1.59764, 0.589888, 0.214909, 0.994398, 7.39872, 6.83448, 1.59764, 0.589888, 0.214909,
        0.994398, 7.39872, 6.83448, 1.59764, 0.589888, 0.214909, 0.994398, 7.39872, 6.83448,
        1.59764, 0.589888, 0.214909, 0.994398, 7.39872, 6.83448, 1.59764, 0.589888, 0.214909,
        0.994398, 7.39872, 6.83448, 1.59764, 0.589888, 0.214909, 0.994398, 7.39872, 6.83508,
        1.59862, 0.598315, 0.216588, 0.996732, 7.40405, 6.83508, 1.59862, 0.598315, 0.216588,
        0.996732, 7.40405, 6.83508, 1.59862, 0.598315, 0.216588, 0.996732, 7.40405, 6.83508,
        1.59862, 0.598315, 0.216588, 0.996732, 7.40405, 6.83508, 1.59862, 0.598315, 0.216588,
        0.996732, 7.40405, 6.83508, 1.59862, 0.598315, 0.216588, 0.996732, 7.40405, 6.83508,
        1.59862, 0.598315, 0.216588, 0.996732, 7.40405, 6.83508, 1.59862, 0.598315, 0.216588,
        0.996732, 7.40405, 6.83508, 1.59862, 0.598315, 0.216588, 0.996732, 7.40405, 6.83508,
        1.59862, 0.598315, 0.216588, 0.996732, 7.40405, 6.83508, 1.59862, 0.598315, 0.216588,
        0.996732, 7.40405, 6.83508, 1.59862, 0.598315, 0.216588, 0.996732, 7.40405, 6.83508,
        1.59862, 0.598315, 0.216588, 0.996732, 7.40405, 6.83508, 1.59862, 0.598315, 0.216588,
        0.996732, 7.40405, 6.83569, 1.59959, 0.606742, 0.218267, 0.999066, 7.40938, 6.83569,
        1.59959, 0.606742, 0.218267, 0.999066, 7.40938, 6.83569, 1.59959, 0.606742, 0.218267,
        0.999066, 7.40938, 6.83569, 1.59959, 0.606742, 0.218267, 0.999066, 7.40938, 6.83569,
        1.59959, 0.606742, 0.218267, 0.999066, 7.40938, 6.83569, 1.59959, 0.606742, 0.218267,
        0.999066, 7.40938, 6.83569, 1.59959, 0.606742, 0.218267, 0.999066, 7.40938, 6.83569,
        1.59959, 0.606742, 0.218267, 0.999066, 7.40938, 6.83569, 1.59959, 0.606742, 0.218267,
        0.999066, 7.40938, 6.83569, 1.59959, 0.606742, 0.218267, 0.999066, 7.40938, 6.83569,
        1.59959, 0.606742, 0.218267, 0.999066, 7.40938, 6.83569, 1.59959, 0.606742, 0.218267,
        0.999066, 7.40938, 6.83569, 1.59959, 0.606742, 0.218267, 0.999066, 7.40938, 6.83569,
        1.59959, 0.606742, 0.218267, 0.999066, 7.40938
    ]
    ref = np.array(ref).reshape((6, -1)).transpose()
    np.testing.assert_almost_equal(ctr_features, ref, decimal=5)


if __name__ == "__main__":
    main()
