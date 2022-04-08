import json
import itertools
import pandas as pd
from preprocess import load_ctr_data, load_categorical_features_info, \
    load_used_model_ctrs, hash_categorical_columns, calc_ctr_features
from hashes import _hash_string_cat
from load_adult import load_adult_train, get_categorical_features


def main():
    cat_features = get_categorical_features()
    X_train, _, _ = load_adult_train("adult.data")
    for col_name in X_train:
        if col_name in cat_features:
            print(f"{col_name:=^40}")
            X_train[col_name] = X_train[col_name].astype("category")
            for cat_code, category in enumerate(X_train[col_name].cat.categories):
                encoded_category = _hash_string_cat(str(category))
                print(f"{cat_code: <2}  {category: <23}  {encoded_category: >10}")

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

    ref_row = X_train.iloc[0:1, :]
    df = pd.concat([ref_row] * n_combo, ignore_index=True)
    for row_id, item in enumerate(cartesian_product):
        df.iloc[row_id, used_cat_features_flat_index] = item

    converted_df = hash_categorical_columns(df, categorical_features=cat_features)
    print(converted_df)
    ctr_features = calc_ctr_features(converted_df=converted_df,
                                     ctr_data=ctr_data,
                                     used_model_ctrs=used_model_ctrs,
                                     cat_features_info=cat_features_info)
    print(f"ctr_features = {ctr_features}")


if __name__ == "__main__":
    main()
