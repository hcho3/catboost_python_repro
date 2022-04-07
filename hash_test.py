import json
from preprocess import load_used_model_ctrs, _calc_ctr_hashes
from hashes import _hash_string_cat
from load_adult import load_adult_train, get_categorical_features


def main():
    cat_features = get_categorical_features()
    X_train, _, _ = load_adult_train("adult.data")
    for col_name in X_train.keys():
        if col_name in cat_features:
            print(f"{col_name:=^40}")
            X_train[col_name] = X_train[col_name].astype("category")
            for cat_code, category in enumerate(X_train[col_name].cat.categories):
                encoded_category = _hash_string_cat(str(category))
                print(f"{cat_code: <2}  {category: <23}  {encoded_category: >10}")

    with open("adult_model.json", "r") as f:
        model = json.load(f)

    used_model_ctrs = load_used_model_ctrs(model)
    print(f"used_model_ctrs = {used_model_ctrs}")

    for ctr_id, model_ctr in enumerate(used_model_ctrs):
        transposed_cat_feature_indexes = model_ctr.ctr.base.projection.cat_features
        binarized_indexes = []
        ctr_hashes = _calc_ctr_hashes(binarized_features=None,
                                      hashed_cat_features=hashed_cat_features,
                                      transposed_cat_feature_indexes=transposed_cat_feature_indexes,
                                      binarized_indexes=binarized_indexes,
                                      doc_count=doc_count)

if __name__ == "__main__":
    main()
