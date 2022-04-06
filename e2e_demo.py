import json
from preprocess import hash_categorical_columns, load_ctr_data, load_float_features_info, \
        load_categorical_features_info, load_used_model_ctrs, calc_converted_input
from evaluate import load_oblivious_trees, predict_leaf
from load_adult import process_df, load_adult_train, load_adult_test, get_categorical_features

def main():
    cat_features = get_categorical_features()
    X_test, _, _ = load_adult_test("adult.test")
    X_test = X_test.head(2)
    print(X_test)
    converted_df = hash_categorical_columns(X_test, categorical_features=cat_features)
    print(converted_df)

    with open("adult_model.json", "r") as f:
        model = json.load(f)
    ctr_data = load_ctr_data(model)
    print(f"ctr_data = {ctr_data}")

    float_features_info = load_float_features_info(model)
    print(f"float_features_info = {float_features_info}")
    cat_features_info = load_categorical_features_info(model)
    print(f"cat_features_info = {cat_features_info}")
    used_model_ctrs = load_used_model_ctrs(model)
    print(f"used_model_ctrs = {used_model_ctrs}")

    converted_input = calc_converted_input(converted_df=converted_df,
                                           ctr_data=ctr_data,
                                           used_model_ctrs=used_model_ctrs,
                                           float_features_info=float_features_info,
                                           cat_features_info=cat_features_info)
    print(f"converted_input = {converted_input}")
    trees = load_oblivious_trees(model,
                                 float_features_info=float_features_info,
                                 used_model_ctrs=used_model_ctrs)
    print(f"trees = {trees}")
    pred_leaf = predict_leaf(trees, converted_input=converted_input, doc_count=len(converted_df))
    print(f"pred_leaf =\n{pred_leaf}")

if __name__ == "__main__":
    main()
