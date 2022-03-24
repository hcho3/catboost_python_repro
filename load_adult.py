#!/usr/bin/env python
import pandas as pd

column_names = ["age", "work_class", "final_weight", "education", "education_num", 
                "marital_status", "occupation", "relationship", "race", "sex", "capital_gain",
                "capital_loss", "hours_per_week", "native_country", "income_over_50k"]


categorical_features = ["work_class", "marital_status", "occupation", "relationship", "race",
                        "sex", "born_usa"]


def get_categorical_features():
    return categorical_features


def process_df(df):
    for column in ["work_class", "education", "marital_status", "occupation",
                   "relationship", "race", "sex", "native_country", "income_over_50k"]:
        df[column] = df[column].str.lstrip()
    # Drop rows with missing values
    df = df[(df["work_class"] != "?") & (df["occupation"] != "?") & (df["native_country"] != "?")]
    # Combine captial variables
    df["capital"] = df["capital_gain"] - df["capital_loss"]
    df = df.drop(["capital_gain", "capital_loss"], axis=1)
    # Drop education, as it's duplicate of education_num
    df = df.drop("education", axis=1)
    # Simplify native_country to USA vs elsewhere
    df["born_usa"] = (df["native_country"] == "United-States").astype("int")
    df = df.drop("native_country", axis=1)
    # Encode binary target
    df["income_over_50k"] = df["income_over_50k"].replace({"<=50K": 0, ">50K": 1,
                                                           "<=50K.": 0, ">50K.": 1})
    # Extract sample weights
    sample_weight = df["final_weight"]
    sample_weight = sample_weight / sample_weight.mean()
    df = df.drop("final_weight", axis=1)
    return df.drop("income_over_50k", axis=1), df["income_over_50k"], sample_weight

def load_adult_train(train_path):
    train = pd.read_csv(train_path, names=column_names)
    X_train, y_train, sample_weight = process_df(train)
    return X_train, y_train, sample_weight

def load_adult_test(test_path):
    test = pd.read_csv(test_path, names=column_names, skiprows=1)
    X_test, y_test, test_sample_weight = process_df(test)
    return X_test, y_test, test_sample_weight
