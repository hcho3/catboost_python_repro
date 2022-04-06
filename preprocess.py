import warnings
import json
import pandas as pd
import numpy as np
from typing import List
from hashes import _hash_string_cat
from class_defs import TFloatSplit, TOneHotSplit, TFeatureCombination, TModelCtrBase, \
    TModelCtr, TBucket, TDenseIndexHashBuilder, TDenseIndexHashView, TCtrValueTable, \
    TCtrData, TCtrFeature, TFeaturePosition, TCatFeature, TFloatFeature

MAX_VALUES_PER_BIN = 254


def _hash_categorical_column(cat_column):
    return pd.Series([_hash_string_cat(str(cat)) for cat in cat_column])


def hash_categorical_columns(df, *, categorical_features):
    converted_df = {}
    for (col_name, column) in df.items():
        if col_name in categorical_features:
            if column.dtype.name == "category":
                raise NotImplementedError("")
            converted_df[col_name] = _hash_categorical_column(column)
        else:
            converted_df[col_name] = column.copy()
    converted_df = pd.DataFrame(converted_df)
    return converted_df


def _load_projection(projection_elements):
    cat_features = []
    bin_features = []
    onehot_features = []
    for feat_ident in projection_elements:
        feat_type = feat_ident["combination_element"]
        if feat_type == "cat_feature_value":
            cat_features.append(feat_ident["cat_feature_index"])
        elif feat_type == "float_feature":
            feat = TFloatSplit(float_feature=feat_ident["float_feature_index"],
                               split=feat_ident["border"])
            bin_features.append(feat)
        else:
            feat = TOneHotSplit(cat_feature_idx=feat_ident["cat_feature_index"],
                                value=feat_ident["value"])
            onehot_features.append(feat)
    projection = TFeatureCombination(cat_features=cat_features,
                                     bin_features=bin_features,
                                     onehot_features=onehot_features)
    return projection


def load_ctr_base_from_str(s):
    ctr_base_json = json.loads(s)
    ctr_type = ctr_base_json["type"]

    projection = _load_projection(ctr_base_json["identifier"])
    ctr_base = TModelCtrBase(projection=projection,
                             ctr_type=ctr_type)
    return ctr_base


def _fast_clp2(t: int):
    """
    Computes the next power of 2 higher or equal to the integer parameter `t`.
     * If `t` is a power of 2 will return `t`.
     * Result is undefined for `t == 0`.
    """
    assert t > 0
    p = 1
    while p < t:
        p *= 2
    return p


def _get_proper_bucket_count(unique_values_count: int, *, load_factor: float = 0.5):
    if unique_values_count == 0:
        return 2
    return _fast_clp2(int(unique_values_count / load_factor))


def _load_learn_ctr(ctr_data_entry, *, ctr_base: TModelCtrBase):
    hash_stride = ctr_data_entry["hash_stride"]
    target_classes_count = hash_stride - 1
    hash_map = ctr_data_entry["hash_map"]
    if ctr_base.ctr_type != "Borders" or target_classes_count != 2:
        raise NotImplementedError("")
    blob_size = len(hash_map) // hash_stride
    bucket_cnt = _get_proper_bucket_count(blob_size)
    index_hash_builder = TDenseIndexHashBuilder(
        hash_mask=bucket_cnt - 1,
        buckets=[TBucket() for _ in range(bucket_cnt)],
        bin_count=0)
    ctr_int_array = np.zeros(blob_size * target_classes_count, dtype=np.int64)
    hash_map_idx = 0
    while hash_map_idx < len(hash_map):
        hash_value = int(hash_map[hash_map_idx])
        hash_map_idx += 1
        index = index_hash_builder.add_index(hash_value)
        for idx in range(index * target_classes_count, (index + 1) * target_classes_count):
            ctr_int_array[idx] = hash_map[hash_map_idx]
            hash_map_idx += 1
    learn_ctr = TCtrValueTable(model_ctr_base=ctr_base,
                               index_buckets=index_hash_builder.buckets,
                               ctr_int_array=ctr_int_array.tolist(),
                               target_classes_count=target_classes_count)
    return learn_ctr


def load_ctr_data(model):
    learn_ctrs = {}
    for key, value in model["ctr_data"].items():
        ctr_base = load_ctr_base_from_str(key)
        if (ctr_base.projection.bin_features or ctr_base.projection.onehot_features or
                len(ctr_base.projection.cat_features) > 1):
            raise NotImplementedError("")
        learn_ctrs[ctr_base] = _load_learn_ctr(value, ctr_base=ctr_base)
    return TCtrData(learn_ctrs=learn_ctrs)


def load_float_features_info(model):
    float_features_info = []
    if "float_features" in model["features_info"]:
        for entry in model["features_info"]["float_features"]:
            feat = TFloatFeature(has_nans=entry["has_nans"],
                                 position=TFeaturePosition(entry["feature_index"],
                                                           entry["flat_feature_index"]),
                                 borders=entry["borders"],
                                 nan_value_treatment=entry["nan_value_treatment"])
            float_features_info.append(feat)
    return float_features_info


def load_categorical_features_info(model):
    cat_features_info = []
    if "categorical_features" in model["features_info"]:
        for cat_entry in model["features_info"]["categorical_features"]:
            if "values" in cat_entry:
                raise NotImplementedError("")
            position = TFeaturePosition(index=cat_entry["feature_index"],
                                        flat_index=cat_entry["flat_feature_index"])
            cat_features_info.append(TCatFeature(position=position))
    for i, cat_feature in enumerate(cat_features_info):
        assert i == cat_feature.position.index
    return cat_features_info


def load_used_model_ctrs(model):
    used_model_ctrs = []
    if "ctrs" in model["features_info"]:
        for ctr_entry in model["features_info"]["ctrs"]:
            model_ctr_base = TModelCtrBase(projection=_load_projection(ctr_entry["elements"]),
                                           ctr_type=ctr_entry["ctr_type"])
            model_ctr = TModelCtr(base=model_ctr_base,
                                  target_border_idx=ctr_entry["target_border_idx"],
                                  prior_num=ctr_entry["prior_numerator"],
                                  prior_denom=ctr_entry["prior_denomerator"],
                                  shift=ctr_entry["shift"],
                                  scale=ctr_entry["scale"])
            if len(ctr_entry["borders"]) >= MAX_VALUES_PER_BIN:
                raise NotImplementedError("")
            used_model_ctrs.append(TCtrFeature(ctr=model_ctr, borders=ctr_entry["borders"]))
    return used_model_ctrs


def _calc_ctr_hashes(*, binarized_features, hashed_cat_features, transposed_cat_feature_indexes,
                     binarized_indexes, doc_count) -> List[int]:
    def _calc_hash(a: np.uint64, b: np.uint64):
        MAGIC_MULT = np.uint64(0x4906ba494954cb65)
        return MAGIC_MULT * (a + MAGIC_MULT * b)

    if binarized_features or binarized_indexes:
        raise NotImplementedError("")

    result = np.zeros(doc_count, dtype=np.uint64)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="overflow encountered in ulong_scalars")
        for feature_idx in transposed_cat_feature_indexes:
            for i in range(doc_count):
                feat = np.uint64(hashed_cat_features[feature_idx * doc_count + i])
                feat = np.uint64(np.int32(feat))
                result[i] = _calc_hash(result[i], feat)
    return result.tolist()


def calc_ctr_features(*, converted_df, ctr_data, used_model_ctrs, cat_features_info):
    hashed_cat_features = []
    for cat_feature in cat_features_info:
        hashed_cat_features.extend(converted_df.iloc[:, cat_feature.position.flat_index].tolist())
    doc_count = len(converted_df)

    ctr_feature_values = []
    for model_ctr in used_model_ctrs:
        transposed_cat_feature_indexes = model_ctr.ctr.base.projection.cat_features
        # binarized_indexes will remain empty, as long as we only include a single categorical
        # feature in each CTR
        binarized_indexes = []
        # Pass None to binarized_features, since we don't yet support computing CTRs from
        # non-categorical feature(s)
        ctr_hashes = _calc_ctr_hashes(binarized_features=None,
                                      hashed_cat_features=hashed_cat_features,
                                      transposed_cat_feature_indexes=transposed_cat_feature_indexes,
                                      binarized_indexes=binarized_indexes,
                                      doc_count=doc_count)
        assert len(ctr_hashes) == doc_count
        ctr_value_table = ctr_data.learn_ctrs[model_ctr.ctr.base]
        hash_index_resolver = TDenseIndexHashView(hash_mask=len(ctr_value_table.index_buckets) - 1,
                                                  buckets=ctr_value_table.index_buckets)
        for e in ctr_hashes:
            ptr_bucket = hash_index_resolver.get_index(e)
            if ptr_bucket:
                a = ctr_value_table.ctr_int_array[ptr_bucket * 2]
                b = ctr_value_table.ctr_int_array[ptr_bucket * 2 + 1]
                res = model_ctr.ctr.calc(b, a + b)
            else:
                res = model_ctr.ctr.calc(0, 0)
            ctr_feature_values.append(res)
    return ctr_feature_values


# Return numerical features, followed by target-encoded categorical features.
def calc_converted_input(*, converted_df, ctr_data, used_model_ctrs, float_features_info,
                         cat_features_info):
    converted_input = []
    for float_feature in float_features_info:
        converted_input.extend(converted_df.iloc[:, float_feature.position.flat_index].tolist())
    ctr_features = calc_ctr_features(converted_df=converted_df,
                                     ctr_data=ctr_data,
                                     used_model_ctrs=used_model_ctrs,
                                     cat_features_info=cat_features_info)
    print(f"ctr_features = {ctr_features}")
    converted_input.extend(ctr_features)
    return converted_input
