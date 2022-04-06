from typing import List
import numpy as np
import numpy.typing as npt
from class_defs import TFloatFeature, TCtrFeature, ObliviousTree


def load_oblivious_trees(
        model,
        *,
        float_features_info: List[TFloatFeature],
        used_model_ctrs: List[TCtrFeature]
) -> List[ObliviousTree]:
    if "trees" in model:
        raise NotImplementedError("")
    assert "oblivious_trees" in model
    border_offsets = [0]
    for float_feature in float_features_info:
        border_offsets.append(border_offsets[-1] + len(float_feature.borders))
    for ctr_feature in used_model_ctrs:
        border_offsets.append(border_offsets[-1] + len(ctr_feature.borders))

    trees = []
    for tree_json in model["oblivious_trees"]:
        feature_id = []
        threshold = []
        for split_json in tree_json["splits"]:
            split_index = split_json["split_index"]
            for fid in range(len(border_offsets) - 1):
                if border_offsets[fid] <= split_index < border_offsets[fid + 1]:
                    feature_id.append(fid)
                    break
            threshold.append(split_json["border"])
        trees.append(ObliviousTree(depth=len(threshold),
                                   feature_id=feature_id,
                                   threshold=threshold))
    return trees


def predict_leaf(
        trees: List[ObliviousTree],
        *,
        converted_input: npt.NDArray[np.float64],
        doc_count: int
) -> npt.NDArray[np.int32]:
    assert converted_input.shape[0] == doc_count
    leaf_out = np.zeros((doc_count, len(trees)), dtype=np.int32)
    for doc_id in range(doc_count):
        for tree_id, tree in enumerate(trees):
            leaf_idx = 0
            for depth, (feature_id, threshold) in enumerate(zip(tree.feature_id, tree.threshold)):
                res = 1 if converted_input[doc_id, feature_id] > threshold else 0
                leaf_idx |= (res << depth)
            leaf_out[doc_id, tree_id] = leaf_idx
    return leaf_out
