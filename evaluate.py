import numpy as np
from class_defs import ObliviousTree


def load_oblivious_trees(model, *, float_features_info, used_model_ctrs):
    if "trees" in model:
        raise NotImplementedError("")
    assert "oblivious_trees" in model
    border_offsets = [0]
    for float_feature in float_features_info:
        border_offsets.append(border_offsets[-1] + len(float_feature.borders))
    for ctr_feature in used_model_ctrs:
        border_offsets.append(border_offsets[-1] + len(ctr_feature.borders))
    print(border_offsets)

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


def predict_leaf(trees, *, converted_input, doc_count):
    assert len(converted_input) % doc_count == 0
    leaf_out = []
    for doc_id in range(doc_count):
        for tree in trees:
            leaf_idx = 0
            for depth, (feature_id, threshold) in enumerate(zip(tree.feature_id, tree.threshold)):
                res = 1 if converted_input[feature_id * doc_count + doc_id] > threshold else 0
                leaf_idx |= (res << depth)
            leaf_out.append(leaf_idx)
    leaf_out = np.array(leaf_out, dtype=np.int32).reshape((doc_count, -1))
    return leaf_out
