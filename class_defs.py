from dataclasses import dataclass
from typing import List, Optional, Dict
from hashes import _multi_hash, _vec_hash

@dataclass
class TFloatSplit:
    float_feature: int
    split: float

    def __hash__(self):
        return _multi_hash(hash(self.float_feature), hash(self.split))

@dataclass
class TOneHotSplit:
    cat_feature_idx: int
    value: int

    def __hash__(self):
        return _multi_hash(hash(self.cat_feature_idx), hash(self.value))

@dataclass
class TFeatureCombination:
    cat_features: List[int]
    bin_features: List[TFloatSplit]
    onehot_features: List[TOneHotSplit]

    def __hash__(self):
        return _multi_hash(_vec_hash(self.cat_features), _vec_hash(self.bin_features),
                           _vec_hash(self.onehot_features))

@dataclass
class TModelCtrBase:
    projection: TFeatureCombination
    ctr_type: str
    target_border_classifier_idx: int = 0  # appears unused

    def __hash__(self):
        # Using string ctr_type, instead of enum, for the sake of simplicity
        # Catboost stores ctr_type as int enum
        return _multi_hash(hash(self.projection), hash(self.ctr_type),
                           self.target_border_classifier_idx)

@dataclass
class TModelCtr:
    base: TModelCtrBase
    target_border_idx: int
    prior_num: float
    prior_denom: float
    shift: float
    scale: float

    def calc(self, count_in_class: float, total_count: float):
        ctr = (count_in_class + self.prior_num) / (total_count + self.prior_denom)
        return (ctr + self.shift) * self.scale

@dataclass
class TBucket:
    hash: Optional[int] = None
    index_value: Optional[int] = None

@dataclass
class TDenseIndexHashBuilder:
    hash_mask: int
    buckets: List[TBucket]
    bin_count: int

    def add_index(self, hash: int):
        zz = hash & self.hash_mask
        while self.buckets[zz].hash is not None:
            if self.buckets[zz].hash == hash:
                return self.buckets[zz].index_value
            zz = (zz + 1) & self.hash_mask
        self.buckets[zz].hash = hash
        self.buckets[zz].index_value = self.bin_count
        self.bin_count += 1
        return self.bin_count - 1

@dataclass
class TDenseIndexHashView:
    hash_mask: int
    buckets: List[TBucket]

    def get_index(self, hash: int):
        zz = hash & self.hash_mask
        while self.buckets[zz].hash is not None:
            if self.buckets[zz].hash == hash:
                return self.buckets[zz].index_value
            zz = (zz + 1) & self.hash_mask
        return None  # not found

@dataclass
class TCtrValueTable:
    model_ctr_base: TModelCtrBase
    index_buckets: List[TBucket]
    ctr_int_array: List[int]
    target_classes_count: int

@dataclass
class TCtrData:
    learn_ctrs: Dict[TModelCtrBase, TCtrValueTable]

@dataclass
class TCtrFeature:
    ctr: TModelCtr
    borders: List[float]

@dataclass
class TFeaturePosition:
    index: int = -1
    flat_index: int = -1

@dataclass
class TCatFeature:
    position: TFeaturePosition
