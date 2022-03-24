from clickhouse_cityhash.cityhash import CityHash64

def _int_hash(key: int):
    key += ~(key << 32)
    key ^= (key >> 22)
    key += ~(key << 13)
    key ^= (key >> 8)
    key += (key << 3)
    key ^= (key >> 15)
    key += ~(key << 27)
    key ^= (key >> 31)
    return key

def _multi_hash(*args):
    assert len(args) >= 2
    ret = args[0]
    for x in args[1:]:
        ret = _int_hash(ret) ^ x
    return ret

def _vec_hash(vec):
    res = 1988712
    for e in vec:
        if isinstance(e, int):
            res = 984121 * res + e
        else:
            res = 984121 * res + hash(e)
    return res

def _hash_string_cat(x):
    return CityHash64(x) & 0xffffffff
