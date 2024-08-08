"""globla utils
"""
def sum_dicts(dicts):
    result = {}
    for d in dicts:
        for key, value in d.items():
            result[key] = result.get(key, 0) + value
    return result

def diff_dicts(d1,d2):
    result = {}
    for key, value in d1.items():
        result[key] = abs(value - d2[key])
    return result
    
    