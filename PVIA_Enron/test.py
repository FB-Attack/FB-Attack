from collections import Counter
def inter_list(a, b):
    a = dict(Counter(a))
    b = dict(Counter(b))
    count = 0
    for key1, val1 in a.items():
        if key1 in b.keys():
            count += min(val1, b[key1])
    return count
la = [1, 2, 3, 4, 5, 5, 6]
lb = [2, 3, 4, 5, 5]
cc = inter_list(la, lb)
print(cc)