import random


# 负样本采样过程
def RandomSelectNegativeSample(self, items):
    ret = dict()
    for i in items.keys():
        ret[i] = 1

    n = 0
    for i in range(0, len(items) * 3):
        item = items_pool[random.randint(0, len(items_pool) - 1)]
        if item in ret:
            continue
        ret[item] = 0
        n += 1
        if n > len(items):
            break
    return ret


def LatentFactorModel(user_items, F, N, alpha, _lambda):
    [P, Q] = InitModel(user_items, F)
    for step in range(0, N):
        for user, items in user_items.items():
            samples = RandSelectNegativeSamples(items)
            for item, rui in samples.items():
                eui = rui - Predict(user, item)
                for f in range(0, F):
                    P[user][f] += alpha * (eui * Q[item][f] - _lambda * P[user][f])
                    Q[item][f] += alpha * (eui * P[user][f] - _lambda * Q[item][f])
        alpha *= 0.9


def Recommend(user, P, Q):
    rank = dict()
    for f, puf in P[user].items():
        for i, qfi in Q[f].items():
            if i not in rank:
                rank[i] += puf * qfi
    return rank


