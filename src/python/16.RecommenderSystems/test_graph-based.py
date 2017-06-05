
def PersonalRank(G, alpha, root):
    rank = dict()
    rank = {x:0 for x in G.keys()}
    rank[root] = 1
    for k in range(20):
        tmp = {x:0 for x in G.keys()}
        for i, ri in G.items():
            for j, wij in ri.items():
                if j not in tmp:
                    tmp[j] = 0
                tmp[j] += 0.6 * rank[i] / (1.0 * len(ri))
                if j == root:
                    tmp[j] += 1 - alpha
        rank = tmp
    return rank

