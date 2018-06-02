#!/usr/bin/python
# coding:utf8

import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

RATE_MATRIX = np.array([[5, 5, 3, 0, 5, 5], [5, 0, 4, 0, 4, 4],
                        [0, 3, 0, 5, 4, 5], [5, 4, 3, 3, 5, 5]])

nmf = NMF(n_components=2)
user_distribution = nmf.fit_transform(RATE_MATRIX)
item_distribution = nmf.components_

item_distribution = item_distribution.T
plt.plot(item_distribution[:, 0], item_distribution[:, 1], "b*")
plt.xlim((-1, 3))
plt.ylim((-1, 3))

plt.title(u'the distribution of items (NMF)')
count = 1
for item in item_distribution:
    plt.text(
        item[0],
        item[1],
        'item ' + str(count),
        bbox=dict(facecolor='red', alpha=0.2),
    )
    count += 1

plt.show()
