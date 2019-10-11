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

users = ['Ben', 'Tom', 'John', 'Fred']
zip_data = zip(users, user_distribution)

plt.title(u'the distribution of users (NMF)')
plt.xlim((-1, 3))
plt.ylim((-1, 4))
for item in zip_data:
    user_name = item[0]
    data = item[1]
    plt.plot(data[0], data[1], "b*")
    plt.text(
        data[0],
        data[1],
        user_name,
        bbox=dict(facecolor='red', alpha=0.2),
    )

plt.show()
