import numpy as np
from numpy.random import randint, randn, normal
from matplotlib import pyplot
from trueskill import Rating, rate_1vs1

## random +- 1
## random -> b-search +- 1
## random trueskill
## random -> b-search trueskill
## reduce sigma -> b-search trueskill

def rank(a):
    return a.sum()

def oppsearch(idx, ks, diff=10):
    p0 = 0
    p1 = len(K)

    trueskill = isinstance(ks[0][0], Rating)

    x = ks[idx][0]
    if trueskill:
        x = x.mu #- x.sigma * 1

    while True:
        p = (p1 - p0) // 2 + p0

        xp = ks[p][0]
        if trueskill:
            xp = xp.mu #- xp.sigma * 1

        dx = xp - x

        if abs(dx) <= diff:
            if idx == p:
                return p+1

            return p

        if dx > 0:
            p1 = p
        else:
            p0 = p

def match(pair, xs, ks):
    a, b = pair
    sign = np.sign(rank(xs[a]) - rank(xs[b]))

    trueskill = isinstance(ks[0][0], Rating)

    if trueskill:
        if sign > 0:
            ks[a][0], ks[b][0] = rate_1vs1(ks[a][0], ks[b][0])
        elif sign < 0:
            ks[b][0], ks[a][0] = rate_1vs1(ks[b][0], ks[a][0])
        else:
            ks[b][0], ks[a][0] = rate_1vs1(ks[b][0], ks[a][0], drawn=True)

    else:
        ks[a][0] += sign
        ks[b][0] -= sign

vocabsize = 16
ntoks = 8

X = th.randint(vocabsize, (1024, ntoks))
X_ = th.randint(vocabsize, (64, ntoks))
# ■ ;;;;;

K = [[0, idx] for idx in range(len(X))]
K = [[Rating(), idx] for idx in range(len(X))]

for pair in randint(len(X), size=(1024, 2)):
    match(pair, X, K)

best = sorted(K, key=lambda x: x[0], reverse=True)[:16]

np.mean([rank(x) for x in X[array([k[1] for k in best])].unbind()])

# ■ ;;;;;

K = [[0, idx] for idx in range(len(X))]
K = [[Rating(), idx] for idx in range(len(X))]

trueskill = isinstance(K[0][0], Rating)

diff = 1
for a in randint(len(X), size=(128)):
    b = oppsearch(a, K, diff=diff)

    match([a, b], X, K)

    K = sorted(K, key=lambda x: x[0])

    mus = [k[0] for k in K]

    if trueskill:
        mus = [k.mu for k in mus]

    diff = np.diff(mus).max() + 1


best = sorted(K, key=lambda x: x[0], reverse=True)[:16]

np.mean([rank(x) for x in X[array([k[1] for k in best])].unbind()])
