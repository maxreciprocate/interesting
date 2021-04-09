if 'get_ipython' in locals():
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

from imports import *
from yogpt import YOGPT

from trueskill import Rating, rate_1vs1

# ■ ;;;;;

vocabsize = 32
ntoks = 16

xs = np.ones(vocabsize, dtype=bool)
for i in range(2, vocabsize // 2):
    for j in range(i+i, vocabsize, i):
        xs[j] = False

primes = from_numpy(np.where(xs)[0][2:])

# ■ ;;;;;

rm = lambda x, y: x + y

import sympy
x = sympy.symbols('x')
xx = sympy.log(1 / (1 + sympy.exp(-x)))

xs = np.linspace(-10, 10)
pyplot.plot(xs, ff(xs));
ff = lambda x: np.log(1 / (1 + np.exp(rm(0, x) - rm(0, 1))))

# ■ ;;;;;

def oppsearch(idx, K, diff=10):
    p0 = 0
    p1 = len(K)
    x = K[idx][0].mu

    while True:
        p = (p1 - p0) // 2 + p0
        dx = K[p][0].mu - x

        if abs(dx) <= diff:
            if idx == p:
                return p+1

            return p

        if dx > 0:
            p1 = p
        else:
            p0 = p

def score(ts):
    # return ts.sum()
    count = (ts[:,None] - primes) == 0
    return th.sum(th.any(count, dim=0))
    # return th.all(th.diff(ts[mask]) == 0, dim=-1).float()

def match(a, b, X, K, trueskill=True):
    if a == b:
        return 0

    sign = th.sign(score(X[K[a][1]]) - score(X[K[b][1]]))

    if not trueskill:
        K[a][0] += sign
        K[b][0] -= sign
        return sign

    if sign > 0:
        K[a][0], K[b][0] = rate_1vs1(K[a][0], K[b][0])
    elif sign < 0:
        K[b][0], K[a][0] = rate_1vs1(K[b][0], K[a][0])
    else:
        K[b][0], K[a][0] = rate_1vs1(K[b][0], K[a][0], drawn=True)

    return sign

X = th.randint(vocabsize, (10000, ntoks))
# K = [[Rating(), idx] for idx in range(X.shape[0])]
K = [[0, idx] for idx in range(X.shape[0])]
# K = sorted(K, key=lambda x: x[0].mu)

# reduce sigma
# sigmas = [k[0].sigma for k in K]

# idxs = np.arange(len(sigmas))

# ps = np.exp(sigmas) / (np.sum(np.exp(sigmas)) + 1)
# ps /= ps.sum()

class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.master = YOGPT(vocabsize=vocabsize, nheads=1, nembd=64, ntoks=ntoks, nlayers=8)
        self.tail = nn.Linear(vocabsize * ntoks, 1)

    def forward(self, x):
        return self.tail(F.relu(self.master(x).view(-1, vocabsize * ntoks)))

# ■ ;;;;;

RM = RewardModel()
opt = th.optim.Adam(RM.parameters())

diff =  1
xs = []

for a, b in randint(len(X), size=(1000, 2)):
    if score(X[a]) > score(X[b]):
        xs.append((th.vstack((X[a], X[b])))[None])
    elif score(X[a]) < score(X[b]):
        xs.append((th.vstack((X[b], X[a])))[None])

# for kaopp in np.random.choice(np.arange(len(X)), size=10000):
#     kbopp = oppsearch(kaopp, K, diff)
#     sign = match(kaopp, kbopp, X, K, trueskill=True)

#     if sign > 0:
#         xs.append((th.vstack((X[kaopp], X[kbopp])))[None])
#     elif sign < 0:
#         xs.append((th.vstack((X[kbopp], X[kaopp])))[None])

#     K = sorted(K, key=lambda x: x[0].mu)
#     mus = [k[0].mu for k in K]
#     diff = np.diff(mus).max() + 1

comparisons = th.vstack(xs)

# yys = th.vstack(yys) + 1
# xxs = th.vstack(xxs)
# ■ ;;;;;
bsize = 64

tbar = tqdm(range(1024))
for _ in tbar:
    idxs = randint(len(xs), size=(bsize,))
    batch = comparisons[idxs].view(-1, ntoks)
    rewards = RM(batch)
    rr = rewards.view(bsize, 2)

    opt.zero_grad()
    # loss = th.mean(th.log_softmax(rr, -1)[:, 0])
    insigma = th.diff(rr)
    loss = th.mean(th.log(th.sigmoid(insigma) + 1e-24))

    loss.backward()
    opt.step()
    tbar.set_description(f'{loss.item()=:.2f}')

# ■ ;;;;;

RM.eval()
sidxs = np.argsort([score(x) for x in X.unbind()])
scores = [score(x).item() for x in X[sidxs].unbind()]
scores[-10:]
sX = X[sidxs]

rewards = RM(sX)
rewards
diff = th.diff(rewards.squeeze())
sdiff = np.diff(scores)
(sdiff > 0).mean()
(diff > 0).float().mean()
diff.mean()
rewards[-1] - rewards[0]

# ■ ;;;;;

RM.eval()

class Ranking:
    def __init__(self, *, nfat, maxtimestep, batchsize, nepisodes, trueskill=True, rewardmodel=False):
        self.X = th.zeros(nepisodes, maxtimestep+1, dtype=th.long)
        self.R = th.zeros(nepisodes, 1, dtype=th.float32)
        self.K = [[Rating() if trueskill else 0, idx] for idx in range(self.X.shape[0])]
        self.trueskill = trueskill
        self.rewardmodel = rewardmodel

        # inverse batchsize
        self.nfat = nfat
        self.idx = 0
        self.nepisodes = nepisodes
        self.batchsize = batchsize
        self.full = False

    def add(self, X):
        newidx = self.idx + self.nfat

        self.X[self.idx:newidx] = X

        if not self.rewardmodel:
            self.K[self.idx:newidx] = [[Rating() if self.trueskill else 0, idx] for idx in range(self.idx, newidx)]

        # matchmaking goes here
            idx_ = range(self.idx, newidx)
            _idx = th.randint(self.nepisodes if self.full else newidx, (self.nfat,))

            for aidx, bidx in zip(idx_, _idx):
                sign = match(aidx, bidx, self.X, self.K, trueskill=self.trueskill)

                if self.trueskill:
                    self.R[aidx] = self.K[aidx][0].mu - 3 * self.K[aidx][0].sigma
                    self.R[bidx] = self.K[bidx][0].mu - 3 * self.K[bidx][0].sigma
                else:
                    self.R[aidx] = self.K[aidx][0]
                    self.R[bidx] = self.K[bidx][0]

        if self.rewardmodel:
            with th.no_grad():
                self.R[self.idx:newidx] = RM(X)


        self.idx = newidx % self.nepisodes

        # overflow, needed to limit empty self.K[]
        if not self.full and newidx > self.idx:
            self.full = True

    def sample(self):
        idxs = th.randint(0, self.nepisodes if self.full else self.idx, (self.batchsize,))

        return (
            self.X[idxs]
          , self.R[idxs]
        )

# vocabsize = 16
# ntoks = 8
model = YOGPT(vocabsize=vocabsize, nheads=1, nembd=64, ntoks=ntoks, nlayers=8)

class CompetitivePrimeEnv:
    def __init__(self, *, ntoks, nfat):
        self.ntoks = ntoks
        self.nfat = nfat
        self.idx = 1
        self.state = th.zeros(self.nfat, self.ntoks, dtype=th.long)

    def step(self, move):
        self.state[:, self.idx] = move
        self.idx += 1

        done = False
        if self.idx == self.ntoks:
            done = True

        return self.state, 0, done, None

    def reset(self):
        self.idx = 1
        self.state.fill_(0)
        self.state[:, 0] = th.randint(vocabsize, (self.nfat,))

        return self.state


nfat = 32
env = CompetitivePrimeEnv(ntoks=ntoks, nfat=nfat)
nin = env.ntoks
nout = 1
maxtimestep = env.ntoks-1

buffer = Ranking(nfat=nfat, nepisodes=256, maxtimestep=env.ntoks-1, batchsize=64, trueskill=True, rewardmodel=False)
opt = th.optim.Adam(model.parameters())

nepisodes = 100
epsilon = 0.1

tbar = tqdm(range(nepisodes))
for iepisode in tbar:
    o = env.reset()

    for t in range(env.ntoks * 100):
        with th.no_grad():
            if rand() < epsilon:
                m = th.randint(vocabsize, size=(nfat,))
            else:
                logits = model(o[..., :t+1])[:, -1, :]

                dist = Categorical(logits=logits)
                m = dist.sample()
                dist.log_prob(m)

            o_, r, done, _ = env.step(m)

            if done:
                break

            o = o_

    buffer.add(o)

    X, R = buffer.sample()

    R = (R - R.mean()) / R.std()

    logp = F.log_softmax(model(X[..., :-1]), -1).gather(-1, X[..., 1:, None]).squeeze(-1)
    loss = -(R * logp).mean()

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1)
    opt.step()

    if not(iepisode % 10):
        tbar.set_description(f'loss = {loss.item():.2f}')
        # master.load_state_dict(model.state_dict())

print(f'total {nfat*nepisodes} comparisons')

model.eval()
stanzas = []

with th.no_grad():
    for context in range(vocabsize):
        o = tensor([context])

        while len(o) < env.ntoks:
            m = model(o[None])[:, -1, :]
            m = Categorical(F.softmax(m, -1)).sample()

            o = th.hstack((o, m))

        stanzas.append(o)

for stanza in stanzas:
    print(f'{score(stanza)}r, {RM(stanza[:,None]).item():.0f}R, {stanza=}, ')
