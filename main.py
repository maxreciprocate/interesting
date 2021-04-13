if 'get_ipython' in locals():
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

from imports import *
from yogpt import YOGPT

from trueskill import Rating, rate_1vs1

# ■ ;;;;;

vocabsize = 32
ntoks = 4

xs = np.ones(vocabsize, dtype=bool)
for i in range(2, vocabsize // 2):
    for j in range(i+i, vocabsize, i):
        xs[j] = False

primes = from_numpy(np.where(xs)[0][2:])

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
    return ts.sum()
    # count = (ts[:,None] - primes) == 0
    # return th.sum(th.any(count, dim=0))
    # return th.all(th.diff(ts[mask]) == 0, dim=-1).float()

def match(a, b, X, K, use_rm=False, trueskill=True):
    if a == b:
        return 0

    if use_rm:
        rewards = RM(th.vstack((X[1], X[2])))
        diff = -th.diff(rewards, dim=0)
        sign = th.sign(diff)
    else:
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

def test_rm(RM, xs=None):
    if xs is None:
        trand = th.random.manual_seed(100)
        xs = th.randint(vocabsize, (1000, ntoks), generator=trand)

        scores = th.hstack([score(x) for x in xs.unbind()])
        sidxs = scores.argsort()

        scores = scores[sidxs]
        left = xs[sidxs][:len(xs) // 2][:, None, :]
        right = xs[sidxs][len(xs) // 2:][:, None, :]

        xs = th.stack((right, left), dim=1).squeeze(2)

    rewards = RM(xs.view(-1, ntoks)).view(-1, 2)
    accuracy = th.mean((th.diff(rewards) < 0).float()).item()

    print(f"rm's {accuracy = :.2f}")

# ■ ;;;;;

normed = lambda xs: (xs - xs.mean()) / (xs.std() + 1e-24)

class Ranking:
    def __init__(self, *, nfat, maxtimestep, batchsize, nepisodes, trueskill=True, rewardmodel=False):
        self.X = th.zeros(nepisodes, maxtimestep+1, dtype=th.long)
        self.R = th.zeros(nepisodes, 1, dtype=th.float32)
        self.logps = th.zeros(nepisodes, maxtimestep+1, dtype=th.float32)
        self.K = [[Rating() if trueskill else 0, idx] for idx in range(self.X.shape[0])]
        self.comparisons = 0
        self.Comps = None
        self.trueskill = trueskill
        self.rewardmodel = rewardmodel

        # inverse batchsize
        self.nfat = nfat
        self.idx = 0
        self.nepisodes = nepisodes
        self.batchsize = batchsize
        self.full = False

    def add(self, X, logps):
        newidx = self.idx + self.nfat

        self.X[self.idx:newidx] = X
        self.logps[self.idx:newidx] = logps

        self.K[self.idx:newidx] = [[Rating() if self.trueskill else 0, idx] for idx in range(self.idx, newidx)]

        idx_ = range(self.idx, newidx)
        # maybe reduce sigma here instead?
        _idx = th.randint(self.nepisodes if self.full else newidx, (self.nfat,))
        comps = []

        # how about sieving RM only through K-table?
        # thus limiting influence of it's inevitable degeneracy
        # but even then RM / real samples ratio will be too high

        for aidx, bidx in zip(idx_, _idx):

            use_rm = False
            if self.comparisons > 1000:
                use_rm = rand() < 0.75

            sign = match(aidx, bidx, self.X, self.K, use_rm=use_rm, trueskill=self.trueskill)

            if not use_rm:
                self.comparisons += 1

                if sign > 0:
                    comps.append(th.vstack((self.X[aidx], self.X[bidx]))[None,:])
                elif sign < 0:
                    comps.append(th.vstack((self.X[bidx], self.X[aidx]))[None,:])

            # take this as truth
            if self.trueskill:
                self.R[aidx] = self.K[aidx][0].mu - 3 * self.K[aidx][0].sigma
                self.R[bidx] = self.K[bidx][0].mu - 3 * self.K[bidx][0].sigma
            else:
                self.R[aidx] = self.K[aidx][0]
                self.R[bidx] = self.K[bidx][0]

        if len(comps) > 0:
            if self.Comps is None:
                self.Comps = th.vstack(comps)
            else:
                # add nonce to differentiate old ones?
                self.Comps = th.vstack((self.Comps, th.vstack(comps)))

        RM.train()

        bsize = 128
        bsize = min(bsize, len(self.Comps))

        for _ in range(4):
            idxs = randint(len(self.Comps), size=(bsize,))
            batch = self.Comps[idxs].view(-1, ntoks)
            rewards = RM(batch)
            rr = rewards.view(bsize, 2)

            RMopt.zero_grad()
            diffrr = th.diff(rr)
            loss = th.mean(th.log(th.sigmoid(diffrr) + 1e-24))

            loss.backward()
            RMopt.step()

        RM.eval()

        self.idx = newidx % self.nepisodes

        # overflow, needed to limit empty self.K[]
        if not self.full and newidx > self.idx:
            self.full = True

    def sample(self):
        if self.full:
            idxs = th.randint(self.nepisodes, (self.batchsize,))
        else:
            idxs = th.randint(0, self.idx, size=(self.batchsize,))

        return (
            self.X[idxs]
          , self.R[idxs]
          , self.logps[idxs]
        )

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

class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.master = YOGPT(vocabsize=vocabsize, nheads=4, nembd=32, ntoks=ntoks, nlayers=4, pdrop=0.1)
        self.tail = nn.Linear(vocabsize * ntoks, 1)

    def forward(self, x):
        return self.tail(F.relu(self.master(x).view(-1, vocabsize * ntoks)))

RM = RewardModel()
RMopt = th.optim.Adam(RM.parameters(), 1e-3)

PI = YOGPT(vocabsize=vocabsize, nheads=4, nembd=32, ntoks=ntoks, nlayers=4, pdrop=0.1)
PIopt = th.optim.Adam(PI.parameters())

VM = RewardModel()
VMopt = th.optim.Adam(VM.parameters())

buffer = Ranking(nfat=nfat, nepisodes=256, maxtimestep=env.ntoks-1, batchsize=64, trueskill=True, rewardmodel=False)

nepisodes = 300
epsilon = 0.1

RngExplore = np.random.RandomState(777)
tbar = tqdm(range(nepisodes))

for iepisode in tbar:
    o = env.reset()

    oldlogp = []

    for t in range(env.ntoks):
        with th.no_grad():
            logits = PI(o[..., :t+1])[:, -1, :]

            dist = Categorical(logits=logits)
            m = dist.sample()

            oldlogp.append(dist.log_prob(m))

            o_, r, done, _ = env.step(m)

            if done:
                break

            o = o_

    oldlogp = th.hstack((th.zeros(nfat, 1), th.vstack(oldlogp).T))

    buffer.add(o, oldlogp)

    X, R, oldlogp = buffer.sample()

    # V(. . .) = G
    # Q(. . .) = G ?

    R = (R - R.mean()) / R.std()

    # V = VM(X)
    # VMopt.zero_grad()
    # VMloss = (V - R).pow(2).mean()
    # VMloss.backward()
    # VMopt.step()

    logps = F.log_softmax(PI(X[..., :-1]), -1)

    logp = logps.gather(-1, X[..., 1:, None]).squeeze(-1)

    clipratio = 0.25
    ratio = th.exp(logp - oldlogp[:, 1:].detach())
    clipped = th.clamp(ratio, 1-clipratio, 1+clipratio)
    PIloss = -th.min(clipped * R, ratio * R).mean()

    pikl = (oldlogp[:, 1:] - logp).mean()

    PIopt.zero_grad()
    PIloss.backward()
    nn.utils.clip_grad_norm_(PI.parameters(), 1)
    PIopt.step()

    ps = th.exp(logps)
    entropy = -(ps * logps).mean()

    tbar.set_description(f'loss={PIloss.item():.2f}, KL={pikl.item():.2f}, H={entropy:.2f}, N={buffer.comparisons}')
        # master.load_state_dict(model.state_dict())

print(f'total {buffer.comparisons}/{nfat*nepisodes} ({buffer.comparisons / (nfat*nepisodes):.2f}) comparisons')


PI.eval()
RM.eval()

test_rm(RM, buffer.Comps)
test_rm(RM)

stanzas = []
totalscore = 0

with th.no_grad():
    for context in range(vocabsize):
        o = tensor([context])

        while len(o) < env.ntoks:
            m = PI(o[None])[:, -1, :]
            m = Categorical(F.softmax(m, -1)).sample()

            o = th.hstack((o, m))

        stanzas.append(o)

for stanza in stanzas:
    totalscore += score(stanza)
    print(f'{score(stanza)}r, {RM(stanza[:,None]).item():.0f}R, {stanza=}, ')

totalscore
# ■ ;;;;;
