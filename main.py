if 'get_ipython' in locals():
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

from imports import *
from yogpt import YOGPT
from trueskill import Rating, rate_1vs1

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

def score(context):
    if test_task == "kink":
        return th.sum(context == vocabsize - 1)
    if test_task == "maxsum":
        return context.sum()
    elif test_task == "unique":
        repeats = th.diff(th.sort(context)[0]) == 0
        count = (context[:,None] - primes) == 0
        return th.sum(th.any(count, dim=0)) #- 0.3 * sum(repeats)
    elif test_task == "sort":
        return th.all(context == th.sort(context)[0]).float()

    raise ValueError('weird test_task')
    # return th.all(th.diff(context[mask]) == 0, dim=-1).float()

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
        # lenient testing
        left = xs[sidxs][:len(xs) // 2][:, None, :]
        right = xs[sidxs][len(xs) // 2:][:, None, :]

        xs = th.stack((right, left), dim=1).squeeze(2)

    rewards = RM(xs.view(-1, ntoks)).view(-1, 2)
    accuracy = th.mean((th.diff(rewards) < 0).float())

    return accuracy.item()

# test whether policy agrees with comparisons collected in buffer
# i.e. logprob of winner should be greater
def test_pi(PI, xs):
    xs = xs.view(-1, ntoks)
    logits = PI(xs)
    logps = F.log_softmax(logits, -1)[..., :-1, :]

    actions = xs[..., 1:]
    logp = logps.gather(-1, actions.unsqueeze(1)).squeeze(1)

    logpairs = logp.sum(-1).view(-1, 2)
    compscore = th.mean((th.diff(logpairs) < 0).float())
    return compscore.item()

def sample(ps):
    ps /= ps.sum()
    cdf = ps.cumsum(-1)
    x = rand()
    for i in range(len(ps)):
        if cdf[i] > x:
            return i

    return len(ps)-1

def nucleus_sample(ps, p=0.9):
    values, indices = ps.sort(descending=True)
    totalp = 0

    for idx in range(len(values)):
        totalp += values[idx]
        if totalp > p:
            return indices[sample(values[:idx+1])]

def unique_star(context):
    ps = th.zeros(vocabsize) + 1e-24
    ps[primes] = 1

    for x in context:
        ps[x] = 0

    return ps / ps.sum()

def maxsum_star(_):
    ps = th.zeros(vocabsize) + 1e-24
    ps[-1] = 1
    return ps

def roll(context):
    print(f"{context} | _")

    with th.no_grad():
        context = th.zeros(1).long()
        while len(context) < env.ntoks:
            logits = PI(context[None])[:, -1, :]
            ps = F.softmax(logits, -1).squeeze(0).detach()

            ps_star = unique_star(context)
            kl = ps_star @ log(ps_star / ps)

            colors = ['slateblue'] * len(ps)
            if test_task == 'unique':
                for idx in primes:
                    if idx not in context:
                        colors[idx] = 'orange'

            elif test_task == 'maxsum':
                colors[-1] = 'orange'

            pyplot.bar(range(len(ps)), ps, color=colors);
            pyplot.xticks(range(len(ps)));
            pyplot.show()

            m = nucleus_sample(ps, p=0.6)

            print(f"{context} | {m} {kl=}")
            context = th.hstack((context, m))

        print(f"{context} ~ {RM(context[None]).item():.2f}R")

    return context

# ■ ;;;;;

def log(xs):
    return th.log(xs + 1e-24)

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

        # batch(idx_, _idx) -> comps
        # update K, R

        comps = []

        # how about sieving RM only through K-table?
        # thus limiting influence of it's inevitable degeneracy
        # but even then RM / real samples ratio will be too high

        for aidx, bidx in zip(idx_, _idx):

            use_rm = False
            # if self.comparisons > 5000:
            #     use_rm = rand() < 0.95

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

        # if self.comparisons < 5000 or rand() < 0.1:
        #     RM.train()

        #     bsize = 128
        #     bsize = min(bsize, len(self.Comps))

        #     for _ in range(4):
        #         idxs = randint(len(self.Comps), size=(bsize,))
        #         batch = self.Comps[idxs].view(-1, ntoks)
        #         rewards = RM(batch)
        #         rr = rewards.view(bsize, 2)

        #         RMopt.zero_grad()
        #         diffrr = th.diff(rr)
        #         loss = th.mean(log(th.sigmoid(diffrr)))

        #         loss.backward()
        #         RMopt.step()

        #     RM.eval()

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

vocabsize = 16
ntoks = 5
test_task = 'unique'

xs = np.ones(vocabsize, dtype=bool)
for i in range(2, vocabsize // 2):
    for j in range(i+i, vocabsize, i):
        xs[j] = False

primes = from_numpy(np.where(xs)[0][2:])

nfat = 16
env = CompetitivePrimeEnv(ntoks=ntoks, nfat=nfat)
nin = env.ntoks
nout = 1
maxtimestep = env.ntoks-1

class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.master = YOGPT(vocabsize=vocabsize, nheads=4, nembd=64, ntoks=ntoks, nlayers=8, pdrop=0.1)
        self.tail = nn.Linear(vocabsize * ntoks, 1)

    def forward(self, x):
        return self.tail(F.relu(self.master(x).view(-1, vocabsize * ntoks)))

class ValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt = YOGPT(vocabsize=vocabsize, nheads=4, nembd=128, ntoks=ntoks, nlayers=4, pdrop=0.1)
        self.head = nn.Linear(vocabsize, 1)

    def forward(self, x):
        return self.head(self.gpt(x))


RM = RewardModel()
RMopt = th.optim.Adam(RM.parameters(), 1e-3)

VM = ValueModel()
VMopt = th.optim.Adam(VM.parameters(), 1e-3)

PI = YOGPT(vocabsize=vocabsize, nheads=8, nembd=128, ntoks=ntoks, nlayers=8, pdrop=0.1)
PIopt = th.optim.Adam(PI.parameters())

buffer = Ranking(nfat=nfat, nepisodes=128, maxtimestep=env.ntoks-1, batchsize=64, trueskill=True, rewardmodel=False)

nepisodes = 200
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

    X, G, oldlogp = buffer.sample()

    # V(. . .) = G
    # Q(. . .) = G ?

    G = (G - G.mean()) / G.std()

    V = VM(X).squeeze(-1)

    VMopt.zero_grad()
    VMloss = (V - G).pow(2).mean()
    VMloss.backward()
    VMopt.step()

    logps = F.log_softmax(PI(X[..., :-1]), -1)
    logp = logps.gather(-1, X[..., 1:, None]).squeeze(-1)

    ps = th.exp(logps)
    entropy = -(ps * logps).sum()

    # clipratio = 0.2
    # ratio = th.exp(logp - oldlogp[:, 1:].detach())
    # clipped = th.clamp(ratio, 1-clipratio, 1+clipratio)
    # PIloss = -th.min(clipped * (G - V[:, :-1]).detach(), ratio * R).mean()
    PIloss = -th.mean(logp * (G - V[:, :-1].detach())) - 0.001 * entropy

    PIopt.zero_grad()
    PIloss.backward()
    nn.utils.clip_grad_norm_(PI.parameters(), 1)
    PIopt.step()

    pikl = (oldlogp[:, 1:] - logp).mean()

    tbar.set_description(f'loss={PIloss.item():.2f}, KL={pikl.item():.2f}, H={entropy:.2f}, N={buffer.comparisons}')

    if not(iepisode % 5):
        roll(tensor([0]))

print(f'total {buffer.comparisons}/{nfat*nepisodes} ({buffer.comparisons / (nfat*nepisodes):.2f}) comparisons')
# ■ ;;;;;

PI.eval()
RM.eval()

# would like to see here >0.9
# print(f"{test_pi(PI, buffer.Comps)=:.2f}")
# print(f"{test_rm(RM, buffer.Comps)=:.2f}")
# print(f"{test_rm(RM)=:.2f}")
# ■ ;;;;;

roll(tensor([0]))

stanzas = []
totalscore = 0

with th.no_grad():
    for context in range(vocabsize):
        o = tensor([context])
        o = tensor([context])

        while len(o) < env.ntoks:
            logits = PI(o[None])[:, -1, :]
            ps = F.softmax(logits, -1).squeeze(0).detach()

            # m = nucleus_sample(ps, p=0.2)
            m = ps.argmax()
            # m = tensor(sample(ps))

            o = th.hstack((o, m))

        stanzas.append(o)

for stanza in stanzas:
    totalscore += score(stanza)
    print(f'{score(stanza)}r, {RM(stanza[:,None]).item():.0f}R, {stanza=}, ')

print(f"total {totalscore.item()}/70")
