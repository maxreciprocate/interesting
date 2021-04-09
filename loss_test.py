from imports import *

model = nn.Sequential(
    nn.Linear(1, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
)

opt = th.optim.Adam(model.parameters())
X = th.randint(1000, size=(10000,)).float()

pairs = []

for _ in range(10000):
    a, b = randint(len(X), size=(2,))

    if X[a] - X[b] > 0:
        pairs.append(th.vstack((X[a], X[b]))[None])
    else:
        pairs.append(th.vstack((X[b], X[a]))[None])

pairs = th.vstack(pairs)

# ■ ;;;;;

tbar = tqdm(range(1000))
for _ in tbar:
    idxs = randint(len(X), size=(256,))
    batch = pairs[idxs]

    batch = batch.view(-1 , 1)
    rewards = model(batch)
    rewards = rewards.view(-1, 2)

    opt.zero_grad()

    loss = th.mean(th.log_softmax(rewards, -1)[:, 1])
    loss = th.mean(th.log(th.sigmoid(-th.diff(rewards)) + 1e-24))

    loss.backward()
    opt.step()
    tbar.set_description(f"{loss.item()=:.2f}")

rewards = model(th.arange(1000).float()[:, None])

th.diff(rewards.squeeze())
