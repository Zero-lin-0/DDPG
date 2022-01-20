import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
import matplotlib.pyplot as plt

# 全局参数
MAX_EPISODENUMS = 500     # 训练回合数
MAX_STEPNUMS = 100          # 每一回合最大步数
LR_ACTOR = 1e-4            # policy网络(actor)的学习率
LR_CRITIC = 1e-3           # value网络(critic)的学习率
GAMMA = 0.99                 # reward discount

MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
TAO = 0.001              # 目标网络的软更新参数

# RENDER = True
RENDER = False
# GAME_NAME = 'Gopher-v0'
GAME_NAME = 'Pendulum-v0'


class ReplayBuffer(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.memory = np.zeros((capacity, dims), dtype=np.float32)
        self.counter = 0

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, action, [reward], next_state))
        index = self.counter % self.capacity
        self.memory[index, :] = transition
        self.counter += 1

    def sample(self, num):
        indices = np.random.choice(self.capacity, size=num)
        return self.memory[indices, :]


class ActorNet(nn.Module):
    def __init__(self, s_dim, a_dim, a_bound):
        super(ActorNet, self).__init__()
        self.state_dim = s_dim
        self.action_dim = a_dim
        self.action_bound = torch.tensor(a_bound)

        # 定义网络结构
        weightNum = 50
        self.hidden = nn.Linear(self.state_dim, weightNum)
        self.output = nn.Linear(weightNum, self.action_dim)

    def forward(self, state):
        action = F.relu(self.hidden(state))
        action = F.relu(self.output(action))
        action = self.action_bound * action
        return action


class CriticNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CriticNet, self).__init__()
        self.state_dim = s_dim
        self.action_dim = a_dim

        weightNum = 50
        self.hidden = nn.Linear(state_dim + action_dim, weightNum)
        self.output = nn.Linear(weightNum, 1)

    def forward(self, state, action):
        q = F.relu(self.hidden(torch.cat([state, action], 1)))
        q = self.output(q)
        return q


# DDPG网络
class DDPG(object):
    def __init__(self, s_dim, a_dim, a_bound,):
        self.memory = ReplayBuffer(MEMORY_CAPACITY, s_dim * 2 + a_dim + 1)
        self.state_dim = s_dim
        self.action_dim = a_dim
        self.action_bound = a_bound

        self.actor = ActorNet(self.state_dim, self.action_dim, self.action_bound)
        self.actor_target = ActorNet(self.state_dim, self.action_dim, self.action_bound)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = CriticNet(self.state_dim, self.action_dim)
        self.critic_target = CriticNet(self.state_dim, self.action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.criticOptimizer = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.criticLossFunc = nn.MSELoss()

    def choose_action(self, s):
        state = torch.FloatTensor(s)
        action = self.actor(state)
        return action.detach().numpy()

    def learn(self):
        # 从经验池采样
        sample_data = self.memory.sample(BATCH_SIZE)
        sample_state = torch.FloatTensor(sample_data[:, : self.state_dim])
        sample_action = torch.FloatTensor(sample_data[:, self.state_dim : self.state_dim + self.action_dim])
        sample_reward = torch.FloatTensor(sample_data[:, -self.state_dim - 1 : -self.state_dim])
        sample_nextState = torch.FloatTensor(sample_data[:, -self.state_dim : ])

        # 计算q值和目标q值等
        qvalue = self.critic(sample_state, sample_action)
        next_action = self.actor_target(sample_nextState)
        qtarget = self.critic_target(sample_nextState, next_action)
        tdTarget = sample_reward + GAMMA * qtarget

        # 更新critic
        criticLoss = self.criticLossFunc(tdTarget, qvalue)
        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        self.criticOptimizer.step()

        # 更新actor
        actorLoss = -self.critic(sample_state, self.actor(sample_state)).mean()
        # actorLoss = -torch.mean(qvalue)
        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        self.actorOptimizer.step()

        # 更新actor_target 和 critic_target
        tao = TAO   # 软更新参数

        actorTarget_layers = self.actor_target.named_children()
        for actTarLayer in actorTarget_layers:
            actTarLayer[1].weight.data.mul_((1 - tao))
            actTarLayer[1].weight.data.add_(tao * self.actor.state_dict()[actTarLayer[0] + '.weight'])
            actTarLayer[1].bias.data.mul_((1 - tao))
            actTarLayer[1].bias.data.add_(tao * self.actor.state_dict()[actTarLayer[0] + '.bias'])

        criticTarget_layers = self.critic_target.named_children()
        for criTarLayer in criticTarget_layers:
            criTarLayer[1].weight.data.mul_((1 - tao))
            criTarLayer[1].weight.data.add_(tao * self.critic.state_dict()[criTarLayer[0] + '.weight'])
            criTarLayer[1].bias.data.mul_((1 - tao))
            criTarLayer[1].bias.data.add_(tao * self.critic.state_dict()[criTarLayer[0] + '.bias'])


# 训练
env = gym.make(GAME_NAME)
env = env.unwrapped
env.seed(1)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high

myddpg = DDPG(state_dim, action_dim, action_bound)
var = 3.0

starttime = time.time()
rewards = []

for eps in range(MAX_EPISODENUMS):
    state = env.reset()
    eposide_reward = 0
    for step in range(MAX_STEPNUMS):
        if RENDER:
            env.render()

        action = myddpg.choose_action(state)
        action = np.clip(np.random.normal(action, var), -2, 2)
        next_state, reward, done_flag, info = env.step(action)
        myddpg.memory.store_transition(state, action, reward, next_state)

        if myddpg.memory.counter > MEMORY_CAPACITY:
            var *= 0.9995
            myddpg.learn()

        state = next_state
        eposide_reward += reward

        if step == MAX_STEPNUMS-1:
            rewards.append(eposide_reward)
            print("EPISODE: {}; reward: {}; Explore: {:.2f}".format(eps, int(eposide_reward), var))

plt.plot(np.arange(0, MAX_EPISODENUMS, 1), rewards, color='blue', label='reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.show()

print("running time: {}".format(time.time()-starttime))


