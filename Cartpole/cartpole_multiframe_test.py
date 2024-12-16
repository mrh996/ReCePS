import argparse
import gym
import numpy as np
from itertools import count
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch Cartpole test')
parser.add_argument('--seed', type=int, default=20, metavar='N',
					help='random seed (default: 20)')
parser.add_argument('--render', action='store_true',
					help='render the environment')
parser.add_argument('--sigma', type=float, default=0.0, metavar='N',
					help='How much noise to smooth observations')
parser.add_argument('--num_evals', type=int, default=10000, metavar='N',
					help='Number of evaluations')
parser.add_argument('--store_all_rewards', action='store_true',
					help='store all rewards (vs just sum)')
parser.add_argument('--checkpoint', type=str,
					help='checkpoint path')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
					help='interval between training status logs (default: 10)')
args = parser.parse_args()
class NoisyObsWrapper(gym.ObservationWrapper):
	def __init__(self, env, sigma):
		super().__init__(env)
		self.sigma = sigma
	def observation(self, obs):
		return obs + self.sigma*np.random.standard_normal(size=obs.shape)


from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.dqn.policies import MlpPolicy as DQNMlp
from stable_baselines3 import DQN
from gym.wrappers import FrameStack, FlattenObservation


if __name__ == '__main__':
	# Multiprocess environment
	sigma = args.sigma
	env =  FlattenObservation(FrameStack(NoisyObsWrapper(gym.make("CartPole-v0"), sigma), 5) )
	env.seed(args.seed)
	reward_accum = []
	state_hist = []
	policy_kwargs = {}
	model = DQN.load(args.checkpoint)
	initial_states=[]

	for i_episode in range(args.num_evals):
		state = env.reset()
		initial_states.append(state)
		ep_reward =0
		policy_rewards = []
		state_hist.append(state)
		for t in range(1, 201): 
			action, _ = model.predict(state,deterministic=True)
			state, reward, done, _ = env.step(action)
			state_hist.append(state)
			if args.render:
				env.render()
			policy_rewards.append(reward)
			ep_reward += reward
			if done:
				break
		if (args.store_all_rewards):
			reward_accum.append(policy_rewards)
		else:
			reward_accum.append(ep_reward)
		if i_episode % args.log_interval == 0:
			print('Episode {}\t'.format(
				i_episode))
	torch.save(reward_accum, args.checkpoint + str(args.num_evals) + '.pth')
	torch.save(initial_states, args.checkpoint +'initial_state'+ '.pth')
