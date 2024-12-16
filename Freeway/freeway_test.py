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
from gym.envs.registration import  register

register(
    id="FreewayHardNoFrameskip-v0",
    entry_point="gym.envs.atari:AtariEnv",
    kwargs={
        "game": "freeway",
        "obs_type": "image",
        "frameskip": 1,
        "repeat_action_probability": 0.25,
        "difficulty": 1
    },  # A frameskip of 1 means we get every frame
    max_episode_steps=4 * 100000,
    nondeterministic=False,
)


from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gym.wrappers import TimeLimit,FrameStack, AtariPreprocessing
import time
if __name__ == '__main__':
	# Multiprocess environment
	sigma = args.sigma *255.
	env =  TimeLimit(FrameStack(NoisyObsWrapper(AtariPreprocessing(gym.make("FreewayHardNoFrameskip-v0")), sigma), 4) ,250)
	

	env.seed(args.seed)
	reward_accum = []
	policy_kwargs = {}
	model = DQN.load(args.checkpoint)

	for i_episode in range(args.num_evals):
		state = env.reset()
		
		ep_reward =0
		policy_rewards = []
		t = 0
		while True:
			t += 1
			state = np.array(state)
			action, _ = model.predict(state)
			# time.sleep(0.2)
			# qs = model.q_net(torch.tensor(state).unsqueeze(0))
			# acts  = [max(qs[0,0],qs[0,1]).item(), max(qs[0,2],qs[0,4]).item(), max(qs[0,3],qs[0,5]).item()]
			# acts.sort()
			# print(round((acts[2] -acts[1]),2))
			#print(str(left) + '   ' + str(right))
			state, reward, done, _ = env.step(action)
			#print(done)
			#print(reward)
			if args.render:
				env.render()
			if (reward != 0):
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
				i_episode), flush=True)
	torch.save(reward_accum, args.checkpoint + '_evals_'+ str(args.num_evals) + '.pth')
