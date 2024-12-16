import argparse
import gym
import numpy as np
from itertools import count
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch cartpole attack')
parser.add_argument('--seed', type=int, default=20, metavar='N',
					help='random seed (default: 20)')
parser.add_argument('--render', action='store_true',
					help='render the environment')
parser.add_argument('--q_threshold', type=float, default=0., metavar='N',
					help='q value threshold')
parser.add_argument('--num_evals', type=int, default=1000, metavar='N',
					help='Number of evaluations')
parser.add_argument('--store_all_rewards', action='store_true',
					help='store all rewards (vs just sum)')
parser.add_argument('--checkpoint', type=str,
					help='checkpoint path')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
					help='interval between training status logs (default: 10)')
parser.add_argument('--attack_eps', type=float, default=0.2, metavar='N',
					help='Attack epsilon, total')
parser.add_argument('--attack_step', type=float, default=0.01, metavar='N',
					help='Attack step size')
parser.add_argument('--attack_step_count_multiplier', type=float, default=2, metavar='N',
					help='Multiplier between steps and budget/attack_step')
args = parser.parse_args()


def attack_clean(state,model,carryover_budget_squared):
	label = model.predict(state,deterministic=True)[0]
	#print(carryover_budget_squared)
	state = torch.tensor(state,device='cuda').float().unsqueeze(0)
	target_logits =  model.q_net(state)[0]
	#target = target_logits.argmin().unsqueeze(0)
	starting_action = target_logits.argmax()
	#clean_out = policy.step_for_attack(state).detach()
	ori_state = copy.deepcopy(state.data)
	state= state.detach()
	obj = torch.nn.CrossEntropyLoss()
	if (carryover_budget_squared <= 0):
		budget = 0
	else:
		budget = math.sqrt(carryover_budget_squared)
	step_count = int(args.attack_step_count_multiplier * budget/args.attack_step)
	targets = list([torch.tensor(x).cuda() for x in range(target_logits.shape[0])])
	[targets.remove(x.item()) for x in (target_logits + args.q_threshold >target_logits[starting_action]).nonzero(as_tuple=False)]
	best_q = target_logits[starting_action]
	best_state = ori_state
	for target in targets:
		state =  copy.deepcopy(ori_state.data)
		for i in range(step_count):
			state.requires_grad = True
			out = model.q_net(state)
			if (out[0].argmax() == target):
				if target_logits[target] < best_q:
					best_q = target_logits[target]
					best_state = state.detach_()
				break
			model.q_net.zero_grad()
			#cost = -out[0,label]
			cost = -obj(out,target.unsqueeze(0))
			grad, = torch.autograd.grad(inputs=state, outputs=cost)
			state = state + args.attack_step*grad/grad.norm()
			if ((state - ori_state).norm() > budget):
				state = ori_state + (state - ori_state) * budget / (state - ori_state).norm()
			state= state.detach_()

	return best_state[0].cpu(), budget**2 - (best_state - ori_state).norm()**2

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.dqn.policies import MlpPolicy as DQNMlp
from stable_baselines3 import DQN



if __name__ == '__main__':
	# Multiprocess environment
	env =  gym.make("CartPole-v0")
	#env =  NoisyObsWrapper(gym.make(cfg["environment"]), sigma) 
	env.seed(args.seed)
	reward_accum = []
	state_hist = []
	policy_kwargs = {}
	model = DQN.load(args.checkpoint)

	for i_episode in range(args.num_evals):
		state = env.reset()
		ep_reward =0
		policy_rewards = []
		state_hist.append(state)
		carryover = args.attack_eps*args.attack_eps
		for t in range(1, 201): 
			#print(carryover)
			observation,carryover = attack_clean(state,model,carryover)
			#print(state)
			#print(observation)
			action, _ = model.predict(observation,deterministic=True)
			#print(action)
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
			print(ep_reward)
			reward_accum.append(ep_reward)
		if i_episode % args.log_interval == 0:
			#print(np.array(state_hist).mean(axis=0))
			#print(np.array(state_hist).std(axis=0))
			print('Episode {}\t'.format(
				i_episode))
	torch.save(reward_accum, args.checkpoint + '_evals_'+ str(args.num_evals) + '_attack_eps_' + str(args.attack_eps) + '_attack_step_count_multiplier_' + str(args.attack_step_count_multiplier) + '_attack_step_'+ str(args.attack_step)+ '_threshold_'+ str(args.q_threshold)+ '.pth')