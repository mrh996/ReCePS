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

import time
from gym.envs.registration import  register

from gym.wrappers import TimeLimit,AtariPreprocessing
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
parser = argparse.ArgumentParser(description='PyTorch freeway attack')
parser.add_argument('--seed', type=int, default=20, metavar='N',
					help='random seed (default: 20)')
parser.add_argument('--render', action='store_true',
					help='render the environment')
parser.add_argument('--num_evals', type=int, default=1000, metavar='N',
					help='Number of evaluations')
parser.add_argument('--num_smoothing_points', type=int, default=128, metavar='N',
					help='Number of points to use for smoothing')
parser.add_argument('--store_all_rewards', action='store_true',
					help='store all rewards (vs just sum)')
parser.add_argument('--checkpoint', type=str,
					help='checkpoint path')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
					help='interval between training status logs (default: 10)')
parser.add_argument('--attack_eps', type=float, default=255.*.1, metavar='N',
					help='Attack epsilon, total')
parser.add_argument('--sigma', type=float, default=255.*.1, metavar='N',
					help='Smoothing std. dev.')
parser.add_argument('--q_threshold', type=float, default=.0, metavar='N',
					help='q value threshold')
parser.add_argument('--attack_step', type=float, default=255.*0.01, metavar='N',
					help='Attack step size')
parser.add_argument('--attack_step_count_multiplier', type=float, default=2, metavar='N',
					help='Multiplier between steps and budget/attack_step')
args = parser.parse_args()

# Use the same noise samples to smooth every element in batch, then return the average of fun accross the batch
def soft_smooth_fun(fun, batch, noise):
	big_batch = (batch.unsqueeze(1) +noise.unsqueeze(0)).reshape([batch.shape[0]*noise.shape[0]]+list(batch.shape[1:]))
	out = fun(big_batch)
	return out.reshape([batch.shape[0],noise.shape[0]]+list(out.shape[1:])).mean(dim=1)
def attack(state,model,carryover_budget_squared, num_smoothing_points = 128, clean_prev_obs_tens = None, dirty_prev_obs_tens = None):

	if (clean_prev_obs_tens is not None):
		#prev_obs_tens = torch.stack(prev_obs_tens).cuda().reshape(-1).unsqueeze(0)
		clean_prev_obs_tens = torch.stack(clean_prev_obs_tens).cuda().unsqueeze(0)
		dirty_prev_obs_tens = torch.cat(dirty_prev_obs_tens,dim=0).cuda().unsqueeze(0)
	#state = torch.tensor(state,device='cuda').float().unsqueeze(0)
	state = torch.tensor(state,device='cuda').float().unsqueeze(0).unsqueeze(0)

	if (clean_prev_obs_tens is not None):
		to_smooth = torch.cat([clean_prev_obs_tens,state],dim=1)
		noise_clean = torch.randn([num_smoothing_points] +list(to_smooth.shape[1:]), device='cuda') * args.sigma
		target_logits =  soft_smooth_fun( lambda x : model.q_net(x), to_smooth, noise_clean)[0]
	else:
		noise_clean = torch.randn([num_smoothing_points] +list(state.shape[1:]), device='cuda') * args.sigma

		target_logits =  soft_smooth_fun( lambda x : model.q_net(x.repeat(1,4,1,1)) ,state,  noise_clean)[0]
	#target = target_logits.argmin().unsqueeze(0)
	starting_action = target_logits.argmax()
	#clean_out = policy.step_for_attack(state).detach()
	ori_state = copy.deepcopy(state.data)
	state= state.detach()
	obj = torch.nn.CrossEntropyLoss()
	if (carryover_budget_squared <= 0):
		budget = 0
	else:
		budget = carryover_budget_squared
	step_count = int(args.attack_step_count_multiplier * budget/args.attack_step)
	#targets = list([torch.tensor(x).cuda() for x in range(target_logits.shape[0])])
	targets = list([torch.tensor(x).cuda() for x in range(target_logits.shape[0])])
	#print('here')
	#print(starting_action)
	#print((target_logits + 0.02 >target_logits[starting_action]).nonzero(as_tuple=True))
	[targets.remove(x.item()) for x in (target_logits + args.q_threshold >target_logits[starting_action]).nonzero(as_tuple=False)]
	best_q = target_logits[starting_action]
	best_state = ori_state
	noise = torch.randn([num_smoothing_points] +list(state.shape[1:]), device='cuda') * args.sigma

	for target in targets:
		state =  copy.deepcopy(ori_state.data)
		for i in range(step_count):
			state.requires_grad = True
			if (clean_prev_obs_tens is not None):
				out =  soft_smooth_fun( lambda x :  model.q_net(torch.cat([dirty_prev_obs_tens.repeat(num_smoothing_points,1,1,1),x],dim=1)), state, noise)
			else:
				out =  soft_smooth_fun( lambda x : model.q_net(x.repeat(1,4,1,1)) , state, noise)
			if (out[0].argmax() == target):
				if target_logits[target] < best_q:
					best_q = target_logits[target]
					best_state = state.detach_()
				break
			model.q_net.zero_grad()
			#cost = -out[0,label]
			cost = -obj(out,target.unsqueeze(0))
			grad, = torch.autograd.grad(inputs=state, outputs=cost)
			state = state + args.attack_step*grad/grad.sum().abs()
			#state = state.round()
			if ((state - ori_state).sum().abs() > budget):
				state = ori_state + (state - ori_state) * budget / (state - ori_state).sum().abs()
			state.clamp_(0,255)
			state= state.detach_()
	return best_state[0].cpu(), budget - (best_state - ori_state).sum().abs()

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.dqn.policies import MlpPolicy as DQNMlp
from stable_baselines3 import DQN


if __name__ == '__main__':
	# Multiprocess environment
	env =  TimeLimit(AtariPreprocessing(gym.make("FreewayHardNoFrameskip-v0")) ,250)


	env.seed(args.seed)
	reward_accum = []
	policy_kwargs = {}
	model = DQN.load(args.checkpoint)

	for i_episode in range(args.num_evals):
		state = env.reset()
		state = np.array(state)	
		ep_reward =0
		policy_rewards = []
		state_hist = []
		carryover = args.attack_eps
		frame_hist = []
		t = 0
		after_first_loss = False
		while True: 
			t += 1
			if (t == 1):
				observation,carryover = attack(state,model,carryover, num_smoothing_points = args.num_smoothing_points)
				observation += torch.randn_like(observation) * args.sigma
				frame_hist.extend([observation]*4)
				state_hist.extend([torch.tensor(state,device='cuda')]*4)
			else:
				observation,carryover = attack(state,model,carryover, clean_prev_obs_tens =state_hist[-3:],dirty_prev_obs_tens =frame_hist[-3:], num_smoothing_points = args.num_smoothing_points)
				observation += torch.randn_like(observation) * args.sigma
				frame_hist.append(observation)
				state_hist.append(torch.tensor(state,device='cuda'))
			#print(frame_hist[-5:])

			action, _ = model.predict(np.array(torch.cat(frame_hist[-4:],dim=0)),deterministic=True)
			#print(action)
			state, reward, done, _ = env.step(action)
			# if (reward == -1 and not after_first_loss):
			# 	after_first_loss = True
			# 	carryover = args.attack_eps*args.attack_eps
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
			#print(np.array(state_hist).mean(axis=0))
			#print(np.array(state_hist).std(axis=0))
			print('Episode '+ str(i_episode), flush=True)
	torch.save(reward_accum, args.checkpoint + '_evals_'+ str(args.num_evals) + '_smooth_attackl1_eps_' + str(args.attack_eps) + '_attack_step_count_multiplier_' + str(args.attack_step_count_multiplier) + '_attack_step_'+ str(args.attack_step)+ '_threshold_'+ str(args.q_threshold) +'_num_smoothing_points_'+ str(args.num_smoothing_points) +'.pth')
