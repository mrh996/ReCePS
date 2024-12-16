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
parser.add_argument('--p', type=float, default=0.1, metavar='N',
					help='Smoothing std. dev.')
parser.add_argument('--q_threshold', type=float, default=.0, metavar='N',
					help='q value threshold')
parser.add_argument('--attack_step', type=float, default=255.*0.01, metavar='N',
					help='Attack step size')
parser.add_argument('--attack_step_count_multiplier', type=float, default=2, metavar='N',
					help='Multiplier between steps and budget/attack_step')
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
from stable_baselines3.common.preprocessing import preprocess_obs

import time
def action_attack(model, state, epsilon=0.1):
		
		

		for i in range(10):
			# Compute the gradient of the Q-value with respect to the input state
			action_probs =  model.q_net(state)
			perturbation = epsilon * torch.sign(action_probs - (1.0 / action_probs.size(0)))
			
			

			# Add the perturbation to the Q-values to obtain the adversarial Q-values
			adversarial_Q_values = action_probs + perturbation

			# Find the action with the highest adversarial Q-value as the adversarial action
			adversarial_action = adversarial_Q_values.argmax().item()
			action=adversarial_action
		
		num_different_actions = int(action != adversarial_action)
		return adversarial_action,num_different_actions

# Define a DQN model in PyTorch
def threshold(model,state):
		state_tensor = torch.tensor(state)
		state_tensor = state_tensor.unsqueeze(0)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
				
		state_tensor = state_tensor.to(device)
		Q_values = model.q_net(state_tensor)
		qs=[]
		for i in Q_values[0]:
			
			qs.append(np.exp(i.item()))
		c=np.max(qs/np.sum(qs))-np.min(qs/np.sum(qs))
		if Q_values.argmax().item()-Q_values.argmin().item()>=c:
			return True
		else:
			return False
if __name__ == '__main__':
	# Multiprocess environment
	
	env =  TimeLimit(FrameStack(NoisyObsWrapper(AtariPreprocessing(gym.make("FreewayHardNoFrameskip-v0")), 0), 4) ,250)
	

	env.seed(args.seed)
	reward_accum = []
	policy_kwargs = {}
	model = DQN.load(args.checkpoint)
	# Extract the parameters from the Stable Baselines 3 model


	# Create a PyTorch model with the same architecture as the DQN agent
	#pytorch_model = DQNModel(input_size, hidden_size, output_size)

	# Load the parameters from the DQN agent to the PyTorch model
	#state_dict = dict(zip(pytorch_model.state_dict().keys(), model_params))
	#pytorch_model.load_state_dict(state_dict, strict=False)
	eps=0.1

	for i_episode in range(args.num_evals):
		state = env.reset()
		
		ep_reward =0
		policy_rewards = []
		t = 0
		l0e=0
		while True:
			t += 1
			if threshold(model,state)==True and l0e<args.attack_eps:
				
			#if np.random.rand() <= args.p and l0e<=args.attack_eps:
				state_tensor = torch.tensor(state)
				state_tensor = state_tensor.unsqueeze(0)
				device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
				
				state_tensor = state_tensor.to(device)
				

				action,diff = action_attack(model,state_tensor)
				l0e=l0e+diff
				
			else:
				state=np.array(state)
				action, _ = model.predict(state,deterministic=True)
			if np.random.rand() <= eps:
				state_tensor = torch.tensor(state)
				state_tensor = state_tensor.unsqueeze(0)
				device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
				
				state_tensor = state_tensor.to(device)
				Q_values = model.q_net(state_tensor)
				action = Q_values.argmin().item()
				
			else:
				state = np.array(state)
				action, _ = model.predict(state,deterministic=True)
			
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
			print(str(ep_reward), flush=True)
	torch.save(reward_accum, args.checkpoint + '_evals_'+ str(args.num_evals) + '_smooth_attackl0_eps_' + str(args.attack_eps) + '_attack_step_count_multiplier_' + str(args.attack_step_count_multiplier) + '_attack_step_'+ str(args.attack_step)+ '_threshold_'+ str(args.q_threshold) +'_num_smoothing_points_'+ str(args.num_smoothing_points) +'.pth')
