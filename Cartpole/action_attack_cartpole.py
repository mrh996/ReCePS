import gym
import numpy as np
import torch
import gym
import numpy as np
from itertools import count
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from stable_baselines3 import DQN
import argparse
# Function to perform an action-space attack using FGSM
parser = argparse.ArgumentParser(description='PyTorch pong attack')
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
parser.add_argument('--attack_eps', type=float, default=.2, metavar='N',
					help='Attack epsilon, total')
parser.add_argument('--p', type=float, default=.16, metavar='N',
					help='Smoothing std. dev.')
parser.add_argument('--q_threshold', type=float, default=0, metavar='N',
					help='q value threshold')
parser.add_argument('--attack_step', type=float, default=0.01, metavar='N',
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
def action_attack(model, state, epsilon=0.1):
		state = torch.FloatTensor(state)
		state.requires_grad = True

		# Get the Q-values for the current state
		
		Q_values = pytorch_model(state)
		

		# Find the action with the highest Q-value
		action = Q_values.argmax().item()
		adversarial_action=1-action
		'''
		for i in range(10):
			# Compute the gradient of the Q-value with respect to the input state
			action_probs = F.softmax(Q_values, dim=-1)
			perturbation = epsilon * torch.sign(action_probs - (1.0 / action_probs.size(0)))
			
			

			# Add the perturbation to the Q-values to obtain the adversarial Q-values
			adversarial_Q_values = Q_values + perturbation

			# Find the action with the highest adversarial Q-value as the adversarial action
			adversarial_action = adversarial_Q_values.argmax().item()
			action=adversarial_action
  
  
		'''
		
		num_different_actions = int(action != adversarial_action)
		return adversarial_action,num_different_actions

# Define a DQN model in PyTorch
def threshold(state):
		Q_values = pytorch_model(state)
		qs=[]
		for i in Q_values:
			qs.append(np.exp(i.item()))
		c=np.max(qs/np.sum(qs))-np.min(qs/np.sum(qs))
		if Q_values.argmax().item()-Q_values.argmin().item()>=c:
			return True
		else:
			return False

class DQNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.dqn.policies import MlpPolicy as DQNMlp
from stable_baselines3 import DQN

from gym.wrappers import FrameStack, FlattenObservation
from gym.wrappers import AtariPreprocessing

if __name__ == '__main__':
	# Multiprocess environment
	env =  FlattenObservation(FrameStack(NoisyObsWrapper(gym.make("CartPole-v0"),0), 5) )
	


	env.seed(args.seed)
	reward_accum = []
	policy_kwargs = {}
	model = DQN.load(args.checkpoint)
	# Extract the parameters from the Stable Baselines 3 model
	input_size = env.observation_space.shape[0] 
	hidden_size = 256
	output_size = env.action_space.n
	model_params = model.policy.parameters()

	# Create a PyTorch model with the same architecture as the DQN agent
	pytorch_model = DQNModel(input_size, hidden_size, output_size)

	# Load the parameters from the DQN agent to the PyTorch model
	state_dict = dict(zip(pytorch_model.state_dict().keys(), model_params))
	pytorch_model.load_state_dict(state_dict, strict=False)

	for i_episode in range(args.num_evals):
		state = env.reset()
		ep_reward =0
		policy_rewards = []
		state_hist = []
		
		frame_hist = []
		t = 0
		l0e=0
		after_first_loss = False
		for t in range(1, 201):
			if threshold(torch.tensor(state))==True and l0e<args.attack_eps:
			#if np.random.rand() <= args.p and l0e<=args.attack_eps:
				

				action,diff = action_attack(model,state)
				l0e=l0e+1
				
			else:
				action, _ = model.predict(state,deterministic=True)
			next_state, reward, done, _ = env.step(action)
			state = next_state
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
			#print(np.array(state_hist).mean(axis=0))
			#print(np.array(state_hist).std(axis=0))
			print(str(ep_reward), flush=True)
	torch.save(reward_accum, args.checkpoint + '_evals_'+ str(args.num_evals) + '_smooth_attackl0_eps_' + str(args.attack_eps) + '_attack_step_count_multiplier_' + str(args.attack_step_count_multiplier) + '_attack_step_'+ str(args.attack_step)+ '_threshold_'+ str(args.q_threshold) +'_num_smoothing_points_'+ str(args.num_smoothing_points) +'.pth')

