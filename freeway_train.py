import gym
import numpy as np
import argparse

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
from stable_baselines3.dqn.policies import CnnPolicy 
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from gym.wrappers import TimeLimit,FrameStack, AtariPreprocessing
parser = argparse.ArgumentParser(description='Freeway DQN example')
parser.add_argument('--sigma', type=float, default=0.0, metavar='N',
					help='How much noise to smooth observations')
parser.add_argument('--index', type=int, default=0, metavar='N',
					help='Index')
args = parser.parse_args()


if __name__ == '__main__':
	# Multiprocess environment
	sigma = args.sigma * 255.
	exploration_frac=0.2
	env =  TimeLimit(FrameStack(NoisyObsWrapper(AtariPreprocessing(gym.make("FreewayHardNoFrameskip-v0")), sigma), 4) ,250)
	eval_env =  TimeLimit(FrameStack(NoisyObsWrapper(AtariPreprocessing(gym.make("FreewayHardNoFrameskip-v0")), sigma), 4) ,250)
	eval_callback = EvalCallback(eval_env, best_model_save_path="freeway_sigma_ef"+ str(exploration_frac)+ '_'+str(args.index),
		log_path="./logs_freeway/"+ str(exploration_frac)+'/'+ str(args.index)+'/', eval_freq=100000, n_eval_episodes=100)
    

	policy_kwargs = {}
	model = DQN(CnnPolicy, env,
				learning_rate=0.0001,
				buffer_size=10000,
				learning_starts=100000,
				exploration_fraction=exploration_frac,
				target_update_interval=1000,
				batch_size=32,
				verbose=1,
				train_freq= 4,
				gradient_steps=1,
				exploration_final_eps= 0.01,
				policy_kwargs={},
				optimize_memory_usage = True,
				tensorboard_log="./logs_freeway/"+ str(exploration_frac)+'/'+ str(args.index)+'/')
	model.learn(total_timesteps=10000000, callback=eval_callback)
