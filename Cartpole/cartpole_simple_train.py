import gym
import numpy as np
import argparse

class NoisyObsWrapper(gym.ObservationWrapper):
	def __init__(self, env, sigma):
		super().__init__(env)
		self.sigma = sigma
	def observation(self, obs):
		return obs + self.sigma*np.random.standard_normal(size=obs.shape)


from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.dqn.policies import MlpPolicy as DQNMlp
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

parser = argparse.ArgumentParser(description='Cartpole DQN example')
parser.add_argument('--sigma', type=float, default=0.0, metavar='N',
					help='How much noise to smooth observations')
args = parser.parse_args()


if __name__ == '__main__':
	# Multiprocess environment
	sigma = args.sigma
	env =  NoisyObsWrapper(gym.make("CartPole-v0"), sigma) 
	eval_env =  NoisyObsWrapper(gym.make("CartPole-v0"), sigma) 
	eval_callback = EvalCallback(eval_env, best_model_save_path="cartpole_simple_sigma_"+ str(sigma),
		log_path="./logs_simple/"+ str(sigma)+'/', eval_freq=2000, n_eval_episodes=10)

	policy_kwargs = {}
	model = DQN(DQNMlp, env,
				gamma=0.99,
				learning_rate=5e-5,
				buffer_size=100000,
				learning_starts=1000,
				exploration_fraction=0.16,
				target_update_interval=10,
				batch_size=1024,
				verbose=1,
				train_freq= 256,
				gradient_steps=128,
				exploration_final_eps= 0.0,
				policy_kwargs={'net_arch': [256, 256]},
				tensorboard_log="./logs_simple/"+ str(sigma)+'/')
	model.learn(total_timesteps=5e5, callback=eval_callback)
	#model.save("deepq_cartpole_simple_sigma_"+ str(sigma))
