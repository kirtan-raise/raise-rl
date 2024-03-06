import gymnasium as gym
from stable_baselines3 import A2C
from training.env.bracket_insertion import Ur20BracketInsertEnv

# Load the inverted pendulum environment
# gym.register("Ur20BracketInsert-V0", Ur20BracketInsertEnv)
env = gym.make('Ur20BracketInsert-V0', render_mode='human')

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)
