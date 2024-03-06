import gym
from stable_baselines3 import A2C

# Load the inverted pendulum environment
env = gym.make('InvertedPendulum-v4', render_mode='human')

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()

for i in range(1000):
    action, _state = model.predict(obs, deterministic = True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
