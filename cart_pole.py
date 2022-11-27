
#import all python packages
pip install pyglet==1.5.27 
# use !pip install pyglet==1.5.27 for Jupyter Notebook
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import warnings
warnings.filterwarnings('ignore')

#defining model
environment_name = 'CartPole-v1' # by OpenAI Gym
environment = gym.make(environment_name)
environment = DummyVecEnv([lambda: environment])
model = PPO('MlpPolicy',environment, verbose = 1)

#model traning
model.learn(total_timesteps=30000) # total_timesteps can be modiyed accordingly

#run model
for episode in range(1, 11):
  score = 0
  obs = environment.reset()
  done = False

  while not done:
    environment.render()
    action, _ = model.predict(obs)
    obs, reward, done, info = environment.step(action)
    score += reward

  print('Episode: ',episode,'score: ',score)
environment.close()

#Evaluate the model
evaluate_policy(model, environment, n_eval_episodes=15, render=True)
