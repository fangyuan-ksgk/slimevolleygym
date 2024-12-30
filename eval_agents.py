"""
Multiagent example.

Evaluate the performance of different trained models in zoo against each other.

This file can be modified to test your custom models later on against existing models.

Model Choices
=============

BaselinePolicy: Default built-in opponent policy (trained in earlier 2015 project)

baseline: Baseline Policy (built-in AI). Simple 120-param RNN.
ppo: PPO trained using 96-cores for a long time vs baseline AI (train_ppo_mpi.py)
cma: CMA-ES with small network trained vs baseline AI using estool
ga: Genetic algorithm with tiny network trained using simple tournament selection and self play (input x(train_ga_selfplay.py)
random: random action agent
"""

import warnings
# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import os
import numpy as np
import argparse
import slimevolleygym
from slimevolleygym.mlp import makeSlimePolicy, makeSlimePolicyLite # simple pretrained models
from slimevolleygym import BaselinePolicy
from time import sleep

#import cv2

np.set_printoptions(threshold=20, precision=4, suppress=True, linewidth=200)

PPO1 = None # from stable_baselines import PPO1 (only load if needed.)
class PPOPolicy:
  def __init__(self, path):
    self.model = PPO1.load(path)
  def predict(self, obs):
    action, state = self.model.predict(obs, deterministic=True)
    return action

class RandomPolicy:
  def __init__(self, path):
    self.action_space = gym.spaces.MultiBinary(3)
    pass
  def predict(self, obs):
    return self.action_space.sample()

def makeBaselinePolicy(_):
  return BaselinePolicy()

def rollout(env, policy0, policy1, render_mode=False):
  """ play one agent vs the other in modified gym-style loop. """
  obs0 = env.reset()
  obs1 = obs0 # same observation at the very beginning for the other agent

  done = False
  total_reward = 0
  frames = []  # Store frames for visualization verification

  while not done:
    action0 = policy0.predict(obs0)
    action1 = policy1.predict(obs1)

    # uses a 2nd (optional) parameter for step to put in the other action
    # and returns the other observation in the 4th optional "info" param in gym's step()
    obs0, reward, done, info = env.unwrapped.step(action0, action1)  # Use unwrapped env for direct access to two-action step method
    obs1 = info['otherObs']

    total_reward += reward

    if render_mode:
      frame = env.render()
      if frame is not None:  # Only append valid frames
        frames.append(frame)
      sleep(0.01)  # Small delay to not overwhelm the system

  return total_reward, frames

def evaluate_multiagent(env, policy0, policy1, render_mode=False, n_trials=1000, init_seed=721):
  history = []
  all_frames = []
  for i in range(n_trials):
    env.seed(seed=init_seed+i)
    cumulative_score, frames = rollout(env, policy0, policy1, render_mode=render_mode)
    print("cumulative score #", i, ":", cumulative_score)
    history.append(cumulative_score)
    if frames:  # Only save frames if we got any
      all_frames.extend(frames)
      
  # Save frames if we collected any
  if all_frames:
    import os
    import numpy as np
    save_dir = '/tmp/eval_frames'
    os.makedirs(save_dir, exist_ok=True)
    frames_array = np.array(all_frames)
    np.save(f'{save_dir}/game_frames.npy', frames_array)
    print(f"Saved {len(all_frames)} frames to {save_dir}/game_frames.npy")
    
    # Save first and last frame as PNG for quick verification
    try:
      import cv2
      cv2.imwrite(f'{save_dir}/first_frame.png', cv2.cvtColor(all_frames[0], cv2.COLOR_RGB2BGR))
      cv2.imwrite(f'{save_dir}/last_frame.png', cv2.cvtColor(all_frames[-1], cv2.COLOR_RGB2BGR))
      print(f"Saved first and last frames as PNG images in {save_dir}/")
    except ImportError:
      print("Warning: cv2 not available, skipping PNG export")
      
  return history

if __name__=="__main__":

  APPROVED_MODELS = ["baseline", "ppo", "ga", "cma", "random"]

  def checkchoice(choice):
    choice = choice.lower()
    if choice not in APPROVED_MODELS:
      return False
    return True

  PATH = {
    "baseline": None,
    "ppo": "zoo/ppo/best_model.zip",
    "cma": "zoo/cmaes/slimevolley.cma.64.96.best.json",
    "ga": "zoo/ga_sp/ga.json",
    "random": None,
  }

  MODEL = {
    "baseline": makeBaselinePolicy,
    "ppo": PPOPolicy,
    "cma": makeSlimePolicy,
    "ga": makeSlimePolicyLite,
    "random": RandomPolicy,
  }

  parser = argparse.ArgumentParser(description='Evaluate pre-trained agents against each other.')
  parser.add_argument('--left', help='choice of (baseline, ppo, cma, ga, random)', type=str, default="baseline")
  parser.add_argument('--leftpath', help='path to left model (leave blank for zoo)', type=str, default="")
  parser.add_argument('--right', help='choice of (baseline, ppo, cma, ga, random)', type=str, default="ga")
  parser.add_argument('--rightpath', help='path to right model (leave blank for zoo)', type=str, default="")
  parser.add_argument('--render', action='store_true', help='render to screen?', default=False)
  parser.add_argument('--day', action='store_true', help='daytime colors?', default=False)
  parser.add_argument('--pixel', action='store_true', help='pixel rendering effect? (note: not pixel obs mode)', default=False)
  parser.add_argument('--seed', help='random seed (integer)', type=int, default=721)
  parser.add_argument('--trials', help='number of trials (default 1000)', type=int, default=1000)

  args = parser.parse_args()

  if args.day:
    slimevolleygym.setDayColors()

  if args.pixel:
    slimevolleygym.setPixelObsMode()

  render_mode = 'rgb_array' if args.render else None
  env = gym.make("SlimeVolley-v0", render_mode=render_mode)
  env.seed(args.seed)

  assert checkchoice(args.right), "pls enter a valid agent"
  assert checkchoice(args.left), "pls enter a valid agent"

  c0 = args.right
  c1 = args.left

  path0 = PATH[c0]
  path1 = PATH[c1]

  if len(args.rightpath) > 0:
    assert os.path.exists(args.rightpath), args.rightpath+" doesn't exist."
    path0 = args.rightpath
    print("path of right model", path0)

  if len(args.leftpath):
    assert os.path.exists(args.leftpath), args.leftpath+" doesn't exist."
    path1 = args.leftpath
    print("path of left model", path1)

  if c0.startswith("ppo") or c1.startswith("ppo"):
    from stable_baselines import PPO1

  policy0 = MODEL[c0](path0) # the right agent
  policy1 = MODEL[c1](path1) # the left agent

  history = evaluate_multiagent(env, policy0, policy1,
    render_mode=args.render, n_trials=args.trials, init_seed=args.seed)

  print("history dump:", history)
  print(c0+" scored", np.round(np.mean(history), 3), "±", np.round(np.std(history), 3), "vs",
    c1, "over", args.trials, "trials.")
