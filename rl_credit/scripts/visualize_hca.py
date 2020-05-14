"""Manually navigate gridworld, then use HCA returns model to evaluate hca factor 
(comparing hca from two models) afterwards, make plots.

Example:

python -m scripts.visualize_hca --env MiniGrid-KeyGoal-6x6-v0 --model
hca_returns_0mem_keygoal --model2 hca_returns_0mem_keygoal2

"""
import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from gym_minigrid.window import Window

import script_utils as utils
from rl_credit.model import ACModelReturnHCA


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--model2", required=True,
                    help="name of the 2nd trained model (REQUIRED)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
parser.add_argument("--tile_size", type=int,
                    help="size at which to render tiles", default=32)
parser.add_argument('--agent_view', default=False,
                    help="draw the agent sees (partially observable view)",
                    action='store_true')


args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, args.seed)
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

obs_space, preprocess_obss = utils.get_obss_preprocessor(env.observation_space)

acmodel = ACModelReturnHCA(obs_space, env.action_space)
acmodel.load_state_dict(utils.get_model_state(utils.get_model_dir(args.model)))
acmodel.to(device)
acmodel.eval()

acmodel2 = ACModelReturnHCA(obs_space, env.action_space)
acmodel2.load_state_dict(utils.get_model_state(utils.get_model_dir(args.model2)))
acmodel2.to(device)
acmodel2.eval()

print("Models loaded\n")


# Buffer of observations for an episode
class Buffer:
    def __init__(self):
        self.observations = []

    def store(self, obs):
        self.observations.append(obs)


def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)


def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()
    buf.store(obs)

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)


def step(action):
    obs, reward, done, info = env.step(action)

    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        #reset()
    else:
        buf.store(obs)
        redraw(obs)


def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

# Begin manual control

buf = Buffer()
    
window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)
reset()

# blocking event loop
window.show(block=True)

num_frames = len(buf.observations)
print('Number of frames:', num_frames)


# Evaluate hca for both models
def get_hca_probs(Z):
    with torch.no_grad():
        _, _, hca_logits = acmodel(preprocess_obss(buf.observations),
                                   torch.tensor([Z]*num_frames))
        hca_probs = F.softmax(hca_logits).numpy()

        _, _, hca_logits2 = acmodel2(preprocess_obss(buf.observations),
                                     torch.tensor([Z]*num_frames))
        hca_probs2 = F.softmax(hca_logits2).numpy()

    return hca_probs, hca_probs2

# plot HCA for a fixed return!
hca_probs, hca_probs2 = get_hca_probs(Z=3.6)

fig, ax = plt.subplots(figsize=(8,6))
action_names=[n.name for n in env.Actions]

for a in range(env.action_space.n):
    plt.plot(hca_probs[:,a], hca_probs2[:,a], label=action_names[a])
plt.legend()
plt.show()


# Evaluate mean divergence b/w HCA distributions for different returns

i = 5  # index corresponding to an arbitrarily chosen observation
hca = []  # hca probs for the observation
divergences = []

zs = np.arange(0, 4, 0.1)
for z in zs:
    hca_probs, hca_probs2 = get_hca_probs(z)
    diverg = (np.log(hca_probs) - np.log(hca_probs2)).mean(axis=1)  # diverg per obs
    divergences.append(diverg)
    hca.append(hca_probs[i,:])


# Plot divergences per observation, per return.
divergences_array = np.array(divergences)
print(divergences_array.shape)
plt.matshow(divergences_array)
plt.xlabel('observation number')
#plt.ylabel('Return')
plt.colorbar()
plt.show()


# For i-th observation (randomly chosen), plot HCA returns probabilities across actions,
# given different returns
print('hca probs')
plt.matshow(np.array(hca))
plt.colorbar()
plt.show()
