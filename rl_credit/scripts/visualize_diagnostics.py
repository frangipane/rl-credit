import argparse
import time
import numpy
import torch
import matplotlib.pyplot as plt
from gym_minigrid.window import Window


import script_utils as utils


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
parser.add_argument("--hcareturns", action="store_true", default=False,
                    help="use HCA returns model")
parser.add_argument("--hcastate", action="store_true", default=False,
                    help="use HCA state model")

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

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    device=device, argmax=args.argmax, use_memory=args.memory, use_text=args.text,
                    hca_returns=args.hcareturns, hca_state=args.hcastate)
print("Agent loaded\n")

# Run the agent

if args.gif:
   from array2gif import write_gif
   frames = []


# Create a window to view the environment
#env.render('human')
class Diagnostics(Window):
    def __init__(self, title):
        super().__init__(title)

    def plot_policy(self, pi, xticks=None):
        self.ax.cla()
        n_actions = len(pi)
        x = range(n_actions)
        self.ax.bar(x, pi)
        self.ax.set_ylabel('Probability')
        if xticks is not None:
            self.ax.set_xticks(x)
            self.ax.set_xticklabels(xticks)
        #self.fig.canvas.draw()

        plt.pause(0.001)

for episode in range(args.episodes):
    piplot = Diagnostics('policy')

    obs = env.reset()
    

    while True:
        env.render('human')
        piplot.show(block=False)
        if args.gif:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

        action, policy, value = agent.get_action(obs)
        piplot.plot_policy(policy, xticks=[n.name for n in env.Actions])

        obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)

        if done:# or env.window.closed:
            break

        if env.window is not None and env.window.closed:
            break

    if env.window.closed:
        break
    env.render(close=True)  # close window after episode is over


if args.gif:
    print("Saving gif... ", end="")
    write_gif(frames, args.gif+".gif", fps=1/args.pause)
    #write_gif(np_frames, args.gif+".gif", fps=1/args.pause)
    #write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")
