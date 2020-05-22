from abc import ABC, abstractmethod
import torch

from rl_credit.format import default_preprocess_obss
from rl_credit.utils import DictList, ParallelEnv
import rl_credit.script_utils as utils

import numpy
import torch.nn.functional as F

from rl_credit.model import A2CAttention


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters

        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward

        # Control parameters

        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.acmodel.to(self.device)
        self.acmodel.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            # preprocessed_obs shape: [num_procs, h, w, c]
            # add extra dim for ep len so shape -> [num_procs, 1, h, w, c]
            preprocessed_obs = torch.unsqueeze(preprocessed_obs, dim=1)

            with torch.no_grad():
                dist, _ = self.acmodel(preprocessed_obs)
            action = dist.sample()

            obs, reward, done, _ = self.env.step(action.cpu().numpy())
            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        # Reshape obss into tensor of (batch size=num_procs, seq len=frames per proc, *(image_dim))
        obss_mat = [None]*(self.num_procs)
        for i in range(self.num_procs):
            obss_mat[i] = self.preprocess_obss([self.obss[j][i]
                                                for j in range(self.num_frames_per_proc)])
        obss_mat = torch.cat(obss_mat).view(self.num_procs, *obss_mat[0].shape)
        
        exps = DictList()
        # exps.obs = [self.obss[i][j]
        #             for j in range(self.num_procs)
        #             for i in range(self.num_frames_per_proc)]
        # T x P -> P x T -> (P * T) x 1
        exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # ===== Add advantage and return to experiences =====

        # Calculate values using whole context from episode
        with torch.no_grad():
            _, value = self.acmodel(obss_mat)

        self.values = value.reshape(self.num_frames_per_proc, self.num_procs)
        # last observations per episode
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        preprocessed_obs = torch.unsqueeze(preprocessed_obs, dim=1)

        # bootstrapped final value for unfinished trajectories cut off by end
        # of epoch (=num frames per proc)
        with torch.no_grad():
            _, next_value = self.acmodel(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        # normalize the advantage
        exps.advantage = (exps.advantage - exps.advantage.mean())/exps.advantage.std()
        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return obss_mat, exps, logs

    @abstractmethod
    def update_parameters(self):
        pass


class AttentionAlgo(BaseAlgo):
    """The Advantage Actor-Critic algorithm."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None, reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

    def update_parameters(self, obss, exps):
        # ===== Calculate losses =====

        dist, value = self.acmodel(obss)

        entropy = dist.entropy().mean()

        policy_loss = -(dist.log_prob(exps.action) * exps.advantage).mean()

        value_loss = (value - exps.returnn).pow(2).mean()

        loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

        # Update actor-critic

        self.optimizer.zero_grad()
        loss.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log some values

        with torch.no_grad():
            # evaluate KL divergence b/w old and new policy
            # policy under newly updated model
            dist, _ = self.acmodel(obss)

            approx_kl = (exps.log_prob - dist.log_prob(exps.action)).mean().item()
            adv_mean = exps.advantage.mean().item()
            adv_max = exps.advantage.max().item()
            adv_min = exps.advantage.min().item()
            adv_std = exps.advantage.std().item()

            # standard deviation of values
            value_std = value.std().item()

        logs = {
            "entropy": entropy.item(),
            "value": value.mean().item(),
            "value_std": value_std,
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "grad_norm": update_grad_norm,
            "adv_max": adv_max,
            "adv_min": adv_min,
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "kl": approx_kl,
        }
        return logs

    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = numpy.arange(0, self.num_frames, self.recurrence)
        return starting_indexes


def get_obss_preprocessor(obs_space):
    import gym
    # Check if it is a MiniGrid observation space
    if isinstance(obs_space, gym.spaces.Dict) and list(obs_space.spaces.keys()) == ["image"]:
        obs_space = obs_space.spaces["image"].shape

        def preprocess_obss(obss, device=None):
            images = numpy.array([obs["image"] for obs in obss])
            return torch.tensor(images, device=device, dtype=torch.float)
    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_procs = 4
    seed = 0

    envs = []
    for i in range(num_procs):
        envs.append(utils.make_env('MiniGrid-KeyGoal-6x6-v0', seed + 10000 * i))

    obs_space, preprocess_obss = get_obss_preprocessor(envs[0].observation_space)
    acmodel = A2CAttention(obs_space, envs[0].action_space,)

    algo_args=dict(device=device,
                   num_frames_per_proc=128,
                   discount=1.,
                   lr=0.001,
                   gae_lambda=1.,
                   entropy_coef=0.,
                   value_loss_coef=0.5,
                   max_grad_norm=0.5,
                   recurrence=1,
                   rmsprop_alpha=0.99,
                   rmsprop_eps=1e-8,
                   preprocess_obss=preprocess_obss)
    algo = AttentionAlgo(envs, acmodel, **algo_args)

    total_frames = 0
    obss, exps, logs = algo.collect_experiences()
    algo.update_parameters(obss, exps)
    total_frames += logs['num_frames']