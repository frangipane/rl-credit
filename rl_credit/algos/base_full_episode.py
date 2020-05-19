"""
Collects full episodes sequentially.
"""
from abc import ABC, abstractmethod
import torch

from rl_credit.format import default_preprocess_obss
from rl_credit.utils import DictList


class BaseAlgoFullEpisode(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, buf_size=None):
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

        #self.env = ParallelEnv(envs)
        self.env = envs[0]  # take the first environment
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
        # num_procs and num_frames_per_proc aren't actually used to parallelize step
        # through environment, only to calculate target num_frames before update.
        self.buf_size = buf_size or self.num_frames

    def collect_experiences(self):
        # Initialize log values
        self.log_return = [0]
        self.log_reshaped_return = [0]
        self.log_num_frames = [0]

        num_frames = 0
        exps = DictList(action=[], value=[], reward=[], advantage=[], returnn=[], log_prob=[], obs=[])

        # Collect full episodes, where total number of frames is >= self.num_frames.
        while num_frames < self.num_frames:
            exps_for_episode = self.rollout()
            num_frames += self.log_num_frames[-1]

            if len(exps) == 0:
                exps = exps_for_episode
            else:
                if self.acmodel.recurrent:
                    exps.memory += exps_for_episode.memory
                    exps.mask += exps_for_episode.mask

                exps.action = torch.cat((exps.action, exps_for_episode.action))
                exps.value = torch.cat((exps.value, exps_for_episode.value))
                exps.reward = torch.cat((exps.reward, exps_for_episode.reward))
                exps.advantage = torch.cat((exps.advantage, exps_for_episode.advantage))
                exps.returnn = torch.cat((exps.returnn, exps_for_episode.returnn))
                exps.log_prob = torch.cat((exps.log_prob, exps_for_episode.log_prob))
                exps.obs.image = torch.cat((exps.obs.image, exps_for_episode.obs.image))
                exps.obs.text = torch.cat((exps.obs.text, exps_for_episode.obs.text))

        # normalize the advantage
        exps.advantage = (exps.advantage - exps.advantage.mean())/exps.advantage.std()

        logs = {
            "return_per_episode": self.log_return,
            "reshaped_return_per_episode": self.log_reshaped_return,
            "num_frames_per_episode": self.log_num_frames,
            "num_frames": num_frames
        }

        return exps, logs

    def rollout(self):
        """Collect experience for a full episode
        """
        # Initialize experience values

        # bufsize is used as a max buffer size for the full episode.
        # if episode terminates before hitting bufsize, the returned
        # experience is shorter
        shape = (self.buf_size, 1)

        obss_ep = [None]*(shape[0])

        if self.acmodel.recurrent:
            memory_ep = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            memories_ep = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        mask_ep = torch.ones(shape[1], device=self.device)
        masks_ep = torch.zeros(*shape, device=self.device)
        actions_ep = torch.zeros(*shape, device=self.device, dtype=torch.int)
        values_ep = torch.zeros(*shape, device=self.device)
        rewards_ep = torch.zeros(*shape, device=self.device)
        advantages_ep = torch.zeros(*shape, device=self.device)
        log_probs_ep = torch.zeros(*shape, device=self.device)

        # Initialize log values

        log_episode_return = 0
        log_episode_reshaped_return = 0
        log_episode_num_frames = 0

        obs, done, i = self.env.reset(), False, 0

        while not done:
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss([obs], device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(preprocessed_obs,
                                                       memory_ep * mask_ep.unsqueeze(1))
                else:
                    dist, value = self.acmodel(preprocessed_obs)
            action = dist.sample()

            obs2, reward, done, _ = self.env.step(action.cpu().numpy())

            # Update experiences values

            obss_ep[i] = obs
            obs = obs2
            if self.acmodel.recurrent:
                memories_ep[i] = memory_ep
                memory_ep = memory
            masks_ep[i] = mask_ep
            mask_ep = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            actions_ep[i] = action
            values_ep[i] = value
            if self.reshape_reward is not None:
                rewards_ep[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                rewards_ep[i] = torch.tensor(reward, device=self.device)
            log_probs_ep[i] = dist.log_prob(action)

            # Update log values

            log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            log_episode_reshaped_return += rewards_ep[i]
            log_episode_num_frames += 1

            i += 1

        # end of episode logs
        self.log_return.append(log_episode_return.item())
        self.log_reshaped_return.append(log_episode_reshaped_return.item())
        self.log_num_frames.append(log_episode_num_frames)

        # Add advantage and return to experiences
        ep_len = i

        preprocessed_obs = self.preprocess_obss([obs], device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, memory_ep * mask_ep.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs)

        for i in reversed(range(ep_len)):
            next_mask = masks_ep[i+1] if i < ep_len - 1 else mask_ep
            next_value = values_ep[i+1] if i < ep_len - 1 else next_value
            next_advantage = advantages_ep[i+1] if i < ep_len - 1 else 0

            delta = rewards_ep[i] + self.discount * next_value * next_mask - values_ep[i]
            advantages_ep[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is ep_len self.num_frames_per_proc,
        #   - P is 1,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [obss_ep[i] for i in range(ep_len)]
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = memories_ep.transpose(0, 1).reshape(-1, *memories_ep.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = masks_ep.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = actions_ep.transpose(0, 1).reshape(-1)[:ep_len]
        exps.value = values_ep.transpose(0, 1).reshape(-1)[:ep_len]
        exps.reward = rewards_ep.transpose(0, 1).reshape(-1)[:ep_len]
        exps.advantage = advantages_ep.transpose(0, 1).reshape(-1)[:ep_len]
        exps.returnn = (exps.value + exps.advantage)[:ep_len]
        exps.log_prob = log_probs_ep.transpose(0, 1).reshape(-1)[:ep_len]

        # Preprocess experiences
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        return exps

    @abstractmethod
    def update_parameters(self):
        pass
