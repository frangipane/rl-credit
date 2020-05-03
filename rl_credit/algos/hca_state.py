import numpy
import torch
import torch.nn.functional as F

from rl_credit.algos.base import BaseAlgo


class HCAState(BaseAlgo):
    """The state HCA Actor-Critic algorithm."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=1,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None, reshape_reward=None):

        if recurrence != 1:
            raise ValueError("Memory is not supported for HCAState, recurrence must be 1.")
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

    def _policy_loss_for_episode(self, exps):
        traj_len = len(exps.reward)
        n_actions = self.env.action_space.n

        k = 0  # pointer starts at beginning of episode
        policy_loss = 0
        hca_loss = 0
        reward_loss = 0

        while k < traj_len - 1:
            with torch.no_grad():
                # for t in range(k + 1, traj_len):
                #     _, _, hca_logits = self.acmodel(exps.obs[k], exps.obs[t])
                #     hca_prob = F.softmax(hca_logits, dim=1)
                #     hca_factor = hca_prob * exps.reward[t]  # todo: include discount factor

                # vectorized version of the above
                pi_dist, _, hca_logits = self.acmodel(exps.obs[k], obs2=exps.obs[k+1:traj_len])
                hca_prob = F.softmax(hca_logits, dim=1)
                discount_factor = torch.tensor([self.discount]).pow(torch.arange(k+1,traj_len))

                # Replace reward in last time step with its Value estimate
                bootstrapped_rewards = torch.cat([exps.reward[k+1:traj_len-1],
                                                  exps.value[-1].view(1)])
                hca_factor = discount_factor.unsqueeze(1) \
                             * bootstrapped_rewards.unsqueeze(1) \
                             * hca_prob / pi_dist.probs
                # hca_factor is size (traj_len - k + 1) x num_actions

                # estimated immediate reward for all actions
                for a in range(n_actions):
                    ohe_action = F.one_hot(torch.as_tensor(a), n_actions).float()
                    _, _, est_reward = self.acmodel(exps.obs[k], action=ohe_action)
                    hca_factor[:, a] += est_reward.item() * self.discount**k

            # Policy loss

            pi_dist, _ = self.acmodel(exps.obs[k])
            # sum over all actions (dim=1) and all time step pairs (dim=0)
            policy_loss += (pi_dist.probs * hca_factor).sum()

            # State HCA cross entropy loss
            _, _, hca_logits = self.acmodel(exps.obs[k], obs2=exps.obs[k+1:traj_len])
            hca_loss += F.cross_entropy(hca_logits, exps.action[k+1:traj_len].long(), reduction='mean')

            k += 1

        # Estimated reward loss
        ohe_action = F.one_hot(exps.action.long(), n_actions).float()
        _, _, est_reward = self.acmodel(exps.obs, action=ohe_action)
        reward_loss = F.mse_loss(est_reward.squeeze(), exps.reward, reduction='mean')

        return policy_loss, hca_loss, reward_loss

    def update_parameters(self, exps):
        exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # Initialize update values

        update_policy_loss = 0
        update_hca_loss = 0
        update_reward_loss = 0

        # Get starting indexes for initial obs in each rollout
        start_indices, end_indices = self._get_indices(exps.mask)

        # Compute policy loss per episode
        for k, t in zip(start_indices, end_indices):
            policy_loss, hca_loss, reward_loss = self._policy_loss_for_episode(exps[k:t+1])
            update_policy_loss += policy_loss
            update_hca_loss += hca_loss
            update_reward_loss += reward_loss
        # Compute mean policy loss over all rollouts.  Change sign for update.
        update_policy_loss /= -1 * len(start_indices)

        dist, value = self.acmodel(exps.obs)

        # TODO: take this out? original HCA algo doesn't include entropy
        entropy = dist.entropy().mean()

        value_loss = (value - exps.returnn).pow(2).mean()

        # TODO: use a separate hca_loss_coef for hca and reward losses
        loss = update_policy_loss - self.entropy_coef * entropy \
               + self.value_loss_coef * value_loss \
               + self.value_loss_coef * update_hca_loss \
               + self.value_loss_coef * update_reward_loss

        # Update actor-critic

        self.optimizer.zero_grad()
        loss.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log some values

        logs = {
            "entropy": entropy.item(),
            "value": value.mean().item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "grad_norm": update_grad_norm,
            #"hca_loss": hca_loss.item()
        }

        return logs

    def _get_indices(self, masks):
        """Gives the indices of the first observation in each rollout.

        Not including the very first observation in exps.obs, the indices
        are the indices of the zeros (denoting done) in the mask, shifted by 1.

        Returns
        -------
        start_indices: list of ints corresponding to start idx of each rollout
        end_indices: list of ints corresponding to end idx of each rollout
        """
        end_indices = torch.where(masks==0)[0].cpu().numpy()
        start_indices = end_indices + 1

        # include index of final element in end_indices
        end_indices = numpy.concatenate([end_indices, numpy.array([len(masks)])])

        # include index of first element in start
        start_indices = numpy.concatenate([numpy.array([0]), start_indices])
        return start_indices, end_indices
