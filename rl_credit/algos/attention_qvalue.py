"""
Vanilla A2C (with or without LSTM memory) with a separate Qvalue model using an attention
layer that takes in image embedding or hidden state from the A2C model, as well as a
 one hot encoded action as input.
"""
import os
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from rl_credit.utils import DictList, ParallelEnv
import rl_credit.script_utils as utils

from rl_credit.algos.base import BaseAlgo
from rl_credit.model import QAttentionModel


class AttentionQAlgo(BaseAlgo):
    """The Advantage Actor-Critic algorithm with a separate attention Qvalue model"""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None, reshape_reward=None,
                 wandb_dir=None, d_key=30):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                         store_embeddings=True)

        # TODO: allow user to pass in QAttentionModel instance and kwargs, e.g. in training script
        self.qmodel = QAttentionModel(embedding_size=self.acmodel.semi_memory_size,
                                      action_size=7,
                                      d_key=d_key)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

        # TODO: allow a separate learning rate for qvalue
        self.qvalue_optimizer = torch.optim.RMSprop(self.qmodel.parameters(), lr,
                                                    alpha=rmsprop_alpha, eps=rmsprop_eps)

        self._update_number = 0  # convenience, for debugging, occasional saves
        self.wandb_dir = wandb_dir

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
        # Step through env using A2C
        exps, logs = super().collect_experiences()

        # ===== Calculate Qvalues using attention (context from experiences) =====

        # Reshape embeddings -> tensor size (num_procs, frames_per_proc, *(embedding_size))
        # T x P x D -> P x T x D
        self.attn_obss = self.embeddings.transpose(0, 1)

        # Reshape actions -> tensor size (num_procs, frames_per_proc, action_space.n)
        # T x P x 1 -> P x T x action_space.n
        self.attn_actions = (torch.nn.functional.one_hot(self.actions.long(), 7)
                            .transpose(0, 1).float())

        # Create block diagonal mask so observations from different
        # episodes don't pay attention to each other.
        # T x P -> P x T -> P x 1 x T -> P x T x T
        seq_labels = (self.seq_labels
                      .transpose(0, 1)
                      .unsqueeze(1)
                      .expand(-1, self.num_frames_per_proc, -1))
        # mask picks out elements outside the block diagonal to be masked out
        self.attn_mask = (seq_labels - seq_labels.transpose(2, 1)) != 0

        # just for debugging
        self.seq_labels_debug = seq_labels
        obss_mat = [None]*(self.num_procs)
        for i in range(self.num_procs):
            obss_mat[i] = torch.tensor(np.array([self.obss[j][i]["image"] for j
                                                 in range(self.num_frames_per_proc)]),
                                       device=self.device,
                                       dtype=float)
        self.obss_mat = torch.cat(obss_mat).view(self.num_procs, *obss_mat[0].shape)
        # self.obss_mat is size (batch size=num_procs, seq len=frames per proc, *(image_dim))

        with torch.no_grad():
            qvalue, scores = self.qmodel(obs=self.attn_obss,
                                         act=self.attn_actions,
                                         mask_future=True,
                                         custom_mask=self.attn_mask)
        # TODO: use qvalues and scores to modify advantage for TVT (not yet implemented)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        return exps, logs

    def update_parameters(self, exps):
        self._update_number += 1
        logs = {}

        # ===== A2C loss =====

        # Compute starting indexes
        inds = self._get_starting_indexes()

        # Initialize update values
        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0

        # Initialize memory

        if self.acmodel.recurrent:
            memory = exps.memory[inds]

        for i in range(self.recurrence):
            # Create a sub-batch of experience

            sb = exps[inds + i]

            # Compute loss

            if self.acmodel.recurrent and not self.store_embeddings:
                dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
            elif self.acmodel.recurrent and self.store_embeddings:
                dist, value, memory, _ = self.acmodel(sb.obs, memory * sb.mask)
            else:
                dist, value = self.acmodel(sb.obs)

            entropy = dist.entropy().mean()

            policy_loss = -(dist.log_prob(sb.action) * sb.advantage).mean()

            value_loss = (value - sb.returnn).pow(2).mean()

            loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

            # Update batch values

            update_entropy += entropy.item()
            update_value += value.mean().item()
            update_policy_loss += policy_loss.item()
            update_value_loss += value_loss.item()
            update_loss += loss

        # Update update values

        update_entropy /= self.recurrence
        update_value /= self.recurrence
        update_policy_loss /= self.recurrence
        update_value_loss /= self.recurrence
        update_loss /= self.recurrence

        # ===== Qvalue loss =====

        qvalue, scores = self.qmodel(obs=self.attn_obss,
                                     act=self.attn_actions,
                                     mask_future=True,
                                     custom_mask=self.attn_mask)

        qvalue_loss = (qvalue - exps.returnn).pow(2).mean()

        # Update actor-critic

        self.optimizer.zero_grad()
        update_loss.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Update Qvalue

        self.qvalue_optimizer.zero_grad()
        qvalue_loss.backward()
        update_grad_norm_q = sum(p.grad.data.norm(2) ** 2 for p in self.qmodel.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.qmodel.parameters(), self.max_grad_norm)
        self.qvalue_optimizer.step()

        # Log some values

        # Save attention scores heatmap every 100 updates
        if self.wandb_dir is not None and self._update_number % 100 == 0:
            import wandb
            import seaborn as sns

            scores0 = scores[0].detach()
            attn_fig = sns.heatmap(scores0.numpy(), xticklabels=10, yticklabels=10).get_figure()
            img_name_base = str(os.path.join(self.wandb_dir,
                                             f'attn_scores_{self._update_number:04}'))
            attn_fig.savefig(img_name_base, fmt='png')
            wandb.save(img_name_base + '*')
            plt.clf()

            # # For debugging
            # labels_fig = (sns.heatmap(self.seq_labels_debug[0].detach().numpy(), xticklabels=10, yticklabels=10)
            #               .get_figure())
            # labels_fig_base = str(os.path.join(self.wandb_dir,
            #                                    f'episode_labels_{self._update_number:04}'))
            # labels_fig.savefig(labels_fig_base, fmt='png')
            # plt.clf()

            mask_fig = (sns.heatmap(self.attn_mask[0].detach().numpy(), xticklabels=10, yticklabels=10)
                        .get_figure())
            mask_fig_base = str(os.path.join(self.wandb_dir,
                                             f'mask_{self._update_number:04}'))
            mask_fig.savefig(mask_fig_base, fmt='png')
            plt.clf()

            self.calculate_and_save_top_attended(scores0, self.obss_mat[0].numpy(),
                                                 self.actions.detach()[:, 0].squeeze(),
                                                 out_dir=self.wandb_dir)

        with torch.no_grad():
            # evaluate KL divergence b/w old and new policy
            # policy under newly updated model
            if self.acmodel.recurrent and not self.store_embeddings:
                dist, _, _ = self.acmodel(exps.obs, exps.memory * exps.mask)
            elif self.acmodel.recurrent and self.store_embeddings:
                dist, _, _, _ = self.acmodel(exps.obs, exps.memory * exps.mask)
            else:
                dist, _ = self.acmodel(exps.obs)

            approx_kl = (exps.log_prob - dist.log_prob(exps.action)).mean().item()
            adv_mean = exps.advantage.mean().item()
            adv_max = exps.advantage.max().item()
            adv_min = exps.advantage.min().item()
            adv_std = exps.advantage.std().item()

            # standard deviation of values
            value_std = value.std().item()

        logs.update({
            "entropy": update_entropy,
            "value": update_value,
            "value_max": exps.value.max().item(),
            "value_min": exps.value.min().item(),
            "value_std": exps.value.std().item(),
            "qvalue_mean": qvalue.mean().item(),
            "qvalue_min": qvalue.min().item(),
            "qvalue_max": qvalue.max().item(),
            "qvalue_std": qvalue.std().item(),
            "policy_loss": update_policy_loss,
            "value_loss": update_value_loss,
            "qvalue_loss": qvalue_loss.item(),
            "grad_norm": update_grad_norm.item(),
            "grad_norm_q": update_grad_norm_q.item(),
            "adv_max": adv_max,
            "adv_min": adv_min,
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "kl": approx_kl,
        })
        return logs

    def calculate_and_save_top_attended(self, scores, obs, actions, k=5, out_dir=None):
        """
        Plot the k-top attended observations as measured by summing attention scores
        across all frames.

        Also plot histogram of importance scores.

        scores : torch tensor (frames_per_proc, frames_per_proc)
        obs : torch tensor (frames_per_proc, *(image-size))
        actions : torch tensor (frames_per_proc)
        k : int, number of top most attended images to save
        """
        importance = torch.sum(scores, dim=0).squeeze(0)
        print('max importance', importance.max())

        _, indices = torch.sort(importance, descending=True)
        indices = indices[:k]
        map_actions = {0: 'left',
                       1: 'right',
                       2:'forward',
                       3: 'pickup',
                       4: 'drop',
                       5: 'toggle',
                       6: 'done'}

        fig = plt.hist(importance.numpy(), bins=30)
        plt.title('Importance from summed attention scores')
        plt.xlabel('Importance')
        plt.ylabel('Counts')
        plt.savefig(os.path.join(out_dir, f'update{self._update_number:04}_importance.png'))
        plt.clf()

        for idx in indices:
            img = self._get_obs_render(obs[idx])
            act = map_actions[actions[idx].item()]
            rnd_score = str(importance[idx].numpy().round(1))
            fname = f"update{self._update_number:04}__fr{idx:03}__score{rnd_score}__{act}.png"
            plt.imsave(os.path.join(out_dir, fname), img)
            plt.clf()

    def _get_obs_render(self, obs, tile_size=16):
        """
        Render an agent observation for visualization
        """
        from gym_minigrid.minigrid import Grid
        agent_view_size = obs.shape[0]
        grid, vis_mask = Grid.decode(obs)

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(agent_view_size // 2, agent_view_size - 1),
            agent_dir=3,
            highlight_mask=vis_mask
        )
        return img

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
        starting_indexes = np.arange(0, self.num_frames, self.recurrence)
        return starting_indexes
