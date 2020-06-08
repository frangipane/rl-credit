"""
Vanilla A2C (with or without LSTM memory) with a separate Qvalue model using an attention
layer that takes in image embedding or hidden state from the A2C model, as well as a
 one hot encoded action as input.
"""
import os
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F

from rl_credit.utils import DictList, ParallelEnv
import rl_credit.script_utils as utils

from rl_credit.algos.base import BaseAlgo
from rl_credit.model import QAttentionModel


class AttentionQAlgo(BaseAlgo):
    """The Advantage Actor-Critic algorithm with a separate attention Qvalue model"""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01,
                 gae_lambda=0.95, entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5,
                 recurrence=4, rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None,
                 reshape_reward=None, plots_dir=None, d_key=30, use_tvt=True,
                 importance_threshold=0.05, tvt_alpha=0.9, y_moving_avg_alpha=0.1, pos_weight=2,
                 embed_actions=False):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                         store_embeddings=True)

        # TODO: allow user to pass in QAttentionModel instance and kwargs, e.g. in training script
        embed_size = self.acmodel.semi_memory_size
        self.qmodel = QAttentionModel(embedding_size=embed_size,
                                      action_size=7,
                                      d_key=d_key,
                                      embed_actions=embed_actions)
        self.qmodel.to(device)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

        # TODO: allow a separate learning rate for qvalue
        self.qvalue_optimizer = torch.optim.RMSprop(self.qmodel.parameters(), lr,
                                                    alpha=rmsprop_alpha, eps=rmsprop_eps)

        self.use_tvt = use_tvt
        self.importance_threshold = importance_threshold  # Must be b/w 0 and 1.
        self.tvt_alpha = tvt_alpha                        # tvt reward multiplier
        self.y_moving_avg_alpha = y_moving_avg_alpha      # higher discounts older obs faster
        self.pos_weight = torch.tensor([pos_weight])      # Weight for positive class in binary CE

        self.y_max_return = 0.
        self._update_number = 0  # convenience, for debugging, occasional saves
        self.plots_dir = plots_dir

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

        # Update max returns (moving average) seen in training run
        max_return = max(logs['return_per_episode'])  # undiscounted returns
        self.y_max_return += self.y_moving_avg_alpha * (max_return - self.y_max_return)

        # ===== Use most highly attended observations for TVT =====

        _, scores, self.attn_mask, self.attn_obss, self.attn_actions = self.get_attention_scores()

        # Calculate score importances (just average score)
        su_scores, su_unmasked, self.importances = self.score_importance(scores, self.attn_mask)

        # Get importances and indices for all obs with score above threshold
        importance_mask = self.importances > self.importance_threshold  # (num_procs, frames_per_proc)
        self.top_imp = self.importances[importance_mask]
        self.top_imp_idxs = torch.nonzero(importance_mask)

        top_rew2go = self.get_top_rew2go(self.top_imp_idxs)

        # TVT: add undiscounted rewards-to-go, excluding rewards accumulated during discount time
        # scale, to the rewards of the most important obs. Recalculate advantages with these
        # new rewards.
        if len(self.top_imp) > 0 and self.use_tvt:
            # logging
            tvt_rewards = []
            modified_rewards = self.rewards.detach().clone()

            for idx, weight, tvt_val in zip(self.top_imp_idxs, self.top_imp, top_rew2go):
                proc, frame = idx
                tvt_reward = tvt_val * self.tvt_alpha * weight
                modified_rewards[frame, proc] += tvt_reward
                tvt_rewards.append(tvt_reward.item())

            self.calculate_advantages(rewards=modified_rewards)
            exps.advantage = self.advantages.transpose(0,1).reshape(-1)
            exps.advantage = (exps.advantage - exps.advantage.mean())/exps.advantage.std()

            tvt_rewards = np.array(tvt_rewards)
            logs.update({'tvt_reward_max': tvt_rewards.max(),
                         'tvt_reward_mean': tvt_rewards.mean(),
                         'tvt_reward_min': tvt_rewards.min(),
            })

        # Log some values
        if len(self.top_imp) > 0:
            logs.update({'top_scores_max': self.top_imp.max().item(),
                         'top_scores_mean': self.top_imp.mean().item(),
                         'top_scores_min': self.top_imp.min().item(),
                         'top_rew2go_max': top_rew2go.max().item(),
                         'top_rew2go_mean': top_rew2go.mean().item(),
                         'top_rew2go_min': top_rew2go.min().item(),
            })

        logs.update({'return_classifier_thresh': self.y_max_return,
                     'num_obs_over_tvt_thresh': len(self.top_imp),
        })

        return exps, logs

    def score_importance(self, scores, masks):
        """
        Calculate importance as average score per column (excluding masked regions).

        scores : tensor (batch_sz, seq_len, seq_len)
        masks : tensor (batch_sz, seq_len, seq_len)

        Returns
        -------
        summed_scores : tensor (batch_sz, seq_len), sum scores
           per batch along columns ("total attendance" w/in an episode)

        summed_unmasked : tensor (batch_sz, seq_len), counts, per batch, of
           unmasked size along columns, used to calculate average attention score

        importances : tensor (batch_sz, seq_len), average score, per batch,
           per column.  Average excludes masked regions, which encompass future obs
           and obs not in the same episode (a block diagonal with triangle mask)
        """
        seq_len = scores.shape[1]

        # sum along columns per batch -> P x T
        summed_scores = torch.sum(scores, dim=1).squeeze(1)

        # Count number of non-zero entries along columns -> P x T.
        # Since mask doesn't include mask from upper triangle, combine
        # them into total mask here.
        future_mask = torch.ones([seq_len, seq_len], device=self.device).tril()
        total_unmasked = ~(masks | (future_mask.expand_as(masks) == 0))
        summed_unmasked = torch.sum(total_unmasked, dim=1)

        importances = torch.div(summed_scores, summed_unmasked)

        return summed_scores, summed_unmasked, importances

    def get_attention_scores(self):
        """
        Classify obs and get attention scores from the classifier network.
        Classifier is a binary classifier that tries to predict if returns for an obs
        are greater than a threshold.
        """
        # if self.acmodel.recurrent:
        #     # Concat image embedding with hidden state
        #     # Reshape embeddings -> (num_procs, frames_per_proc, *(2*embedding_size))
        #     # T x P x (2D)
        #     self.attn_obss = torch.cat((self.embeddings,
        #                                 self.memories[:, :, :self.acmodel.semi_memory_size]), 2)
        # else:
        attn_obss = self.embeddings

        # P X T X D
        attn_obss = attn_obss.transpose(0, 1)

        # Reshape actions -> tensor size (num_procs, frames_per_proc, action_space.n)
        # T x P x 1 -> P x T x action_space.n
        attn_actions = (torch.nn.functional.one_hot(self.actions.long(), 7)
                        .transpose(0, 1).float())

        # Create block diagonal mask so observations from different
        # episodes don't pay attention to each other.
        # T x P -> P x T -> P x 1 x T -> P x T x T
        seq_labels = (self.seq_labels
                      .transpose(0, 1)
                      .unsqueeze(1)
                      .expand(-1, self.num_frames_per_proc, -1))
        # mask picks out elements outside the block diagonal to be masked out
        attn_mask = (seq_labels - seq_labels.transpose(2, 1)) != 0

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
            qvalue, scores = self.qmodel(obs=attn_obss,
                                         act=attn_actions,
                                         mask_future=True,
                                         custom_mask=attn_mask)

        return qvalue, scores, attn_mask, attn_obss, attn_actions

    def get_top_rew2go(self, top_idxs):
        """
        Return rewards-to-go, excluding near term rewards (with timescale set
        by the discount factor).

        idxs: tensor, shape (N, 2), where N is the number
            of obs over the importance threshold
        """
        try:
            discount_t = int(np.round(1/(1 - self.discount)))  # discount factor timescale
        except ZeroDivisionError:
            raise ValueError("Discount factor must be less than 1.0 to use tvt")

        top_rew2go = torch.zeros(top_idxs.shape[0], device=self.device)

        for i, idx in enumerate(top_idxs):
            proc, frame = idx
            future_frame =  min(frame + discount_t, self.num_frames_per_proc - 1)

            seq_for_obs = self.seq_labels[frame, proc]
            seq_discount_t_steps_in_future = self.seq_labels[future_frame, proc]
            future_is_same_episode = (seq_for_obs == seq_discount_t_steps_in_future)

            top_rew2go[i] = self.rewards_togo[future_frame, proc] * future_is_same_episode
        return top_rew2go

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

        y_target = (exps.rewards_togo > self.y_max_return).float().unsqueeze(1).to(self.device)
        pos_weight = torch.tensor([self.pos_weight], device=self.device)
        qvalue_loss = F.binary_cross_entropy_with_logits(qvalue, y_target, pos_weight=pos_weight)

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
        if self.plots_dir is not None and self._update_number % 100 == 0:
            self.save_attention_plots(scores)
            self.save_top_attended_obs(k=10)

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
            "num_returns_above_thresh": y_target.sum().item(),
            "frac_returns_above_thresh": y_target.sum().item()/len(y_target)
        })
        return logs

    def save_attention_plots(self, scores):
        # Importance scores (averaged attention weights)
        attn_fig = sns.heatmap(self.importances.cpu().numpy()).get_figure()
        self._save_fig(attn_fig, self.plots_dir, f'importances_heatmap_{self._update_number:04}')

        # Raw (unaveraged) scores for 0th batch (proc=0)
        scores0 = scores[0].detach()
        attn_fig = sns.heatmap(scores0.cpu().numpy(), xticklabels=15, yticklabels=15).get_figure()
        self._save_fig(attn_fig, self.plots_dir, f'attn_scores_{self._update_number:04}')

        # Within 0th batch inter-episode mask
        mask_fig = (sns.heatmap(self.attn_mask[0].detach().cpu().numpy(), xticklabels=15, yticklabels=15)
                    .get_figure())
        self._save_fig(mask_fig, self.plots_dir, f'mask_{self._update_number:04}')

        # Histogram of importances
        fig, ax = plt.subplots()
        plt.hist(self.top_imp.flatten().cpu().numpy(), bins=50, label='top', alpha=0.5)
        ax.set_title('Top importance scores')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Counts')
        self._save_fig(fig, self.plots_dir, f'top_importances_hist_{self._update_number:04}.png')

    def save_top_attended_obs(self, k=10):
        """Save top k most important obs"""
        map_actions = {0: 'left',
                       1: 'right',
                       2: 'forward',
                       3: 'pickup',
                       4: 'drop',
                       5: 'toggle',
                       6: 'done'}

        sorted_top_imp, sorted_idxs = torch.sort(self.top_imp, descending=True)

        for i in range(min(k, len(self.top_imp))):
            proc, frame = self.top_imp_idxs[sorted_idxs[i]]
            score = str(sorted_top_imp[i].cpu().numpy().round(2))
            act = map_actions[self.actions[frame, proc].item()]
            fname = f'obs_{self._update_number:04}_proc{proc}_fr{frame:03}_score{score}__{act}.png'

            self._save_obs(self.obss_mat[proc, frame].cpu().numpy(),
                           self.plots_dir,
                           fname)

    @staticmethod
    def _save_fig(fig, out_dir, fname):
        dest = str(os.path.join(out_dir, fname))
        fig.savefig(dest, fmt='png')
        plt.clf()

    @staticmethod
    def _save_obs(obs, out_dir, fname, tile_size=12):
        """
        Render an agent observation and save as image
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
        plt.imsave(os.path.join(out_dir, fname), img)
        plt.clf()

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
