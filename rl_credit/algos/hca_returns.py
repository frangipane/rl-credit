import numpy
import torch
import torch.nn.functional as F

from rl_credit.algos.base import BaseAlgo


class HCAReturns(BaseAlgo):
    """The Advantage Actor-Critic algorithm."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=1,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None, reshape_reward=None):

        if recurrence != 1:
            raise ValueError("Memory is not supported for HCAReturns, recurrence must be 1.")
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)


    def update_parameters(self, exps):
        logs = {}

        # Compute loss
        dist, value, hca_logits = self.acmodel(exps.obs, exps.returnn)

        # Pick the HCA probability indexed by the selected action
        with torch.no_grad():
            hca_prob = torch.gather(F.softmax(hca_logits, dim=1),
                                    dim=1,
                                    index=exps.action.view(-1,1).long()).squeeze()
            pi_hca_ratio = torch.exp(exps.log_prob) / hca_prob
            hca_factor = (1 - pi_hca_ratio) * exps.returnn

            # for logging
            hca_mean = hca_factor.mean().item()
            hca_std = hca_factor.std().item()
            adv_mean = exps.advantage.mean().item()
            adv_max = exps.advantage.max().item()
            adv_min = exps.advantage.min().item()
            adv_std = exps.advantage.std().item()

            pearson_corr = ((hca_factor - hca_mean) * (exps.advantage - adv_mean)).mean() \
                           / (hca_std * adv_std)
            logs['hca_adv_corr'] = pearson_corr.item()
            logs['hca_prob_max'] = hca_prob.max().item()
            logs['hca_prob_min'] = hca_prob.min().item()
            logs['hca_prob_mean'] = hca_prob.mean().item()
            logs['pi_hca_ratio_max'] = pi_hca_ratio.max().item()
            logs['pi_hca_ratio_min'] = pi_hca_ratio.min().item()
            logs['pi_hca_ratio_mean'] = pi_hca_ratio.mean().item()

        entropy = dist.entropy().mean()

        # policy loss using hca factor
        #policy_loss = -(dist.log_prob(exps.action) * hca_factor).mean()
        policy_loss = -(dist.log_prob(exps.action) * exps.advantage).mean()

        value_loss = (value - exps.returnn).pow(2).mean()

        # Cross-entropy loss against action taken, cross_entropy expects
        # target to be of dtype long
        hca_loss = F.cross_entropy(hca_logits, exps.action.long(), reduction='mean')

        loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss \
               + self.value_loss_coef * hca_loss  # TODO: use a separate hca_loss_coef

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
            dist, _, = self.acmodel(exps.obs)
            approx_kl = (exps.log_prob - dist.log_prob(exps.action)).mean().item()

            # standard deviation of values
            value_std = value.std().item()

        logs.update({
            "entropy": entropy.item(),
            "value": value.mean().item(),
            "value_std": value_std,
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "grad_norm": update_grad_norm,
            "value_std": value_std,
            "hca_loss": hca_loss.item(),
            "hca_max": hca_factor.max().item(),
            "hca_min": hca_factor.min().item(),
            "hca_mean": hca_mean,
            "hca_std": hca_std,
            "adv_max": adv_max,
            "adv_min": adv_min,
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "kl": approx_kl,
        })

        return logs
