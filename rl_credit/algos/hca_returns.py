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
        # Initialize update values

        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_hca_loss = 0
        update_loss = 0

        # Compute loss
        dist, value, hca_logits = self.acmodel(exps.obs, exps.returnn)

        # Pick the HCA probability indexed by the selected action
        with torch.no_grad():
            hca_prob = torch.gather(F.softmax(hca_logits, dim=1),
                                    dim=1,
                                    index=exps.action.view(-1,1).long()).squeeze()
            hca_factor = (1 - torch.exp(exps.log_prob) / hca_prob ) * exps.returnn

        entropy = dist.entropy().mean()

        # policy loss using hca factor
        policy_loss = -(dist.log_prob(exps.action) * hca_factor).mean()
        #policy_loss = -(dist.log_prob(exps.action) * exps.advantage).mean()

        value_loss = (value - exps.returnn).pow(2).mean()

        # Cross-entropy loss against action taken, cross_entropy expects
        # target to be of dtype long
        hca_loss = F.cross_entropy(hca_logits, exps.action.long(), reduction='mean')

        loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss \
               + self.value_loss_coef * hca_loss  # TODO: use a separate hca_loss_coef

        # Update batch values

        update_entropy += entropy.item()
        update_value += value.mean().item()
        update_policy_loss += policy_loss.item()
        update_value_loss += value_loss.item()
        update_hca_loss += hca_loss.item()
        update_loss += loss

        # Update actor-critic

        self.optimizer.zero_grad()
        update_loss.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log some values

        logs = {
            "entropy": update_entropy,
            "value": update_value,
            "policy_loss": update_policy_loss,
            "value_loss": update_value_loss,
            "grad_norm": update_grad_norm
        }

        return logs