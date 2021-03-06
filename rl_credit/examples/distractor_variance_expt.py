import numpy as np
import gym
import wandb

from rl_credit.examples.environment import (
    DISCOUNT_FACTOR,
    VaryGiftsGoalEnv,
)
from rl_credit.examples.train import train


DISCOUNT_TIMESCALE = int(np.round(1/(1 - DISCOUNT_FACTOR)))
DELAY_STEPS = 0.5 * DISCOUNT_TIMESCALE


####################################################
# Environments: Variance of reward in distractor phase
# Mean reward is 5 (but default original was 3).


class Var0_Gifts(VaryGiftsGoalEnv):
    """variance=0"""
    def __init__(self):
        distractor_xtra_kwargs = {'max_steps': DELAY_STEPS, 'gift_reward': [5, 5]}
        super().__init__(distractor_xtra_kwargs)


class Var1_3_Gifts(VaryGiftsGoalEnv):
    """variance=1.33"""
    def __init__(self):
        distractor_xtra_kwargs = {'max_steps': DELAY_STEPS, 'gift_reward': [3, 7]}
        super().__init__(distractor_xtra_kwargs)


class Var8_3_Gifts(VaryGiftsGoalEnv):
    """variance=8.33"""
    def __init__(self):
        distractor_xtra_kwargs = {'max_steps': DELAY_STEPS, 'gift_reward': [0, 10]}
        super().__init__(distractor_xtra_kwargs)


####################################################
# Config params shared among all experiments

common_train_config = dict(
    num_procs=16,
    save_interval=300,
    total_frames=16*600*2500,  #24_000_000
    log_interval=1,
)

common_algo_kwargs = dict(
    num_frames_per_proc=600,
    discount=DISCOUNT_FACTOR,
    lr=0.001,
    gae_lambda=0.95,
    entropy_coef=0.01,
    value_loss_coef=0.5,
    max_grad_norm=0.5,
    rmsprop_alpha=0.99,
    rmsprop_eps=1e-8,
    reshape_reward=None,
)

mean_reward = 'mean_reward=5'
delay_steps = f'delay_steps={DELAY_STEPS}'


####################################################
# Experiment-specific configs

##************ experiment 1 ************
expt1a = dict(
    model_dir_stem='a2c_mem10_giftvar0',
    expt_train_config = dict(
        env_id='GiftDistractorVar0-v0',
        algo_name='a2c',
        recurrence=10,
    ),
    expt_algo_kwargs = {},
    distractor_var = 'gift_var=0',
    wandb_notes = 'A2C with recurrence=10, gift reward=5, gift var=0, delay=50 steps'
)

expt1b = dict(
    model_dir_stem='tvt_mem10_giftvar0',
    expt_train_config = dict(
        env_id='GiftDistractorVar0-v0',
        algo_name='tvt',
        recurrence=10,
    ),
    expt_algo_kwargs = dict(
        d_key=150,  # same as fixed episode len
        use_tvt=True,
        importance_threshold=0.15,
        tvt_alpha=0.5,
        y_moving_avg_alpha=0.03,
        pos_weight=2,
        embed_actions=True,
        mask_future=True,
    ),
    distractor_var = 'gift_var=0',
    wandb_notes = 'TVT, recurrence=10, d_key=150, action embed, gift reward=5, gift var=0, delay=50 steps'
)


##************ experiment 2 ************
expt2a = dict(
    model_dir_stem='a2c_mem10_giftvar1_3',
    expt_train_config = dict(
        env_id='GiftDistractorVar1_3-v0',
        algo_name='a2c',
        recurrence=10,
    ),
    expt_algo_kwargs = {},
    distractor_var = 'gift_var=1.33',
    wandb_notes = 'A2C with recurrence=10, gift reward=5, gift var=1.33 delay=50 steps'
)

expt2b = dict(
    model_dir_stem='tvt_mem10_giftvar1_3',
    expt_train_config = dict(
        env_id='GiftDistractorVar1_3-v0',
        algo_name='tvt',
        recurrence=10,
    ),
    expt_algo_kwargs = dict(
        d_key=150,
        use_tvt=True,
        importance_threshold=0.15,
        tvt_alpha=0.5,
        y_moving_avg_alpha=0.03,
        pos_weight=2,
        embed_actions=True,
        mask_future=True,
    ),
    distractor_var = 'gift_var=1.33',
    wandb_notes = 'TVT, recurrence=10, d_key=150, action embed, gift reward=5, gift var=1.33, delay=50 steps'
)


##************ experiment 3 ************
expt3a = dict(
    model_dir_stem='a2c_mem10_giftvar8_3',
    expt_train_config = dict(
        env_id='GiftDistractorVar8_3-v0',
        algo_name='a2c',
        recurrence=10,
    ),
    expt_algo_kwargs = {},
    distractor_var = 'gift_var=8.33',
    wandb_notes = 'A2C with recurrence=10, gift reward=5, gift var=8.33, delay=50 steps'
)

expt3b = dict(
    model_dir_stem='tvt_mem10_giftvar8_3',
    expt_train_config = dict(
        env_id='GiftDistractorVar8_3-v0',
        algo_name='tvt',
        recurrence=10,
    ),
    expt_algo_kwargs = dict(
        d_key=150,
        use_tvt=True,
        importance_threshold=0.15,
        tvt_alpha=0.5,
        y_moving_avg_alpha=0.03,
        pos_weight=2,
        embed_actions=True,
        mask_future=True,
    ),
    distractor_var = 'gift_var=8.33',
    wandb_notes = 'TVT, recurrence=10, d_key=150, action embed, gift reward=5, gift var=8.33, delay=50 steps'
)


def main(model_dir_stem, expt_train_config, expt_algo_kwargs, distractor_var,
         wandb_notes, seed):
    wandb_params = {}
    algo_kwargs = common_algo_kwargs
    algo_kwargs.update(expt_algo_kwargs)

    train_config = common_train_config
    expt_train_config['model_dir_stem'] = f"{model_dir_stem}_seed{seed}"
    train_config.update(expt_train_config)
    train_config['seed'] = seed

    train_config.update({'algo_kwargs': algo_kwargs})

    # expt run params to record in wandb
    wandb_params.update(train_config)
    wandb_params.update(common_algo_kwargs)
    wandb_params.update({'env_params': str(vars(gym.make(train_config['env_id'])))})
    wandb_name = f"{train_config['algo_name']}|mem={train_config['recurrence']}|{train_config['env_id']}"

    wandb.init(
        project="distractor_reward_variance",
        config=wandb_params,
        name=wandb_name,
        tags=[train_config['algo_name'],
              distractor_var,
              delay_steps,
              mean_reward,
              train_config['env_id'],
              f'discount_timescale={DISCOUNT_TIMESCALE}',
              f"discount_factor={DISCOUNT_FACTOR}"],
        notes=wandb_notes,
        reinit=True,
        group=wandb_name,
        job_type='training',
    )

    wandb_dir = wandb.run.dir
    train_config.update({'wandb_dir': wandb_dir})
    train(**train_config)

    wandb.join()


if __name__ == '__main__':
    expts = [expt1b]*5 + [expt2b]*5 + [expt3b]*5
    for i, expt in enumerate(expts):
        main(**expt, seed=i)
