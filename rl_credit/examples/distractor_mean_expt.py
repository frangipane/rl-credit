import numpy as np
import gym
import wandb

from rl_credit.examples.environment import (
    DISCOUNT_FACTOR,
    VaryGiftsGoalEnv,
)
from rl_credit.examples.train import train


DISCOUNT_TIMESCALE = int(np.round(1/(1 - DISCOUNT_FACTOR)))
DELAY_STEPS = 1.0 * DISCOUNT_TIMESCALE


####################################################
# Environments: Mean reward size in distractor phase (0 variance)


class Reward0_Gifts(VaryGiftsGoalEnv):
    def __init__(self):
        distractor_xtra_kwargs = {'max_steps': DELAY_STEPS, 'gift_reward': 0}
        super().__init__(distractor_xtra_kwargs)


class Reward1_Gifts(VaryGiftsGoalEnv):
    def __init__(self):
        distractor_xtra_kwargs = {'max_steps': DELAY_STEPS, 'gift_reward': 1}
        super().__init__(distractor_xtra_kwargs)


class Reward5_Gifts(VaryGiftsGoalEnv):
    def __init__(self):
        distractor_xtra_kwargs = {'max_steps': DELAY_STEPS, 'gift_reward': 5}
        super().__init__(distractor_xtra_kwargs)


class Reward8_Gifts(VaryGiftsGoalEnv):
    def __init__(self):
        distractor_xtra_kwargs = {'max_steps': DELAY_STEPS, 'gift_reward': 8}
        super().__init__(distractor_xtra_kwargs)


####################################################
# Config params shared among all experiments

common_train_config = dict(
    num_procs=16,
    save_interval=300,
    total_frames=16*600*2000,  #19_200_000
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

delay_steps = f'delay_steps={DELAY_STEPS}'


####################################################
# Experiment-specific configs

##************ experiment 1 ************
expt1a = dict(
    model_dir_stem='a2c_mem10_giftreward0',
    expt_train_config = dict(
        env_id='GiftDistractorReward0-v0',
        algo_name='a2c',
        recurrence=10,
    ),
    expt_algo_kwargs = {},
    # wandb metadata (tags, notes)
    distractor_reward = 'gift_reward=0',
    wandb_notes = 'A2C with recurrence=10, gift reward=0, delay=100 steps',
)

expt1b = dict(
    model_dir_stem='tvt_mem10_giftreward0',
    expt_train_config = dict(
        env_id='GiftDistractorReward0-v0',
        algo_name='tvt',
        recurrence=10,
    ),
    expt_algo_kwargs = dict(
        d_key=200,  # same as fixed episode len
        use_tvt=True,
        importance_threshold=0.15,
        tvt_alpha=0.5,
        y_moving_avg_alpha=0.03,
        pos_weight=2,
        embed_actions=True,
        mask_future=True,
    ),
    distractor_reward = 'gift_reward=0',
    wandb_notes = 'TVT, recurrence=10, d_key=200, action embed, gift reward=0, delay=100 steps'
)


##************ experiment 2 ************
expt2a = dict(
    model_dir_stem='a2c_mem10_giftreward1',
    expt_train_config = dict(
        env_id='GiftDistractorReward1-v0',
        algo_name='a2c',
        recurrence=10,
    ),
    expt_algo_kwargs = {},
    distractor_reward = 'gift_reward=1',
    wandb_notes = 'A2C with recurrence=10, gift reward=1, delay=100 steps',
)

expt2b = dict(
    model_dir_stem='tvt_mem10_giftreward1',
    expt_train_config = dict(
        env_id='GiftDistractorReward1-v0',
        algo_name='tvt',
        recurrence=10,
    ),
    expt_algo_kwargs = dict(
        d_key=200,
        use_tvt=True,
        importance_threshold=0.15,
        tvt_alpha=0.5,
        y_moving_avg_alpha=0.03,
        pos_weight=2,
        embed_actions=True,
        mask_future=True,
    ),
    distractor_reward = 'gift_reward=1',
    wandb_notes = 'TVT, recurrence=10, d_key=200, action embed, gift reward=1, delay=100 steps'
)


##************ experiment 3 ************
expt3a = dict(
    model_dir_stem='a2c_mem10_giftreward5',
    expt_train_config = dict(
        env_id='GiftDistractorReward5-v0',
        algo_name='a2c',
        recurrence=10,
    ),
    expt_algo_kwargs = {},
    distractor_reward = 'gift_reward=5',
    wandb_notes = 'A2C with recurrence=10, gift reward=5, delay=100 steps',
)

expt3b = dict(
    model_dir_stem='tvt_mem10_giftreward5',
    expt_train_config = dict(
        env_id='GiftDistractorReward5-v0',
        algo_name='tvt',
        recurrence=10,
    ),
    expt_algo_kwargs = dict(
        d_key=200,
        use_tvt=True,
        importance_threshold=0.15,
        tvt_alpha=0.5,
        y_moving_avg_alpha=0.03,
        pos_weight=2,
        embed_actions=True,
        mask_future=True,
    ),
    distractor_reward = 'gift_reward=5',
    wandb_notes = 'TVT, recurrence=10, d_key=200, action embed, gift reward=5, delay=100 steps'
)


##************ experiment 4 ************
expt4a = dict(
    model_dir_stem='a2c_mem10_giftreward8',
    expt_train_config = dict(
        env_id='GiftDistractorReward8-v0',
        algo_name='a2c',
        recurrence=10,
    ),
    expt_algo_kwargs = {},
    distractor_reward = 'gift_reward=8',
    wandb_notes = 'A2C with recurrence=10, gift reward=8, delay=100 steps',
)

expt4b = dict(
    model_dir_stem='tvt_mem10_giftreward8',
    expt_train_config = dict(
        env_id='GiftDistractorReward8-v0',
        algo_name='tvt',
        recurrence=10,
    ),
    expt_algo_kwargs = dict(
        d_key=200,
        use_tvt=True,
        importance_threshold=0.15,
        tvt_alpha=0.5,
        y_moving_avg_alpha=0.03,
        pos_weight=2,
        embed_actions=True,
        mask_future=True,
    ),
    distractor_reward = 'gift_reward=8',
    wandb_notes = 'TVT, recurrence=10, d_key=200, action embed, gift reward=8, delay=100 steps'
)


def main(model_dir_stem, expt_train_config, expt_algo_kwargs,
         distractor_reward, wandb_notes, seed):
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
        project="distractor_reward_size",
        config=wandb_params,
        name=wandb_name,
        tags=[train_config['algo_name'],
              distractor_reward,
              delay_steps,
              train_config['env_id'],
              f'discount_timescale={DISCOUNT_TIMESCALE}',
              f"discount_factor={DISCOUNT_FACTOR}",
              'phase3=fixed 70steps'],
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
    expts = [expt1b]*5 + [expt2b]*5 + [expt3b]*5 + [expt4b]*5
    for i, expt in enumerate(expts):
        main(**expt, seed=i)
