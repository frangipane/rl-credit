from gym.envs.registration import register


register(
    id='GiftDistractorDelay0-v0',
    entry_point='rl_credit.examples.distractor_delay_expt:Delay0_Gifts'
)

register(
    id='GiftDistractorDelay0_5-v0',
    entry_point='rl_credit.examples.distractor_delay_expt:Delay0_5_Gifts'
)

register(
    id='GiftDistractorDelay1-v0',
    entry_point='rl_credit.examples.distractor_delay_expt:Delay1_Gifts'
)

register(
    id='GiftDistractorReward0-v0',
    entry_point='rl_credit.examples.distractor_mean_expt:Reward0_Gifts'
)

register(
    id='GiftDistractorReward1-v0',
    entry_point='rl_credit.examples.distractor_mean_expt:Reward1_Gifts'
)

register(
    id='GiftDistractorReward5-v0',
    entry_point='rl_credit.examples.distractor_mean_expt:Reward5_Gifts'
)
