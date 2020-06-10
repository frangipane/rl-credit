from gym_minigrid.envs.delayed_reward_multiphase import ThreePhaseDelayedReward
from gym_minigrid.envs.opengifts import GiftsEnv
from gym_minigrid.envs.goalkeyoptional import GoalKeyOptionalEnv


# gamma = 0.99
DISCOUNT_FACTOR = 0.99
DISCOUNT_TIMESCALE = 100
STEPS_IN_PHASE1 = 30
STEPS_IN_PHASE3 = 70


class KeyGiftsGoalBaseEnv(ThreePhaseDelayedReward):

    def __init__(self, distractor_kwargs):
        super().__init__(
            key_kwargs=dict(
                size=6,
                key_color='yellow',
                start_by_key=False,
                max_steps=STEPS_IN_PHASE1,
                done_when_fetched=False,
            ),
            distractor_kwargs=distractor_kwargs,
            distractor_env=GiftsEnv,
            delayed_reward_env=GoalKeyOptionalEnv,
            delayed_reward_kwargs=dict(
                size=7,
                max_steps=STEPS_IN_PHASE3,
                goal_reward=5.,
                key_reward=15.,
                done_when_goal_reached=False,
            ),
            key_teleports_to_end_only=True
        )


class VaryGiftsGoalEnv(KeyGiftsGoalBaseEnv):

    def __init__(self, distractor_xtra_kwargs):

        assert 'max_steps' in distractor_xtra_kwargs.keys()

        distractor_kwargs=dict(
            size=8,
            num_objs=4,
            gift_reward=3,
            max_steps=0.5*DISCOUNT_TIMESCALE,
            done_when_all_opened=False,
        )
        distractor_kwargs.update(distractor_xtra_kwargs)

        super().__init__(distractor_kwargs=distractor_kwargs)
