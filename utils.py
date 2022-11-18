import gym
import numpy as np
import warnings
from atari_wrappers import wrap_atari, FrameBuffer


try:
    import envpool
except ImportError:
    envpool = None

def epsilon(t, config):
    '''
    Calculates exploration rate (epsilon) as a function of global step.
    Epsilon decays exponentially with initial value config.max_eps, 
    final value config.min_eps, and decay rate config.eps_decay
    '''
    return config.min_eps + (config.max_eps - config.min_eps)* np.exp(- (config.eps_decay) * t )


def make_env(config, clip_rewards=True, seed=None):
    '''
    Creates gym environment wrapped in some atari wrappers.
    '''
    env = gym.make(config.envname)
    if seed is not None:
        env.seed(seed)
    env = wrap_atari(env, clip_rewards)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    return env


def make_atari_env(config):
    if envpool is not None:
        if config.scale == 0:
            warnings.warn(
                "EnvPool does not include ScaledFloatFrame wrapper, "
                "please set `x = x / 255.0` inside CNN network's forward function."
            )
        # parameters convertion
        env = envpool.make_gym(
            config.envname,
            num_envs=config.training_num,
            seed=config.seed,
            episodic_life=True,
            reward_clip=True,
            stack_num=config.frames_stack,
        )
        # test_envs = envpool.make_gym(
        #     config.envname,
        #     num_envs=config.test_num,
        #     seed=config.seed,
        #     episodic_life=False,
        #     reward_clip=False,
        #     stack_num=config.frames_stack,
        # )
        env.seed(config.seed)
        # train_envs.seed(config.seed)
        # test_envs.seed(config.seed)
    return env