import gym
import pybullet_envs


# Environments in pybullet
PYBULLET_ENVS = ["CartPoleBulletEnv", "CartPoleContinuousBulletEnv", "MinitaurBulletEnv", "MinitaurBulletDuckEnv", "RacecarGymEnv", "KukaGymEnv", "KukaCamGymEnv", "KukaDiverseObjectEnv"]
PYBULLET_ENVS_DEEPMIMIC = ["HumanoidDeepMimicBackflipBulletEnv", "HumanoidDeepMimicWalkBulletEnv"]
PYBULLET_ENVS_PENDULUM = ["InvertedPendulumBulletEnv", "InvertedDoublePendulumBulletEnv", "InvertedPendulumSwingupBulletEnv"]
PYBULLET_ENVS_MANIPULATOR = ["ReacherBulletEnv", "PusherBulletEnv", "ThrowerBulletEnv", "StrikerBulletEnv"]
PYBULLET_ENVS_LOCOMOTION = ["Walker2DBulletEnv", "HalfCheetahBulletEnv", "AntBulletEnv", "HopperBulletEnv", "HumanoidBulletEnv", "HumanoidFlagrunBulletEnv"]



def make_env(env_name, render=False):
    if env_name in PYBULLET_ENVS:
        from pybullet_envs import bullet
        env = getattr(bullet, env_name)(render=render)
        return env

    if env_name in PYBULLET_ENVS_DEEPMIMIC:
        from pybullet_envs.deep_mimic.gym_env import deep_mimic_env
        env = getattr(deep_mimic_env, env_name)(renders=render)
        return env

    if env_name in PYBULLET_ENVS_PENDULUM:
        from pybullet_envs import gym_pendulum_envs
        env = getattr(gym_pendulum_envs, env_name)()
        if render:
            env.render(mode='human')
        return env
    
    if env_name in PYBULLET_ENVS_MANIPULATOR:
        from pybullet_envs import gym_manipulator_envs
        env = getattr(gym_manipulator_envs, env_name)(render=render)
        return env

    if env_name in PYBULLET_ENVS_LOCOMOTION:
        from pybullet_envs import gym_locomotion_envs
        env = getattr(gym_locomotion_envs, env_name)(render=render)
        return env

    # Else
    env = gym.make(env_name)    
    return env
