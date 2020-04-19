import torch
import numpy as np
import toml
import tqdm
import argparse
import time
import shutil
import os

from envs import make_env
from utils import load_toml_config, create_save_dir, lineplot
from memory import ReplayBuffer
import agents


MOVING_AVG_COEF = 0.1



# Set random variable
np.random.seed(int(time.time()))
torch.manual_seed(int(time.time()))
if torch.cuda.is_available():
    torch.cuda.manual_seed(int(time.time()))



def train(config_filepath, save_dir, device, visualize_interval):
    conf = load_toml_config(config_filepath)
    data_dir, log_dir = create_save_dir(save_dir)
    # Save config file
    shutil.copyfile(config_filepath, os.path.join(save_dir, os.path.basename(config_filepath)))
    device = torch.device(device)

    # Set up log metrics
    metrics = {
        'episode': [],
        'episodic_step': [], 'collected_total_samples': [], 'reward': [],
        'q_loss': [], 'policy_loss': [], 'alpha_loss':[], 'alpha': [],
        'policy_switch_epoch': [], 'policy_switch_sample': [], 
        'test_episode': [], 'test_reward': [],
    }

    policy_switch_samples = conf.policy_switch_samples if hasattr(conf, "policy_switch_samples") else None
    total_collected_samples = 0

    # Create environment
    env = make_env(conf.environment, render=False)

    # Instantiate modules
    memory = ReplayBuffer(int(conf.replay_buffer_capacity), env.observation_space.shape, env.action_space.shape)
    agent = getattr(agents, conf.agent_type)(env.observation_space, env.action_space, device=device, **conf.agent)

    # Load checkpoint if specified in config
    if conf.checkpoint != '':
        ckpt = torch.load(conf.checkpoint, map_location=device)
        metrics = ckpt['metrics']
        agent.load_state_dict(ckpt['agent'])
        memory.load_state_dict(ckpt['memory'])
        policy_switch_samples = ckpt['policy_switch_samples']
        total_collected_samples = ckpt['total_collected_samples']

    def save_checkpoint():
        # Save checkpoint
        ckpt = {
            'metrics': metrics, 'agent': agent.state_dict(), 'memory': memory.state_dict(),
            'policy_switch_samples': policy_switch_samples, 'total_collected_samples': total_collected_samples
        }
        path = os.path.join(data_dir, 'checkpoint.pth')
        torch.save(ckpt, path)

        # Save agent model only
        model_ckpt = {'agent': agent.state_dict()}
        model_path = os.path.join(data_dir, 'model.pth')
        torch.save(model_ckpt, model_path)

        # Save metrics only
        metrics_ckpt = {'metrics': metrics}
        metrics_path = os.path.join(data_dir, 'metrics.pth')
        torch.save(metrics_ckpt, metrics_path)


    # Train agent
    init_episode = 0 if len(metrics['episode']) == 0 else metrics['episode'][-1] + 1
    pbar = tqdm.tqdm(range(init_episode, conf.episodes))
    reward_moving_avg = None
    agent_update_count = 0
    for episode in pbar:
        episodic_reward = 0
        o = env.reset()
        q1_loss, q2_loss, policy_loss, alpha_loss, alpha = None, None, None, None, None

        for t in range(conf.horizon):
            if total_collected_samples <= conf.random_sample_num:  # Select random actions at the begining of training.
                h = env.action_space.sample()
            elif memory.step <= conf.random_sample_num:  # Select actions from random latent variable soon after inserting a new subpolicy.
                h = agent.select_action(o, random=True)
            else:
                h = agent.select_action(o)

            a = agent.post_process_action(o, h)  # Convert abstract action h to actual action a

            o_next, r, done, _  = env.step(a)
            total_collected_samples += 1
            episodic_reward += r
            memory.push(o, h, r, o_next, done)
            o = o_next

            if memory.step > conf.random_sample_num:
                # Update agent
                batch_data = memory.sample(conf.agent_update_batch_size)
                q1_loss, q2_loss, policy_loss, alpha_loss, alpha = agent.update_parameters(batch_data, agent_update_count)
                agent_update_count += 1

            if done:
                break

        # Describe and save episodic metrics
        reward_moving_avg = (1. - MOVING_AVG_COEF) * reward_moving_avg + MOVING_AVG_COEF * episodic_reward if reward_moving_avg else episodic_reward
        pbar.set_description("EPISODE {} (total samples {}, subpolicy samples {}) --- Step {}, Reward {:.1f} (avg {:.1f})".format(episode, total_collected_samples, memory.step, t, episodic_reward, reward_moving_avg))
        metrics['episode'].append(episode)    
        metrics['reward'].append(episodic_reward)
        metrics['episodic_step'].append(t)
        metrics['collected_total_samples'].append(total_collected_samples)
        if episode % visualize_interval == 0:
            # Visualize metrics
            lineplot(metrics['episode'][-len(metrics['reward']):], metrics['reward'], 'REWARD', log_dir)
            reward_avg = np.array(metrics['reward']) / np.array(metrics['episodic_step'])
            lineplot(metrics['episode'][-len(reward_avg):], reward_avg, 'AVG_REWARD', log_dir)
            lineplot(metrics['collected_total_samples'][-len(metrics['reward']):], metrics['reward'], 'SAMPLE-REWARD', log_dir, xaxis='sample')

        # Save metrics for agent update
        if q1_loss is not None:
            metrics['q_loss'].append(np.mean([q1_loss, q2_loss]))
            metrics['policy_loss'].append(policy_loss)
            metrics['alpha_loss'].append(alpha_loss)
            metrics['alpha'].append(alpha)
            if episode % visualize_interval == 0:
                lineplot(metrics['episode'][-len(metrics['q_loss']):], metrics['q_loss'], 'Q_LOSS', log_dir)
                lineplot(metrics['episode'][-len(metrics['policy_loss']):], metrics['policy_loss'], 'POLICY_LOSS', log_dir)
                lineplot(metrics['episode'][-len(metrics['alpha_loss']):], metrics['alpha_loss'], 'ALPHA_LOSS', log_dir)
                lineplot(metrics['episode'][-len(metrics['alpha']):], metrics['alpha'], 'ALPHA', log_dir)



        # Insert new subpolicy layer and reset memory if a specific amount of samples is collected
        if policy_switch_samples and len(policy_switch_samples) > 0 and total_collected_samples >= policy_switch_samples[0]:
            print("----------------------\nInser new policy\n----------------------")
            agent.insert_subpolicy()
            memory.reset()
            metrics['policy_switch_epoch'].append(episode)
            metrics['policy_switch_sample'].append(total_collected_samples)
            policy_switch_samples = policy_switch_samples[1:]

    

        # Test a policy
        if episode % conf.test_interval == 0:
            test_rewards = []
            for _ in range(conf.test_times):
                episodic_reward = 0
                obs = env.reset()
                for t in range(conf.horizon):
                    h = agent.select_action(obs, eval=True)
                    a = agent.post_process_action(o, h)
                    obs_next, r, done, _  = env.step(a)
                    episodic_reward += r
                    obs = obs_next

                    if done:
                        break

                test_rewards.append(episodic_reward)

            test_reward_avg, test_reward_std = np.mean(test_rewards), np.std(test_rewards)
            print("   TEST --- ({} episodes) Reward {:.1f} (pm {:.1f})".format(conf.test_times, test_reward_avg, test_reward_std))
            metrics['test_episode'].append(episode)
            metrics['test_reward'].append(test_rewards)
            lineplot(metrics['test_episode'][-len(metrics['test_reward']):], metrics['test_reward'], "TEST_REWARD", log_dir)
            

        # Save checkpoint
        if episode % conf.checkpoint_interval:
            save_checkpoint()

    # Save the final model
    torch.save({'agent': agent.state_dict()}, os.path.join(data_dir, 'final_model.pth'))



    
def evaluate(config_filepath: str, model_filepath: str, render: bool):
    conf = load_toml_config(config_filepath)
    env = make_env(conf.environment, render=render)
    agent = getattr(agents, conf.agent_type)(env.observation_space.shape[0], env.action_space.shape[0], **conf.agent)
    ckpt = torch.load(model_filepath, map_location='cpu')
    agent.load_state_dict(ckpt['agent'])

    o = env.reset()
    if render:
        env.render()
    done = False
    episode_reward = 0
    t = 0
    while not done:
        h = agent.select_action(o, eval=True)
        a = agent.post_process_action(o, h)
        o_next, r, done, _ = env.step(a)
        episode_reward += r
        o = o_next
        t += 1

    print("STEPS: {}, REWARD: {}".format(t, episode_reward))
    input("OK? >")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test trying a walker.')
    parser.add_argument('--config', default='default_config.toml', help='Config file path')
    parser.add_argument('--save-dir', default=os.path.join('results', 'test'), help='Save directory')
    parser.add_argument('--visualize-interval', default=25, type=int, help='Interval to draw graphs of metrics.')
    parser.add_argument('--device', default='cpu', choices={'cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'}, help='Device for computation.')
    parser.add_argument('-e', '--eval', action='store_true', help='Run model evaluation.')
    parser.add_argument('-m', '--model-filepath', default='', help='Path to trained model for evaluation.')
    parser.add_argument('-r', '--render', action='store_true', help='Render agent behavior during evaluation.')
    args = parser.parse_args()

    if args.eval:
        evaluate(args.config, args.model_filepath, args.render)
    else:
        train(args.config, args.save_dir, args.device, args.visualize_interval)
