import torch
from MATD3 import MATD3
from Model import Actor, CriticTwin
from Parameters import AgentParams
from sandbox import DefendingSandbox, ParallelContainer
from utils import *
from itertools import count
import numpy as np
import tqdm
from functools import partial


def train():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 设置了训练使用的设备
    exp = ExperimentManager('./run')
    exp.start_experiment('./sandbox/configs/defending_td3.json')
    n_envs = exp.args['train/n_envs']
    num_agents = exp.args['sandbox/mission/num_agents']
    envs = ParallelContainer(n_envs, partial(DefendingSandbox, exp.args['sandbox']))

    train_param = exp.args['train']
    # env = gym.make('Pendulum-v1', render_mode='human')
    param = AgentParams()
    param.from_config(train_param)
    agent = MATD3(Actor, CriticTwin, num_agents, 16, 2, param)

    pool = ReplayBuffer(train_param['capacity'], n_envs, num_agents, (16, 2, 1, 1), (),
                        mode='off_policy',
                        use_advantage=False)
    ob, _ = envs.reset()
    ob = np.array(ob)
    a_loss_list, c_loss_list, r_list = [], [], []
    max_e = train_param['max_step']
    reward_sum = np.zeros([n_envs, num_agents])
    with tqdm.tqdm(total=max_e) as pbar:
        for c in count():
            if c >= max_e:
                break
            pbar.update(1)

            ob = torch.tensor(ob, dtype=torch.float32, device=device).reshape(n_envs, num_agents, -1)

            with torch.no_grad():
                action = agent.get_action(ob)
            if c < train_param['explore_step']:
                action += torch.randn_like(action)
            # u = action[:, 0].cpu().detach().numpy()
            u = action.cpu().detach().numpy() * np.array([1, 3])

            next_ob, rew, dones, truncation, infos = envs.step(u)
            reward_sum += rew

            for i, info in enumerate(infos):
                if '_final_info' in info.keys():
                    pbar.set_postfix({'reward': reward_sum[i]})
                    exp.add_scalars('reward', {'episode reward': np.mean(reward_sum[i])})
                    reward_sum[i] = np.zeros_like(rew[i])
            # next_ob = next_ob / np.array([1, 1, 8])

            next_ob = torch.tensor(next_ob, dtype=torch.float32, device=device).reshape(n_envs, num_agents, -1)
            action = action.reshape(n_envs, num_agents, -1).to(device)
            reward = torch.tensor(rew, dtype=torch.float32, device=device).reshape(n_envs, -1, 1)
            # print(reward)
            done_factor = torch.tensor(
                [[train_param['gamma']] * num_agents if not done else [0] * num_agents for done in dones],
                dtype=torch.float32,
                device=device).reshape(n_envs, num_agents, -1)

            pool.push(ob, action, reward, done_factor)

            show_dict = {}
            if len(pool) > train_param['batch_size']:
                for _ in range(train_param['repeat']):
                    data = pool.sample(train_param['batch_size'])

                    a_loss, c_loss = agent.train(data)
                    exp.add_scalars('loss', {'pg_loss': a_loss, 'c_loss': c_loss})
                    # pbar.set_postfix(show_dict)
                    a_loss_list.append(a_loss)
                    c_loss_list.append(c_loss)
                # print(a_loss, c1_loss)
            ob = next_ob
            if (c + 1) % 1000 == 0:
                agent.save_model(exp.args['weights_dir'], str(c + 1))
    envs.join()
    # plt.plot(range(len(a_loss_list)), a_loss_list, c_loss_list)
    # plt.show()


if __name__ == '__main__':
    train()
