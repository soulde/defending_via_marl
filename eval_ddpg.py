import torch
from MADDPG import MADDPG
from Model import Actor, Critic
from Parameters import AgentParams
from sandbox import DefendingSandbox
from utils import *
from itertools import count
import numpy as np
import tqdm


np.random.seed(0)
torch.manual_seed(0)


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 设置了训练使用的设备
    exp = ExperimentManager('./run')
    exp.start_experiment('./sandbox/configs/defending_ddpg.json', 'eval')
    sandbox = DefendingSandbox(exp.args['sandbox'])
    train_param = exp.args['train']
    eval_times = 100
    n_envs = 1
    model_path = './run/train_defending_ddpg_20240517_113105/weights'

    num_enemy = exp.args['sandbox/mission/enemy_info/init/num_agents']
    num_agents = exp.args['sandbox/mission/num_agents']
    # env = gym.make('Pendulum-v1', render_mode='human')
    param = AgentParams()
    param.from_config(train_param)
    agent = MADDPG(Actor, Critic, num_agents, 16, 2, param, np.array([1, 3]))
    index = 200000

    if not index == 0:
        agent.load_model(model_path, index)
    total_num = 0
    for _ in tqdm.trange(eval_times):
        ob, _ = sandbox.reset()
        for _ in count():
            ob = torch.tensor(ob, dtype=torch.float32, device=device).reshape(n_envs, num_agents, -1)
            with torch.no_grad():
                action = agent.get_action(ob)
            u = action[0].cpu().detach().numpy() * np.array([0.1, 1])
            observation, reward, termination, truncation, info = sandbox.step(u)

            next_ob = torch.tensor(observation, dtype=torch.float32, device=device).reshape(n_envs, num_agents, -1)
            if '_final_info' in info.keys():
                if 'captured_num' in info.keys():
                    total_num += sum(info['captured_num'])

            if termination or truncation:
                break
            ob = next_ob

    cap_rate = total_num / (num_enemy * eval_times)
    print(cap_rate)


if __name__ == '__main__':
    main()
