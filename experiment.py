import logging
import torch
from qnetwork import Qnet
from value_buffer import ValueBuffer
from replay_buffer import ReplayBuffer
from config import Config, parse_arguments
from train_test import train
import utils
import datetime, json, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

env = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config()
logging.basicConfig(level=20)

def main():
    ''' 
    run the experiment
    '''
    global env
    logging.info("Started...")
    envpool = config.envpool
    env = utils.make_atari_env(config)
    # env = utils.make_env(config)
    n_actions = env.action_space.n
    qnet = Qnet(n_actions, embedding_size=config.embedding_size).to(device)
    target_net = Qnet(n_actions, embedding_size=config.embedding_size).to(device)
    target_net.load_state_dict(qnet.state_dict())
    target_net.eval()
    replay_buffer = ReplayBuffer(config.replay_buffer_size, config.embedding_size, config.path_length)
    optimizer = torch.optim.Adam(qnet.parameters(), lr=config.lr)
    value_buffer = ValueBuffer(config.value_buffer_size)
    eval_rewards = train(env, qnet, target_net, optimizer, replay_buffer, value_buffer, config, device, envpool)

    reward_save_path = 'results/' + config.envname + '/EVA'
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    os.makedirs(reward_save_path, exist_ok=True)
    reward_save_path = reward_save_path + '/' + now + '.json'
    print(reward_save_path)
    with open(reward_save_path, 'w') as f:
        json.dump(eval_rewards, f)


if __name__ == "__main__":
    parse_arguments(config)
    main()

