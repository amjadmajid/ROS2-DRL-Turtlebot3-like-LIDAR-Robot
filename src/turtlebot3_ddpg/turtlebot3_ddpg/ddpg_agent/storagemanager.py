import os
import pickle
import json

import torch


def new_session_dir(models_dir):
    i = 0
    session_dir = os.path.join(models_dir, "ddpg_%s" % i)
    while(os.path.exists(session_dir)):
        i += 1
        session_dir = os.path.join(models_dir, "ddpg_%s" % i)
    print(f"making new model dir: ddpg_{i}")
    os.mkdir(session_dir)
    return session_dir


def network_save_weights(network, model_dir, stage, episode):
    filepath = os.path.join(model_dir, str(network.name) + '_stage'+str(stage)+'_episode'+str(episode)+'.pt')
    print(f"saving {network.name} model for episode: {episode}")
    torch.save(network.state_dict(), filepath)


def save_session(ddpg_self, session_dir, episode):
    print(f"saving data for episode: {episode}")
    network_save_weights(ddpg_self.actor, session_dir, ddpg_self.stage, episode)
    network_save_weights(ddpg_self.critic, session_dir, ddpg_self.stage, episode)
    network_save_weights(ddpg_self.target_actor, session_dir, ddpg_self.stage, episode)
    network_save_weights(ddpg_self.target_critic, session_dir, ddpg_self.stage, episode)

    # Store parameters state
    param_keys = ['stage', 'noise_sigma', 'batch_size', 'learning_rate',
                  'discount_factor', 'episode_size', 'action_size',  'state_size', 'memory_size', 'tau']
    param_values = [ddpg_self.stage, ddpg_self.actor_noise.sigma, ddpg_self.batch_size,
                    ddpg_self.learning_rate, ddpg_self.discount_factor, ddpg_self.episode_size, ddpg_self.action_size,
                    ddpg_self.state_size, ddpg_self.memory_size, ddpg_self.tau]
    param_dictionary = dict(zip(param_keys, param_values))
    with open(os.path.join(ddpg_self.session_dir, 'stage'+str(ddpg_self.stage)+'_episode'+str(episode)+'.json'), 'w') as outfile:
        json.dump(param_dictionary, outfile)

    # Store replay buffer state
    with open(os.path.join(ddpg_self.session_dir, 'stage'+str(ddpg_self.stage)+'_episode'+str(episode)+'.pkl'), 'wb') as f:
        pickle.dump(ddpg_self.memory, f, pickle.HIGHEST_PROTOCOL)


def network_load_weights(network, model_dir, stage, episode):
    filepath = os.path.join(model_dir, str(network.name) + '_stage'+str(stage)+'_episode'+str(episode)+'.pt')
    print(f"loading: {network.name} model from file: {filepath}")
    network.load_state_dict(torch.load(filepath))


def load_session(ddpg_self, session_dir, load_episode):
    # Load stored weights for network
    network_load_weights(ddpg_self.actor, session_dir, ddpg_self.stage, load_episode)
    network_load_weights(ddpg_self.critic, session_dir, ddpg_self.stage, load_episode)
    network_load_weights(ddpg_self.target_actor, session_dir, ddpg_self.stage, load_episode)
    network_load_weights(ddpg_self.target_critic, session_dir, ddpg_self.stage, load_episode)

    # load hyperparameters
    with open(os.path.join(session_dir, 'stage'+str(ddpg_self.stage)+'_episode'+str(load_episode)+'.json')) as outfile:
        param = json.load(outfile)
        ddpg_self.actor_noise.sigma = param.get('noise_sigma')

    # load replay memory buffer
    # with open(os.path.join(session_dir, 'stage'+str(ddpg_self.stage)+'_episode'+str(load_episode)+'.pkl'), 'rb') as f:
    #     ddpg_self.memory = pickle.load(f)

    print(f"memory length: {ddpg_self.memory.get_length()}")
    print(f"continuing session: {session_dir}")
