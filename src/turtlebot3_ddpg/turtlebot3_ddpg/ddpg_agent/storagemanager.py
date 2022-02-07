import os
import pickle
import json
import socket

import torch


def new_session_dir(models_dir):
    i = 0
    models_dir = os.path.join(models_dir, str(socket.gethostname()))
    session_dir = os.path.join(models_dir, "ddpg_%s" % i)
    while(os.path.exists(session_dir)):
        i += 1
        session_dir = os.path.join(models_dir, "ddpg_%s" % i)
    print(f"making new model dir: ddpg_{i}")
    os.makedirs(session_dir)
    return session_dir

def delete_file(path):
    if os.path.exists(path):
        os.remove(path)


def network_save_weights(network, model_dir, stage, episode):
    filepath = os.path.join(model_dir, str(network.name) + '_stage'+str(stage)+'_episode'+str(episode)+'.pt')
    print(f"saving {network.name} model for episode: {episode}")
    torch.save(network.state_dict(), filepath)


def save_session(ddpg_self, session_dir, episode):
    print(f"saving data for episode: {episode}")
    network_save_weights(ddpg_self.actor, session_dir, ddpg_self.stage, episode)
    network_save_weights(ddpg_self.critic, session_dir, ddpg_self.stage, episode)
    # network_save_weights(ddpg_self.target_actor, session_dir, ddpg_self.stage, episode)
    network_save_weights(ddpg_self.target_critic, session_dir, ddpg_self.stage, episode)

    # Store parameters state
    param_keys = ['stage', 'noise_sigma', 'batch_size', 'learning_rate',
                  'discount_factor', 'episode_size', 'action_size',  'state_size', 'memory_size', 'tau', 'alpha']
    param_values = [ddpg_self.stage, ddpg_self.actor_noise.sigma, ddpg_self.batch_size,
                    ddpg_self.learning_rate, ddpg_self.discount_factor, ddpg_self.episode_size, ddpg_self.action_size,
                    ddpg_self.state_size, ddpg_self.memory_size, ddpg_self.tau, ddpg_self.alpha]
    param_dictionary = dict(zip(param_keys, param_values))
    # with open(os.path.join(ddpg_self.session_dir, 'stage'+str(ddpg_self.stage)+'_episode'+str(episode)+'.json'), 'w') as outfile:
    #     json.dump(param_dictionary, outfile)

    # Store replay buffer state
    pickle_data = [ddpg_self.memory, ddpg_self.rewards_data, ddpg_self.avg_critic_loss_data, ddpg_self.avg_actor_loss_data]
    with open(os.path.join(ddpg_self.session_dir, 'stage'+str(ddpg_self.stage)+'_episode'+str(episode)+'.pkl'), 'wb') as f:
        pickle.dump(pickle_data, f, pickle.HIGHEST_PROTOCOL)

    # Delete previous iterations (except every 1000th episode)
    if (episode % 1000 == 0):
        for i in range(episode, episode - 1000, 100):
            delete_file(os.path.join(session_dir, 'actor' + '_stage'+str(ddpg_self.stage)+'_episode'+str(i)+'.pt'))
            # delete_file(os.path.join(session_dir, 'target_actor' + '_stage'+str(ddpg_self.stage)+'_episode'+str(i)+'.pt'))
            delete_file(os.path.join(session_dir, 'critic' + '_stage'+str(ddpg_self.stage)+'_episode'+str(i)+'.pt'))
            delete_file(os.path.join(session_dir, 'target_critic' + '_stage'+str(ddpg_self.stage)+'_episode'+str(i)+'.pt'))
            delete_file(os.path.join(session_dir, 'stage'+str(ddpg_self.stage)+'_episode'+str(i)+'.json'))
            delete_file(os.path.join(session_dir, 'stage'+str(ddpg_self.stage)+'_episode'+str(i)+'.pkl'))




def network_load_weights(network, model_dir, stage, episode):
    filepath = os.path.join(model_dir, str(network.name) + '_stage'+str(stage)+'_episode'+str(episode)+'.pt')
    print(f"loading: {network.name} model from file: {filepath}")
    network.load_state_dict(torch.load(filepath))


def load_session(ddpg_self, session_dir, load_episode):
    # Load stored weights for network
    network_load_weights(ddpg_self.actor, session_dir, ddpg_self.stage, load_episode)
    network_load_weights(ddpg_self.critic, session_dir, ddpg_self.stage, load_episode)
    # network_load_weights(ddpg_self.target_actor, session_dir, ddpg_self.stage, load_episode)
    network_load_weights(ddpg_self.target_critic, session_dir, ddpg_self.stage, load_episode)

    # load hyperparameters
    with open(os.path.join(session_dir, 'stage'+str(ddpg_self.stage)+'_episode'+str(load_episode)+'.json')) as outfile:
        param = json.load(outfile)
        ddpg_self.actor_noise.sigma = param.get('noise_sigma')

    # load replay memory buffer and graph data
    with open(os.path.join(ddpg_self.session_dir, 'stage'+str(ddpg_self.stage)+'_episode'+str(load_episode)+'.pkl'), 'rb') as f:
        pickle_data = pickle.load(f)
        ddpg_self.memory = pickle_data[0]
        ddpg_self.rewards_data = pickle_data[1]
        ddpg_self.avg_critic_loss_data = pickle_data[2]
        ddpg_self.avg_actor_loss_data = pickle_data[3]
    print(f"memory length: {len(ddpg_self.memory.buffer)}")

    print(f"continuing session: {session_dir}")
