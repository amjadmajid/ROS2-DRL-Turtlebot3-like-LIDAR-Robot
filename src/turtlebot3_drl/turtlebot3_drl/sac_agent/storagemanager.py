import os
import pickle
import json
import socket

import torch


def new_session_dir(models_dir):
    i = 0
    models_dir = os.path.join(models_dir, str(socket.gethostname()))
    session_dir = os.path.join(models_dir, "sac_%s" % i)
    while(os.path.exists(session_dir)):
        i += 1
        session_dir = os.path.join(models_dir, "sac_%s" % i)
    print(f"making new model dir: sac_{i}")
    os.makedirs(session_dir)
    return session_dir

def delete_file(path):
    if os.path.exists(path):
        os.remove(path)


def network_save_weights(network, model_dir, stage, episode):
    filepath = os.path.join(model_dir, str(network.name) + '_stage'+str(stage)+'_episode'+str(episode)+'.pt')
    print(f"saving {network.name} model for episode: {episode}")
    torch.save(network.state_dict(), filepath)


def save_session(agent_self, agent, session_dir, episode):
    print(f"saving data for episode: {episode}")
    network_save_weights(agent.actor, session_dir, agent_self.stage, episode)
    network_save_weights(agent.critic, session_dir, agent_self.stage, episode)
    network_save_weights(agent.target_critic, session_dir, agent_self.stage, episode)

    # Store parameters state
    param_keys = ['stage', 'batch_size', 'learning_rate', 'discount_factor',
                    'action_size',  'state_size', 'buffer_size', 'tau']
    param_values = [agent_self.stage, agent_self.batch_size, agent_self.learning_rate, agent_self.discount_factor,
                    agent_self.action_size, agent_self.state_size, agent_self.buffer_size, agent_self.tau]
    param_dictionary = dict(zip(param_keys, param_values))
    with open(os.path.join(agent_self.session_dir, 'stage'+str(agent_self.stage)+'_episode'+str(episode)+'.json'), 'w') as outfile:
        json.dump(param_dictionary, outfile)

    # Store replay buffer state
    pickle_data = [agent_self.replay_buffer, agent_self.rewards_data, agent_self.avg_critic_loss_data, agent_self.avg_actor_loss_data]
    with open(os.path.join(agent_self.session_dir, 'stage'+str(agent_self.stage)+'_episode'+str(episode)+'.pkl'), 'wb') as f:
        pickle.dump(pickle_data, f, pickle.HIGHEST_PROTOCOL)

    # Delete previous iterations (except every 1000th episode)
    if (episode % 1000 == 0):
        for i in range(episode, episode - 1000, 100):
            delete_file(os.path.join(session_dir, 'actor' + '_stage'+str(agent_self.stage)+'_episode'+str(i)+'.pt'))
            delete_file(os.path.join(session_dir, 'critic' + '_stage'+str(agent_self.stage)+'_episode'+str(i)+'.pt'))
            delete_file(os.path.join(session_dir, 'target_critic' + '_stage'+str(agent_self.stage)+'_episode'+str(i)+'.pt'))
            delete_file(os.path.join(session_dir, 'stage'+str(agent_self.stage)+'_episode'+str(i)+'.json'))
            delete_file(os.path.join(session_dir, 'stage'+str(agent_self.stage)+'_episode'+str(i)+'.pkl'))




def network_load_weights(network, model_dir, stage, episode):
    filepath = os.path.join(model_dir, str(network.name) + '_stage'+str(stage)+'_episode'+str(episode)+'.pt')
    print(f"loading: {network.name} model from file: {filepath}")
    network.load_state_dict(torch.load(filepath))


def load_session(agent_self, agent, session_dir, load_episode):
    # Load stored weights for network
    network_load_weights(agent.actor, session_dir, agent_self.stage, load_episode)
    network_load_weights(agent.critic, session_dir, agent_self.stage, load_episode)
    network_load_weights(agent.target_critic, session_dir, agent_self.stage, load_episode)

    # load hyperparameters
    with open(os.path.join(session_dir, 'stage'+str(agent_self.stage)+'_episode'+str(load_episode)+'.json')) as outfile:
        param = json.load(outfile)
        agent_self.batch_size = param.get('batch_size')
        agent_self.learning_rate = param.get('learning_rate')
        agent_self.discount_factor = param.get('discount_factor')
        agent_self.action_size = param.get('action_size')
        agent_self.state_size = param.get('state_size')
        agent_self.buffer_size = param.get('buffer_size')
        agent_self.tau = param.get('tau')
        # agent_self.alpha = param.get('alpha')

    # load replay buffer and graph data
    with open(os.path.join(agent_self.session_dir, 'stage'+str(agent_self.stage)+'_episode'+str(load_episode)+'.pkl'), 'rb') as f:
        pickle_data = pickle.load(f)
        agent_self.replay_buffer = pickle_data[0]
        agent_self.rewards_data = pickle_data[1]
        agent_self.avg_critic_loss_data = pickle_data[2]
        agent_self.avg_actor_loss_data = pickle_data[3]
    print(f"replay buffer length: {len(agent_self.replay_buffer.buffer)}")

    print(f"continuing session: {session_dir}")
