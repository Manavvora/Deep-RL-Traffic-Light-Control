from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile
import numpy as np

from dqn.training_simulation import Simulation
from generator import TrafficGenerator
from dqn.memory import Memory
from dqn.model import TrainModel
from visualization import Visualization
from dqn.utils import import_train_configuration, set_sumo, set_train_path


def run_dqn_train():

    config = import_train_configuration(config_file='dqn/training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    visualize = Visualization(
        path, 
        dpi=96
    )
    
    counter = 0
    timestamp_start = datetime.datetime.now()
    num_runs = 10
    reward_multiple_runs = np.zeros((num_runs,config['total_episodes']))
    return_multiple_runs = np.zeros((num_runs,config['total_episodes']))
    delay_multiple_runs = np.zeros((num_runs,config['total_episodes']))
    queue_length_multiple_runs = np.zeros((num_runs,config['total_episodes']))
    

    for n_run in range(num_runs):
        Model = TrainModel(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        input_dim=config['num_states'], 
        output_dim=config['num_actions']
        )

        Memory_1 = Memory(
            config['memory_size_max'], 
            config['memory_size_min']
        )

        TrafficGen = TrafficGenerator(
            config['max_steps'], 
            config['n_cars_generated']
        )

        Simulation_DQN = Simulation(
        Model,
        Memory_1,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs']
        )
        
        print(f"Starting run number {n_run+1}")
        print("-"*25)
        episode = 0
        while episode < config['total_episodes']:
            print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
            epsilon = 1.0 - (episode / config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
            simulation_time, training_time, episode_reward, episode_return, episode_cumulative_delay, episode_queue_length = Simulation_DQN.run(episode, epsilon)  # run the simulation
            print(episode_reward[episode])
            reward_multiple_runs[n_run,episode] = episode_reward[episode]/config['n_cars_generated']
            return_multiple_runs[n_run,episode] = episode_return[episode]/config['n_cars_generated']
            delay_multiple_runs[n_run,episode] = episode_cumulative_delay[episode]/config['n_cars_generated']
            queue_length_multiple_runs[n_run,episode] = episode_queue_length[episode]
            print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
            episode += 1
            counter += 1
    
    print(reward_multiple_runs)
    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    Model.save_model(path)

    # copyfile(src='dqn/training_settings.ini', dst=os.path.join(path, 'dqn/training_settings.ini'))

    visualize.save_data_and_plot(data=reward_multiple_runs, filename='reward', xlabel='Episode', ylabel='Cumulative negative reward')
    visualize.save_data_and_plot(data=return_multiple_runs, filename='return', xlabel='Episode', ylabel='Discounted return')
    visualize.save_data_and_plot(data=delay_multiple_runs, filename='delay', xlabel='Episode', ylabel='Cumulative delay (s)')
    visualize.save_data_and_plot(data=queue_length_multiple_runs, filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)')

    return reward_multiple_runs, return_multiple_runs, delay_multiple_runs, queue_length_multiple_runs