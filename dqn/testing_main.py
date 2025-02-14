from __future__ import absolute_import
from __future__ import print_function

import os
from shutil import copyfile
import numpy as np

from dqn.testing_simulation import Simulation
from generator import TrafficGenerator
from dqn.model import TestModel
from visualization import Visualization
from dqn.utils import import_test_configuration, set_sumo, set_test_path


def run_dqn_test():

    config = import_test_configuration(config_file='dqn/testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])

    Model = TestModel(
        input_dim=config['num_states'],
        model_path=model_path
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    visualize = Visualization(
        plot_path, 
        dpi=96
    )
        
    Simulation_DQN = Simulation(
        Model,
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions']
    )

    num_runs = 1
    reward_multiple_runs =[]
    queue_length_multiple_runs = []
    policy_multiple_runs = []
    waiting_times_multiple_runs = []

    for n_runs in range(num_runs):
        print(f'Starting run number {n_runs}')
        print('-'*25)
        print('\n----- Test episode')
        simulation_time = Simulation_DQN.run(config['episode_seed'])  # run the simulation
        reward_multiple_runs.append(Simulation_DQN._reward_episode) 
        queue_length_multiple_runs.append(Simulation_DQN._queue_length_episode)
        policy_multiple_runs.append(Simulation_DQN._policy)
        waiting_times_multiple_runs.append(Simulation_DQN._waiting_times_episode)
        print('Simulation time:', simulation_time, 's')

    # print(reward_multiple_runs)
    print("----- Testing info saved at:", plot_path)

    # copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))

    visualize.save_data_and_plot_test(data=reward_multiple_runs[0], filename='reward', xlabel='Action step', ylabel='Reward',x_data=np.arange(len(Simulation_DQN._reward_episode))+1, title='DQN Algorithm')
    visualize.save_data_and_plot_test(data=queue_length_multiple_runs[0], filename='queue', xlabel='Step', ylabel='Queue length (vehicles)',x_data=np.arange(config['max_steps'])+1, title='DQN Algorithm')
    visualize.save_data_and_plot_test(data=policy_multiple_runs[0], filename='policy', xlabel='Step', ylabel='Action',x_data=np.arange(len(Simulation_DQN._policy))+1, title='DQN Algorithm')
    visualize.save_data_and_plot_test(data=waiting_times_multiple_runs[0], filename='delay', xlabel='Step', ylabel='Total Waiting Time (seconds)',x_data=np.arange(len(Simulation_DQN._waiting_times_episode))+1, title='DQN Algorithm')

    return Simulation_DQN._reward_episode, Simulation_DQN._policy, Simulation_DQN._queue_length_episode, Simulation_DQN._waiting_times_episode