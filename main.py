import os
import numpy as np

from ppo.testing_main_ppo import run_ppo_test
from ppo.training_main_ppo import run_ppo_train
from dqn.testing_main import run_dqn_test
from dqn.training_main import run_dqn_train
from visualization import Visualization
from ppo.utils_ppo import import_test_configuration, set_compare_path


def main():

    print('Running PPO')
    print('-'*25)
    reward_ppo_train, return_ppo_train, queue_ppo_train, delay_ppo_train = run_ppo_train()
    reward_ppo_test, policy_ppo_test, queue_ppo_test, delay_ppo_test = run_ppo_test()

    print('Running DQN')
    print('-'*25)
    reward_dqn_train, return_dqn_train, queue_dqn_train, delay_dqn_train = run_dqn_train()
    reward_dqn_test, policy_dqn_test, queue_dqn_test, delay_dqn_test = run_dqn_test()

    config = import_test_configuration(config_file='ppo/testing_settings_ppo.ini')
    model_path, plot_path = set_compare_path(config['models_path_name'], config['model_to_test'])

    visualize = Visualization(
        plot_path, 
        dpi=96
    )
    visualize.compare_algos_train(algo_1_data=reward_ppo_train, algo_2_data=reward_dqn_train, filename='compare_train_reward', xlabel='Episode', ylabel='Average negative reward', label_algo_1='PPO', label_algo_2='DQN')
    visualize.compare_algos_train(algo_1_data=queue_ppo_train, algo_2_data=queue_dqn_train, filename='compare_train_queue', xlabel='Episode', ylabel='Average queue length (vehicles)', label_algo_1='PPO', label_algo_2='DQN')
    visualize.compare_algos_train(algo_1_data=delay_ppo_train, algo_2_data=delay_dqn_train, filename='compare_train_delay', xlabel='Episode', ylabel='Average delay (s)', label_algo_1='PPO', label_algo_2='DQN')
    visualize.compare_algos_train(algo_1_data=return_ppo_train, algo_2_data=return_dqn_train, filename='compare_train_return', xlabel='Episode', ylabel='Average negative return', label_algo_1='PPO', label_algo_2='DQN')
    
    visualize.compare_algos_test(algo_1_data=reward_ppo_test, algo_2_data=reward_dqn_test, filename='compare_test_reward', xlabel='Step', ylabel='Cumulative negative reward', label_algo_1='PPO', label_algo_2='DQN', x_data_algo_1=np.arange(len(reward_ppo_test))+1, x_data_algo_2=np.arange(len(reward_dqn_test))+1)
    visualize.compare_algos_test(algo_1_data=queue_ppo_test, algo_2_data=queue_dqn_test, filename='compare_test_queue', xlabel='Step', ylabel='Average queue length (vehicles)', label_algo_1='PPO', label_algo_2='DQN', x_data_algo_1=np.arange(len(queue_ppo_test))+1, x_data_algo_2=np.arange(len(queue_dqn_test))+1)
    visualize.compare_algos_test(algo_1_data=delay_ppo_test, algo_2_data=delay_dqn_test, filename='compare_test_delay', xlabel='Step', ylabel='Cumulative delay (s)', label_algo_1='PPO', label_algo_2='DQN', x_data_algo_1=np.arange(len(delay_ppo_test))+1, x_data_algo_2=np.arange(len(delay_dqn_test))+1)
    visualize.compare_algos_test(algo_1_data=policy_ppo_test, algo_2_data=policy_dqn_test, filename='compare_test_policy', xlabel='Step', ylabel='Cumulative negative return', label_algo_1='PPO', label_algo_2='DQN', x_data_algo_1=np.arange(len(policy_ppo_test))+1, x_data_algo_2=np.arange(len(policy_dqn_test))+1)


if __name__ == "__main__":
    main()