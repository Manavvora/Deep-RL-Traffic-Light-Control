import matplotlib.pyplot as plt
import os
import numpy as np
from ppo.utils_ppo import import_train_configuration
# from testing_simulation import Simulation 
# from training_simulation import Simulation as Simulation_train

config = import_train_configuration(config_file='ppo/training_settings_ppo.ini')

class Visualization:
    def __init__(self, path, dpi):
            self._path = path
            self._dpi = dpi


    def save_data_and_plot(self, data, filename, xlabel, ylabel):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt
        """
        min_val = np.min(data)
        max_val = np.max(data)
        mean = np.mean(data,axis=0)
        std = np.std(data,axis=0)
        episode_array = np.arange(config['total_episodes']) + 1
        plt.rcParams.update({'font.size': 24})  # set bigger font size

        plt.plot(episode_array, mean, label='Average')
        plt.fill_between(episode_array, mean-std, mean+std, alpha=.2, label=r'1-$\sigma$ Error')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        with open(os.path.join(self._path, 'plot_'+filename + '_data.txt'), "w") as file:
            for value in data:
                    file.write("%s\n" % value)

    def save_data_and_plot_test(self, data, filename, xlabel, ylabel, x_data, title):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt
        """
        min_val = np.min(data)
        max_val = np.max(data)
        # mean = np.mean(data,axis=0)
        # std = np.std(data,axis=0)
        # step_array = np.arange(config['max_steps']) + 1
        plt.rcParams.update({'font.size': 24})  # set bigger font size

        plt.plot(x_data, data, label='Plot for a fixed route-file')
        # plt.fill_between(step_array, mean-std, mean+std, alpha=.2, label=r'1-$\sigma$ Error')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.legend()
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        with open(os.path.join(self._path, 'plot_'+filename + '_data.txt'), "w") as file:
            for value in data:
                    file.write("%s\n" % value)
    
    def save_data_and_plot_test_reward(self, data, filename, xlabel, ylabel, x_data):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt
        """
        min_val = np.min(data)
        max_val = np.max(data)
        # mean = np.mean(data,axis=0)
        # std = np.std(data,axis=0)
        # step_array = np.arange(len(Simulation._reward_episode)) + 1
        plt.rcParams.update({'font.size': 24})  # set bigger font size

        plt.plot(x_data, data, label='Reward for a fixed route-file')
        # plt.fill_between(step_array, mean-std, mean+std, alpha=.2, label=r'1-$\sigma$ Error')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        with open(os.path.join(self._path, 'plot_'+filename + '_data.txt'), "w") as file:
            for value in data:
                    file.write("%s\n" % value)
    
    def save_data_and_plot_test_policy(self, data, filename, xlabel, ylabel,x_data):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt
        """
        min_val = np.min(data)
        max_val = np.max(data)
        # mean = np.mean(data,axis=0)
        # std = np.std(data,axis=0)
        # step_array = np.arange(len(Simulation._policy)) + 1
        plt.rcParams.update({'font.size': 24})  # set bigger font size

        plt.plot(x_data, data, label='Policy for a fixed route-file')
        # plt.fill_between(step_array, mean-std, mean+std, alpha=.2, label=r'1-$\sigma$ Error')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        with open(os.path.join(self._path, 'plot_'+filename + '_data.txt'), "w") as file:
            for value in data:
                    file.write("%s\n" % value)
    
    def compare_algos_train(self, algo_1_data, algo_2_data, filename, xlabel, ylabel, label_algo_1, label_algo_2):
        """
        Produce a plot comparing training performance of different algorithms
        """
        min_val = min(np.min(algo_1_data), np.min(algo_2_data))
        max_val = max(np.max(algo_1_data), np.max(algo_2_data))
        mean_algo_1 = np.mean(algo_1_data,axis=0)
        std_algo_1 = np.std(algo_1_data,axis=0)
        mean_algo_2 = np.mean(algo_2_data,axis=0)
        std_algo_2 = np.std(algo_2_data,axis=0)
        episode_array = np.arange(config['total_episodes']) + 1
        plt.rcParams.update({'font.size': 24})  # set bigger font size

        plt.plot(episode_array, mean_algo_1, label='Average : '+label_algo_1)
        plt.fill_between(episode_array, mean_algo_1-std_algo_1, mean_algo_1+std_algo_1, alpha=.2, label=r'1-$\sigma$ Error : '+label_algo_1)
        plt.plot(episode_array, mean_algo_2, label='Average : '+label_algo_2)
        plt.fill_between(episode_array, mean_algo_2-std_algo_2, mean_algo_2+std_algo_2, alpha=.2, label=r'1-$\sigma$ Error : '+label_algo_2)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

    def compare_algos_test(self, algo_1_data, algo_2_data, filename, xlabel, ylabel, label_algo_1, label_algo_2, x_data_algo_1, x_data_algo_2):
        """
        Produce a plot comparing testing performance of different algorithms
        """
        min_val = min(np.min(algo_1_data), np.min(algo_2_data))
        max_val = max(np.max(algo_1_data), np.max(algo_2_data))
        plt.rcParams.update({'font.size': 24})

        plt.plot(x_data_algo_1, algo_1_data, label=label_algo_1)
        plt.plot(x_data_algo_2, algo_2_data, label=label_algo_2)
        # plt.fill_between(step_array, mean-std, mean+std, alpha=.2, label=r'1-$\sigma$ Error')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        
