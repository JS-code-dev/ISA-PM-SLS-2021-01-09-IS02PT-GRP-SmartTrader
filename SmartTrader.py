import import_ipynb
from SmartTraderLibrary import *
import pandas as pd
import numpy as np
import sys, os
import tkinter as tk
from tkinter import filedialog

# Environment parameters
EnvPara = {
    'tnx_cost':0,
    'funding_cost':0,
    'clip_rewards':False,
    'debug':False,
    'history_t':90
}

# Set file directory path
def setFilePath(initDir = os.curdir):
    root = tk.Tk()
    root.withdraw()
    filepath = tk.filedialog.askopenfilename(parent=root, initialdir=initDir, title='Please input share historical data')
    return filepath

# Prints first and last n rows of dataframe
def printFirstLastofDF(df, n=5):
    print('')
    print(df.head(n))
    print('...  ...  ...')
    print('...  ...  ...')
    print('...  ...  ...')
    print(df.tail(n))
    print('')

# Read and extract historical data
def inputHistoricalData(filepath):
    data = pd.read_csv(filepath)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    print(data.index.min(), data.index.max())
    printFirstLastofDF(data)
    return data

# Load model and intialize environment
def loadPolicyNetwork(model, env):
    Policy_Network = Q_Network(input_size=env.feature_size, output_size=len(act_dict))
    chainer.serializers.load_hdf5(model, Policy_Network)
    return Policy_Network

# Set up environment
def setupEnv(data):
    env = Environment1(data, tnx_cost=EnvPara['tnx_cost'],
                   funding_cost=EnvPara['funding_cost'],
                   clip_rewards=EnvPara['clip_rewards'],
                   debug=EnvPara['debug'],
                   history_t=EnvPara['history_t'])
    env.reset()
    return env

# Prompt for user input, and process into dataframe
def userInput():
    date = ''
    price = ''
    print('')
    print('Note: Enter \'0\' to exit program for either inputs')
    while(date == ''):
        date = input('Enter date of price with format yyyy-mm-dd: ')
        if date == '0': sys.exit()
    while(price == ''):
        price = input('Enter price of share on indicated date: ')
        if price == '0': sys.exit()

    try:
        input_data = pd.DataFrame({'Date': [date], 'Close': [float(price)]})
        input_data['Date'] = pd.to_datetime(input_data['Date'])
        input_data = input_data.set_index('Date')
    except:
        print('Error with input data!!! Please try again...')
        return 0

    return input_data

####### Main fuction starts here #########
filepath = setFilePath()
data = inputHistoricalData(filepath)
env = setupEnv(data)
Policy_Network = loadPolicyNetwork("model.hdf5", env)

plot_train_test(test = data)

train_env, test_env, \
train_actions, train_rewards, train_invested_capital, train_position_value, train_profits, \
test_actions, test_rewards, test_invested_capital, test_position_value, test_profits \
= apply_trained_model(Policy_Network = Policy_Network, test_env = env)

plot_train_test_by_q(algorithm_name="DQN",
                     test_env=test_env,
                     test_actions=test_actions,
                     test_rewards=test_rewards,
                     test_position_value=test_position_value,
                     test_profits=test_profits)

while(1):
    input_data = userInput()
    # if input_data == 0: continue
    print(input_data)

    data = data.append(input_data)
    printFirstLastofDF(data)

    # Recommend next action to take based on user input
    env.data = data
    state = [env.position_value] + env.history
    action_value = Policy_Network(np.array(state, dtype=np.float32).reshape(1, -1))

        # Hard constrain to not be able to sell when there are no positions
    if len(env.positions) == 0 and np.argmax(action_value.data) == 2:
        action_value.data[0][np.argmax(action_value.data)] = np.min(action_value.data)

    action = np.argmax(action_value.data)
    next_state, reward, done = env.step(action)

    print('')
    print('Action to take: ' + str(act_dict[action]))
    print('Reward from taking action: ' + str(reward))
    print('Current total profits: ' + str(env.profits))
