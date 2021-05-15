import import_ipynb
from Environment import *
import pandas as pd
import numpy as np
import sys, os
import tkinter as tk
from tkinter import filedialog

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
    print(df.iloc[(-1*n):])
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
def loadModelandEnv(data):
    Policy_Network = Q_Network()
    chainer.serializers.load_hdf5("model.hdf5", Policy_Network)
    env = Environment1(data)
    return Policy_Network, env

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

# Main fuction starts here
filepath = setFilePath()
data = inputHistoricalData(filepath)
Policy_Network, env = loadModelandEnv(data)

plot_train_test(data)
plot_train_test_by_q(test_env = env, Q = Policy_Network, algorithm_name = 'DQN')

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
    print('Action to take: ' + str(action))
    print('Reward from taking action: ' + str(reward))
    print('Current total profits: ' + str(env.profits))
