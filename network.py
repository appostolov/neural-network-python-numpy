import numpy as np
from input import email_to_array
import pandas as pd
import json

class Network:
    body: list
    def __init__(self, sceleton: list):
        self.body = sceleton
    def run(self, input: list, cache_signal: bool = False )->float:
        hidden1 = []
        for n in self.body[0]:
            weights = n[0]
            bias = n[1]
            signal = ( np.array(weights) @ np.array(input) ) + bias
            hidden1.append( signal )
            if( cache_signal ): n.append( signal )
        hidden2 = []
        for n in self.body[1]:
            weights = n[0]
            bias = n[1]
            signal = ( np.array(weights) @ np.array(hidden1) ) + bias
            hidden2.append( signal )
            if( cache_signal ): n.append( signal )
        weights = self.body[2][0]
        bias = self.body[2][1]
        output = ( np.array(weights) @ np.array(hidden2) ) + bias
        return 1 / ( 1 + np.exp( -output ) )
    def backpropagation( self, input: list, output: float, expected: float, learn_rate: float):
        # Adjust the output neuron
        sigmoid_derivative = output * (1.0 - output)
        delta_output = (output - expected) * sigmoid_derivative
        self.body[2][1] -= learn_rate * delta_output
        for i in range(len( self.body[2][0] )):
            self.body[2][0][i] -= learn_rate * delta_output * self.body[1][i][2]
        # Adjust the hidden before the output
        delta_hidden = []
        for i in range(len( self.body[1] )):
            error_from_output = delta_output * self.body[2][0][i]
            relu_derivative = 1.0 if self.body[1][i][2] > 0 else 0.0
            delta_hidden.append( error_from_output * relu_derivative )
            self.body[1][i][1] -= learn_rate * delta_hidden[i]
            for j in range(len(self.body[1][i][0])):
                self.body[1][i][0][j] -= learn_rate * delta_hidden[i] * self.body[0][i][2]
        # Adjust the hidden layer who recieves the input
        for i in range(len( self.body[0] )):
            error_from_next = 0.0
            for j in range(len(delta_hidden)):
                error_from_next += delta_hidden[j] * self.body[1][j][0][i]
            relu_derivative = 1.0 if self.body[0][i][2] > 0 else 0.0
            delta_first_hidden = error_from_next * relu_derivative
            self.body[0][i][1] -= learn_rate * delta_first_hidden
            for k in range(len( input )):
                self.body[0][i][0][k] -= learn_rate * delta_first_hidden * input[k]
    def train(self, path: str):
        learn_rate = 0.07
        df = pd.read_csv(path)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_data = df.iloc[:16000]
        test_data = df.iloc[16000:]
        for i, row in enumerate(train_data.values):
            input = email_to_array( email = row[0] )
            output = self.run( input = input, cache_signal = True )
            expected = 1.0 if row[1] == 1 else 0.0
            self.backpropagation( input = input, output = output, expected = expected, learn_rate = learn_rate )

            print(f"\r⏳ Processing row: {i}/16,000", end="", flush=True)
        print()
        correct = 0
        for i, row in enumerate(test_data.values):
            input = email_to_array( email = row[0] )
            output = self.run( input = input )
            result = 1.0 if output >= 0.5 else 0.0
            expected = 1.0 if row[1] == 1 else 0.0
            if( result == expected ): correct += 1
            print(f"\r⏳ Testing row: {i}/4,000", end="", flush=True)
        print()
        print(learn_rate)
        print( ( correct / 4000 ) * 100 )
    def save(self, path: str = "nn.json"):
        with open(path, 'w') as file:
            json.dump(self.body, file, indent=4)






    



