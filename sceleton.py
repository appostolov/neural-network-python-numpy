import random
import math

def network(input_size: int) -> list:
    size1 = math.floor( 64 )
    hidden1 = layer( size1, input_size )
    size2 = math.floor( size1 / 2 )
    hidden2 = layer( size2, size1 )
    output = output_neuron( size2 )
    return [hidden1, hidden2, output]

def deviation(size: int) -> float:
    return ( 2 / size ) ** ( 1 / 2 )

def weight(size: int) -> float:
    d = deviation(size)
    return random.uniform(-d, d)

def neuron(size: int)->list:
    weights = []
    for i in range( size ):
        weights.append( weight( size ) )
    return [weights, 0.0]

def layer(size: int, input_size: int)->list:
    neurons = []
    for i in range( size ):
        neurons.append( neuron( input_size ) )
    return neurons

def output_neuron(input_size: int)->list:
    deviation = ( 6 / ( input_size + 1 ) ) ** ( 1 / 2 )
    weights = []
    for i in range( input_size ):
        weights.append( random.uniform(-deviation, deviation) )
    return [weights, 0.0]