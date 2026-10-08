import random
import math
from chars import valid_email_chars

char_vector_size = 3
email_max_length = 50
first_hidden_size = 64
input_layer_size = char_vector_size * email_max_length

def network() -> list:
    input = dictionary()
    size1 = first_hidden_size
    hidden1 = layer( size1, input_layer_size )
    size2 = math.floor( size1 / 2 )
    hidden2 = layer( size2, size1 )
    output = output_neuron( size2 )
    return [input, hidden1, hidden2, output]

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

def dictionary()->list[float]:
    size = ( len(valid_email_chars) + 1 ) * char_vector_size
    neurons = []
    for i in range( size ):
        neurons.append( random.uniform(-0.1, 0.1) )
    return neurons

def char_to_indexes(char: str)->list[int]:
    if char in valid_email_chars:
        index = valid_email_chars.index( char )
    else: index = len(valid_email_chars)
    start = index * char_vector_size
    result = []
    for i in range( char_vector_size ):
        result.append( start + i )
    return result

def email_to_indexes(email: str)->list[int]:
    email = email_normal(email = email)
    indexes = []
    for char in email:
        indexes.extend(char_to_indexes(char))
    return indexes

def email_normal(email: str)->str:
    if len(email) > email_max_length: email = email[0:email_max_length]
    elif len(email) < email_max_length: email = email.ljust(email_max_length)
    return email

def map_input(email: str, dictionary: list[float])->list[float]:
    email = email_normal(email = email)
    indexes = email_to_indexes( email = email )
    neurons = []
    for index in indexes:
        neurons.append( dictionary[index] )
    return neurons