from sceleton import network
from network import Network
import json
from input import email_to_array

print( "sceleton" )
_sceleton = network( 50 * 7 )
print( "network" )
_network = Network( _sceleton )
print( "train" )
_network.train("dataset.csv")
_network.save()