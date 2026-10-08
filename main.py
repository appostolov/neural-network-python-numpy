from skeleton import network
from network import Network

print( "skeleton" )
_skeleton = network()
print( "network" )
_network = Network( _skeleton )
print( "train" )
_network.train("dataset.csv")
_network.save()