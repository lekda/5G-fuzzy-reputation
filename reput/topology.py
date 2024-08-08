"""Topology of the different RAN elements. 
"""

from typing import List, Tuple, Callable, Any

# TODO Class network with capacity as inherited attribute 
class AccessNetwork():
    """An access network is composed of one or several peers. 
    It is interconnected to one or multiple core networks. 
    """
    def __init__(self, peer_nb, id)->None:
        self.peers:List[Any] = []
        self.peer_nb:int = peer_nb
        
        # Loging purpose 
        self.id:str=id
        self.no_capacity_for_transaction:int = 0 
        self.no_peer_for_transaction:int = 0      
        
    @property
    def capacity(self)->int: 
        return self.peer_nb - len(self.peers)

    def check_labels(self, label:str)->bool: 
        """Consult wether a specific label is present in the an peers or not. 

        Args:
            label (str): label to check

        Returns:
            bool: True if label is present, false otherwise
        """
        for p in self.peers : 
            if p.label == label :
                return True
        return False
    
class CoreNetwork():
    """A core networks is composed of several peers. 
    It is interconnected to multiple AccessNetwork.
    It can be interconnectd to other CoreNetwork 
    """
    def __init__(self, peer_nb, id)->None:
        self.peers:List[Any] = []
        self.peer_nb:int = peer_nb
        
        # Loging purpose 
        self.id:str=id

    @property
    def capacity(self)->int: 
        return self.peer_nb - len(self.peers)
