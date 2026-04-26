from typing import Optional

class Connection:  
    # The definition of the connection between a pair of agents
    def __init__(
        self, 
        src: str, 
        dst: str
    ):
        self.src = src
        self.dst = dst
    
    @property
    def connection_type(self):
        return self.__class__.__name__

class Decentralized_Connection(Connection):
    def __init__(
        self, 
        src: str, 
        dst: str
    ):
        super().__init__(src, dst)

class Sequential_Connection(Connection):
    def __init__(
        self, 
        src: str, 
        dst: str
    ):
        super().__init__(src, dst)
