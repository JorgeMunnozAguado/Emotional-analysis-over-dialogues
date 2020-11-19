from abc import ABC, abstractmethod

class Transformation(ABC):

    @abstractmethod
    def transform():
        pass
    
    @abstractmethod
    def shape(self):
        pass
