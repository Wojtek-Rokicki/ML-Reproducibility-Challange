from abc import ABC, abstractmethod


class Optimizer(ABC):
    name: str

    @abstractmethod
    def optimize(self, w_0, tx, y, max_iter):
        pass
