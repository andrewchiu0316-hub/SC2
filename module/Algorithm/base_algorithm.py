from abc import ABC, abstractmethod
import torch

class BaseAlgorithm(ABC):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = None
        self.log_writer = None

    def set_log_writer(self, log_writer):  # 把外面傳進來的 logger 存起來
        self.log_writer = log_writer

    @abstractmethod
    def train(self):
        """
        給予 traj recoder, 回傳 loss
        """
        pass
    
    @abstractmethod
    def sample_action(self):
        """
        給予 current episode traj, 回傳 action id list
        """
        pass

    def episode_init(self):
        """
        Episode initial 時呼叫的函式
        """
        pass

    def episode_reset(self):
        """
        Episode reset 時呼叫的函式
        """
        pass

    def save_model(self, path: str):
        pass

    def load_model(self, path: str):
        pass

    def update_target_network():
        """
        Value-based 方式，將當前網路參數同步至 target network
        """
        pass
