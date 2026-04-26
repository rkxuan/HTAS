from typing import Dict, Any
from abc import ABC, abstractmethod


class PromptSet(ABC):
    """
    Abstract base class for a set of prompts.
    """
    @staticmethod
    @abstractmethod
    def get_domain_description() -> str:
        """ TODO """

    @staticmethod
    @abstractmethod
    def get_role(operation:str) -> str:
        """ TODO """

    @staticmethod
    @abstractmethod
    def get_role_prompt(operation:str) -> str:
        """ TODO """
    
    @staticmethod
    @abstractmethod
    def get_thought_constraint():
        """ TODO """
    
    @staticmethod
    @abstractmethod
    def get_answer_constraint():
        """ TODO """

    @staticmethod
    @abstractmethod
    def get_adversarial_thought_constraint():
        """ TODO """

    @staticmethod
    @abstractmethod
    def get_adversarial_answer_constraint():
        """ TODO """


    