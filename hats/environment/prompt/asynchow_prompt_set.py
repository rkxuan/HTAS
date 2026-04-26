from typing import Union, Dict, Any, List
import itertools

from hats.environment.prompt.prompt_set import PromptSet
from hats.environment.prompt.prompt_set_registry import PromptSetRegistry
from hats.system import Message

ROLE2ROLE = {
    "AsyncCodeStep":
        "Code Writter",
    "AsyncStep":
        "Plan Organizer",
    "AsyncReflection":
        "Improver"
}

ROLE2PROMPT = {
    "SystemDecision":(
        "You are the top decision-maker and are good at analyzing and summarizing other agents' opinions, finding errors and giving final answers."
    ),
    "AsyncCodeStep": (
        "You are a programmer.\n"
        "You are good at computer science, engineering, and physics.\n"
        "You have experience in designing and developing computer software and hardware.\n"
    ),
    "AsyncStep": (
        "You are a helpful plan organizer to address problems.\n"
    ),
    "AsyncReflection": (
        "You are an improver.\n"
    ),
}

ROLE2CONSTRAINT = {
    "AsyncCodeStep": (
        #"Question: Assume that you need to execute all the steps to complete the task and that infinite resources are available. "
        #"What is the shortest possible time to finish the task?\n"
        "I will give you a task with multiple steps and their ordering constraints. "
        "Assume that you need to execute all the steps to complete the task and that infinite resources are available. "
        "You should only use a Python code block to tell me the shortest possible time of the task. "
        "We will execute your code and get namespace['result'] as your answer. "
        "The variable result(string) should contain 'Xd Ys' where X is the number of days and Y is the number of seconds.\n"
        "Answer example:\n"
        "```python\n"
        "def formalize_time(time_str):\n"
        "   <your code>\n"
        "def shortest_time(steps):\n"
        "   <your code>\n"
        "time = shortest_time(steps)\n"
        "# result: '0d 2100s'\n"
        "result = formalize_time(time)\n"
        "```"
        "" 
    ),
    "AsyncStep": (
        #"Question: Assume that you need to execute all the steps to complete the task and that infinite resources are available. "
        #"What is the shortest possible time to finish the task?\n"
        "I will give you a task with multiple steps and their ordering constraints. "
        "Assume that you need to execute all the steps to complete the task and that infinite resources are available. "
        "You should tell me the shortest possible time to finish the task. "
        "Give your analysis and final answer in format: Xd Ys where X is the number of days and Y is the number of seconds.\n"
        "Answer example:\n" 
        "Since step 1 must precede step 5, step 2 must precede step 3, step 3 must precede step 4, step 4 must precede step 5, "
        "we can conclude that we must execute step 2, step 3, step 4, then step 5 sequentially, and since step 1 happens before step 5, "
        "it can be done in parallel with step 2, 3, and 4, preceding step 5. Since sequentially executing step 2, 3, 4, and 5 takes 10 + "
        "15 + 5 + 25 = 55min, while sequentially executing step 1 then step 5 only takes 10 + 25 = 35min, 35min = 0d 2100s.\n"
        "The answer is 0d 2100s."
    ),
    "AsyncReflection": (
        #"Question: Assume that you need to execute all the steps to complete the task and that infinite resources are available. "
        #"What is the shortest possible time to finish the task?\n"
        "I will give you a task with multiple steps and their ordering constraints. "
        "Assume that other agents need to execute all the steps to complete the task and that infinite resources are available. "
        "Please point out potential issues in other agents' solutions one by one. "
        "If you do not receive any solution from other agents, please keep silent."
    )
}



@PromptSetRegistry.register('asynchow')
class AsynchowPromptSet(PromptSet):
    """
    Asynchow prompt set for asynchronous planning
    """
    @staticmethod
    def get_domain_description():
        return "a multi-agent collaboration environment for asynchronous planning."

    @staticmethod
    def get_role(role: str):
        if role in ROLE2ROLE:
            return ROLE2ROLE[role]
        else:
            raise Exception(f"No such {role} in asynchow prompt set")

    @staticmethod
    def get_role_prompt(role: str):
        if role in ROLE2PROMPT:
            return ROLE2PROMPT[role]
        else:
            raise Exception(f"No such {role} in asynchow prompt set")

    @staticmethod
    def get_thought_constraint(role: str):
        if role in ROLE2CONSTRAINT:
            return ROLE2CONSTRAINT[role]
        else:
            raise Exception(f"No such {role} in asynchow prompt set")

    @staticmethod
    def get_answer_constraint(**kwargs):
        return (
            "I will give you a task with multiple steps and their ordering constraints. "
            "Assume that you need to execute all the steps to complete the task and that infinite resources are available. "
            "You should tell me the shortest possible time to finish the task. "
            "Give your analysis and final answer in format: Xd Ys where X is the number of days and Y is the number of seconds.\n"
            "Answer example:\n" 
            "Since step 1 must precede step 5, step 2 must precede step 3, step 3 must precede step 4, step 4 must precede step 5, "
            "we can conclude that we must execute step 2, step 3, step 4, then step 5 sequentially, and since step 1 happens before step 5, "
            "it can be done in parallel with step 2, 3, and 4, preceding step 5. Since sequentially executing step 2, 3, 4, and 5 takes 10 + "
            "15 + 5 + 25 = 55min, while sequentially executing step 1 then step 5 only takes 10 + 25 = 35min, 35min = 0d 2100s.\n"
            "The answer is 0d 2100s."
        )

    @staticmethod
    def get_message_prompt(
        raw_inputs: Message, 
        neighbor_messages: List[Message], 
        self_messages: List[Message], 
        **kwargs
    ):
        task_prompt = f"Question: {raw_inputs.task}\n"
        
        neighbor_prompt = ""
        for neighbor_message in neighbor_messages:
            if not neighbor_message.EOF:
                neighbor_prompt += f"Agent({neighbor_message.agent_id}) role is {neighbor_message.agent_role}, output is:\n{neighbor_message.textual_output}\n"

        neighbor_prompt = f"At the same time, outputs of other agents are as follows:\n {neighbor_prompt}\n" if neighbor_prompt else ""
        
        self_prompt = ""
        for self_message in self_messages:
            if not self_message.EOF:
                self_prompt += f"Moreover, your previous output is:\n{self_message.textual_output}\n"
                break
        return task_prompt + neighbor_prompt + self_prompt

    @staticmethod
    def get_adversarial_thought_constraint(**kwargs):
        return NotImplementedError

    @staticmethod
    def get_adversarial_answer_constraint(**kwargs):
        return NotImplementedError
