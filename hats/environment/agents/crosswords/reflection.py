from typing import Optional


from hats.system import Agent
from hats.environment.agents.agent_registry import AgentRegistry
from hats.environment.operations import Reflection, GreedyStep


@AgentRegistry.register('CrosswordsReflection')
class CrosswordsReflection(Agent):
    def __init__(
        self, 
        domain: str, 
        model_name: Optional[str] = None,  
        num_reflections:int = 1, 
        num_inner_iters:int = 2
    ):
        self.num_reflections = num_reflections
        self.num_inner_iters = num_inner_iters
        super().__init__(domain, model_name)
        self.agent_description = "an agent to learn from a solution and improve the solution iteratively."

    def build_agent(self):
        last_step = None
        for i in range(self.num_reflections + 1):
            for j in range(self.num_inner_iters):
                step = GreedyStep(self.domain, self.model_name)
                self.add_operation(step)
                if last_step:
                    last_step.add_successor(step)
                    last_step = step
                else:
                    self.input_operations[step.id] = step
                    last_step = step

            if i < self.num_reflections:
                reflection = Reflection(self.domain, self.model_name)
                self.add_operation(reflection)
                if last_step is None:
                    raise Exception("last_step should not be None")
                last_step.add_successor(reflection)
                last_step = reflection
                
        if last_step is None:
            raise Exception("last_step should not be None")
        self.output_operations[last_step.id] = last_step