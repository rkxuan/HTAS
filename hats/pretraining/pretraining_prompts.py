from typing import List, Dict, Tuple, Optional

from hats.system import Agent
from hats.llm import LLMRegistry, LLM
from hats.environment.prompt import PromptSetRegistry, PromptSet

def generate_sequential_topology_prompt(
    domain: str,
    agents: Dict[str, Agent], 
    failure: Optional[str] = None
) -> Tuple[str, str]:
    # agent_class, agent_description, agent_id
    prompt_set = PromptSetRegistry.get(domain)
    domain_description = prompt_set.get_domain_description()

    agents_descriptions = ""
    for i, agent in enumerate(list(agents.values())):
        agent_des = f"agent {i+1}: {agent.agent_name} description: {agent.agent_description} id: {agent.id}\n"
        agents_descriptions += agent_des

    system_prompt = (
        "You are a professional multi-agent task planner. "
        "I will provide you with the background of a task, and agents in a system. "
        "The information of agents includes their classes, descriptions, and ids."
        "Your task is to generate a sequential topology for the agents to complete the task. \n"
        "Your generated topology should follow: "
        "(1) no cycle in the topology. "
        "(2) all agents should be included in the topology. "
        "(3) SystemDecision is the final output agent, therefore it has no successors.\n"
        "If I tell you about your previous incorrect answers and the reasons for them, please make improvements.\n"
        "Please follow the answer template strictly. "
        "Answer template:\n"
        "<your detailed thought of how to generate the topology>\n"
        "Therefore, the topology is composed of following connections:\n"
        "Sequential connection 1: agent_id -> agent_id\n"
        "Sequential connection 2: agent_id -> agent_id\n"
        "...\n"
        "Example:\n"
        "Q: we should address a multi-agent collaboration environment for solving python programming problems.\n"
        "Here is the information of agents:\n"
        "agent 1: ProgrammingExpert description: a programming expert to generate full code implementation. id: 3yYT\n"
        "agent 2: ProgrammingExpert description: a programming expert to generate full code implementation. id: 42Mk\n"
        "agent 3: ProjectManager description: a project manager who is responsible for overseeing the overall structure of the code. id:5u8K\n"
        "agent 4: BugFixer description: a bug fixer to provide modified and improved python code based on the current overall code design. id:6f1l\n"
        "agent 5: SystemDecision description: the agent to generate final decisions based on the information from other agents. id:7v9L\n"
        "A: Based on the information of agents, firstly, the global planning of ProjectManager should be executed by two ProgrammingExperts. "
        "Subsequently, the generated code should be checked by the BugFixer. Finally, SystemDecision will receive the test result from the BugFixer to generate the final decision.\n"
        "Therefore, the topology is composed of following connections:\n"
        "Sequential connection 1: 5u8K -> 3yYT\n"
        "Sequential connection 2: 5u8K -> 42Mk\n"
        "Sequential connection 3: 3yYT -> 6f1l\n"
        "Sequential connection 4: 42Mk -> 6f1l\n"
        "Sequential connection 5: 6f1l -> 7v9L\n"
    )

    user_prompt = f"Q: we should address {domain_description}\nHere is the information of agents:\n{agents_descriptions}\n{failure}\n"
    return system_prompt, user_prompt

def generate_decentralized_topology_prompt(
    domain: str,
    agents: Dict[str, Agent], 
    failure: Optional[str] = None
) -> Tuple[str, str]:
    prompt_set = PromptSetRegistry.get(domain)
    domain_description = prompt_set.get_domain_description()

    agents_descriptions = ""
    for i, agent in enumerate(list(agents.values())):
        agent_des = f"agent {i+1}: {agent.agent_name} description: {agent.agent_description} id: {agent.id}\n"
        agents_descriptions += agent_des

    system_prompt = (
        "You are a professional multi-agent task planner. "
        "I will provide you with the background of a task, and agents in a system. "
        "The information of agents includes their classes, descriptions, and ids."
        "Your task is to generate a decentralized topology for the agents to complete the task. "
        "And in a decentralized graph, each agent communicates synchronously with its neighbors.\n"
        "Your generated topology should follow: "
        "(1) all agents should be included in the topology. "
        "(2) SystemDecision is the final output agent, therefore it has no successors."
        "(3) any connected topology is allowed due to communication is across all nodes.\n"
        "If I tell you about your previous incorrect answers and the reasons for them, please make improvements.\n"
        "Please follow the answer template strictly. "
        "Answer template:\n"
        "<your detailed thought of how to generate the topology>\n"
        "Therefore, the topology is composed of following connections:\n"
        "Decentralized connection 1: agent_id -> agent_id\n"
        "Decentralized connection 2: agent_id -> agent_id\n"
        "...\n"
        "Example:\n"
        "Q: we should address a multi-agent collaboration environment for solving python programming problems.\n"
        "Here is the information of agents:\n"
        "agent 1: ProgrammingExpert description: a programming expert to generate full code implementation. id: 3yYT\n"
        "agent 2: ProgrammingExpert description: a programming expert to generate full code implementation. id: 42Mk\n"
        "agent 3: ProjectManager description: a project manager who is responsible for overseeing the overall structure of the code. id:5u8K\n"
        "agent 4: BugFixer description: a bug fixer to provide modified and improved python code based on the current overall code design. id:6f1l\n"
        "agent 5: SystemDecision description: the agent to generate final decisions based on the information from other agents. id:7v9L\n"
        "A: Based on the information of agents, I think ProjectManager should tell how to generate codes to two ProgrammingExperts. "
        "Two ProgrammingExperts should communicate with each other to share information. "
        "Two ProgrammingExperts should communicate with BugFixer as well. "
        "Two ProgrammingExperts should generate useful codes for SystemDecision.\n"
        "Therefore, the topology is composed of following connections:\n"
        "Decentralized connection 1: 5u8K -> 3yYT\n"
        "Decentralized connection 2: 5u8K -> 42Mk\n"
        "Decentralized connection 3: 3yYT -> 42Mk\n"
        "Decentralized connection 4: 42Mk -> 3yYT\n"
        "Decentralized connection 5: 3yYT -> 6f1l\n"
        "Decentralized connection 6: 42Mk -> 6f1l\n"
        "Decentralized connection 7: 6f1l -> 3yYT\n"
        "Decentralized connection 8: 6f1l -> 42Mk"
        "Decentralized connection 9: 3yYT -> 7v9L\n"
        "Decentralized connection 10: 42Mk -> 7v9L\n"
    )

    user_prompt = f"Q: we should address {domain_description}\nHere is the information of agents:\n{agents_descriptions}\n{failure}\n"
    return system_prompt, user_prompt


def generate_hybrid_topology_prompt(
    domain: str,
    agents: Dict[str, Agent], 
    failure: Optional[str] = None
) -> Tuple[str, str]:
    prompt_set = PromptSetRegistry.get(domain)
    domain_description = prompt_set.get_domain_description()

    agents_descriptions = ""
    for i, agent in enumerate(list(agents.values())):
        agent_des = f"agent {i+1}: {agent.agent_name} description: {agent.agent_description} id: {agent.id}\n"
        agents_descriptions += agent_des

    system_prompt = (
        "You are a professional multi-agent task planner. "
        "I will provide you with the background of a task, and agents in a system. "
        "The information of agents includes their classes, descriptions, and ids."
        "And there are two topologies in hybrid topology: "
        "Sequential connections for ordered inter-step communication and decentralized connections for synchronous inner-step communication. "
        "Your task is to generate a hybrid topology for the agents to complete the task with three steps: "
        "Firstly, you need to assign each agent to a specific step. "
        "Subsequently, you need to construct inner-step decentralized connections."
        "Finally, you need to construct inter-step sequential connections."
        
        "Your generated topology should follow: "
        "(1) each agent is assigned to a unique step. "
        "(2) within a step, all agents should be included by decentralized connections. "
        "(3) all steps should be included by sequential connections. "
        "(4) there is no cycles in sequential topology. "
        "(5) any connected topology is allowed within a step. "
        "(6) SystemDecision is the final output agent, which is a single step with no successors."
        
        "If I tell you about your previous incorrect answers and the reasons for them, please make improvements.\n"
        
        "Please follow the answer template strictly. "
        "Answer template:\n"
        "<your detailed thought of how to assign agents to steps>\n"
        "Therefore, steps are as follows:\n"
        "step 1 <step name>: <list of agent_id>\n"
        "step 2 <step name>: <list of agent_id>\n"
        "...\n"
        "<your detailed thought of how to generate decentralized topologies for each step>\n"
        "Therefore, decentralized connections are as follows:\n"
        "Decentralized connection 1: agent_id -> agent_id within step id\n"
        "Decentralized connection 2: agent_id -> agent_id within step id\n"
        "...\n"
        "<your detailed thought of how to generate sequential topologies between two steps>\n"
        "Therefore, sequential connections are as follows:\n"
        "Sequential connection 1: agent_id -> agent_id from step id to step id\n"
        "Sequential connection 2: agent_id -> agent_id from step id to step id\n"
        "...\n"
        "Example:\n"
        "Q: we should address a multi-agent collaboration environment for solving python programming problems.\n"
        "Here is the information of agents:\n"
        "agent 1: ProgrammingExpert description: a programming expert to generate full code implementation. id: 3yYT\n"
        "agent 2: ProgrammingExpert description: a programming expert to generate full code implementation. id: 42Mk\n"
        "agent 3: ProjectManager description: a project manager who is responsible for overseeing the overall structure of the code. id:5u8K\n"
        "agent 4: BugFixer description: a bug fixer to provide modified and improved python code based on the current overall code design. id:6f1l\n"
        "agent 5: SystemDecision description: the agent to generate final decisions based on the information from other agents. id:7v9L\n"
        "A: Based on the information of agents, I think the code should first be generated, then bugs of code should be fixed, finally feasible code is sent to SystemDecision."
        "Therefore, steps are as follows:\n"
        "step 1 code generation: [3yYT, 42Mk, 5u8K]\n"
        "step 2 bug fixing: [6f1l]\n"
        "step 3 final decision: [7v9L]\n"
        "Subsequently, I think within step 1, ProjectManager should tell two ProgrammingExperts how to generate code, "
        "and ProjectManagers should discuss with each other. "
        "And I find both step 2 and step 3 only contain one agent, so there is no decentralized connection. "
        "Therefore, decentralized connections are as follows:\n"
        "Decentralized connection 1: 5u8K -> 3yYT within step 1\n"
        "Decentralized connection 2: 5u8K -> 42Mk within step 1\n"
        "Decentralized connection 3: 3yYT -> 42Mk within step 1\n"
        "Decentralized connection 4: 42Mk -> 3yYT within step 1\n"
        "And from step 1 to step 2, generated codes of ProgrammingExperts should be sent to BugFixer. "
        "From step2 to step 3, fixed code should be sent to SystemDecision. "
        "Therefore, sequential connections are as follows:\n"
        "Sequential connection 1: 3yYT -> 6f1l from step 1 to step 2\n"
        "Sequential connection 2: 42Mk -> 6f1l from step 1 to step 2\n"
        "Sequential connection 3: 6f1l -> 7v9L from step 2 to step 3\n"
        "Sequential connection 4: 6f1l -> 7v9L from step 2 to step 3\n"
    )

    user_prompt = f"Q: we should address {domain_description}\nHere is the information of agents:\n{agents_descriptions}\n{failure}\n"
    return system_prompt, user_prompt



