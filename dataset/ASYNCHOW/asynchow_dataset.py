import re
from typing import List, Tuple, Union, Literal, Iterator, Any

from hats.system import Message

def record_to_system_input(task:str):
    return Message(False, None, None, None, task, None)

def asynchow_agent_answer_process(answers:List[Message]):
    final_answer = ""
    for answer in answers:
        if not answer.EOF:
            final_answer = answer.textual_output

    final_answer = final_answer.splitlines()[-1] if final_answer else ""

    day_match = re.search(r"(\d+)d", final_answer)
    sec_match = re.search(r"(\d+)s", final_answer)

    day = int(day_match.group(1)) if day_match else -1
    sec = int(sec_match.group(1)) if sec_match else -1
    return (day, sec)

def asynchow_system_answer_process(answer:str):
    day_match = re.search(r"(\d+)d", answer)
    sec_match = re.search(r"(\d+)s", answer)

    day = int(day_match.group(1)) if day_match else 0
    sec = int(sec_match.group(1)) if sec_match else 0

    if day == 0 and sec == 0:
        raise Exception("check {}, there is no valid answer.".format(answer))
    return (day, sec)


def asynchow_judge(
    agent_answer:Tuple[int, int], 
    system_anwer:Tuple[int, int],
)->int:
    agent_day, agent_sec = agent_answer
    system_day, system_sec = system_anwer

    agent_day, agent_sec = agent_day + agent_sec//(3600*24), agent_sec%(3600*24)
    system_day, system_sec = system_day + system_sec//(3600*24), system_sec%(3600*24)

    if agent_day == system_day and agent_sec == system_sec:
        return 1
    else:
        return 0