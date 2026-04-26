import re
from copy import deepcopy
from typing import List, Union, Literal, Iterator, Any

from hats.system import Message
from hats.environment.tools.crosswords.env import MiniCrosswordsEnv


def record_to_system_input(
    env: MiniCrosswordsEnv,
    idx: int
):
    _env = deepcopy(env)
    _env.reset(idx)
    return Message(False, None, None, None, [_env], None)

def crosswords_get_predict(answers: List[Message]):
    answers = answers[0].textual_output
    answer = None
    try:
        answer = max(answers, key=lambda x: getattr(x, 'r_word'))
    except:
        for answer in answers:
            print(answer)
    if not answer:
        score = 0.0 
    else:
        score = answer.r_word
    return score
