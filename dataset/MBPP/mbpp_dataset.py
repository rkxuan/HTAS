import re
from typing import List, Union, Literal, Iterator, Any

from hats.system import Message

def record_to_system_input(record):
    return Message(False, None, None, None, record["task"], None)

def mbpp_data_process(dataset):
    list_data_dict = []
    for data in dataset:
        text = data["text"]
        test_case_list = data["test_list"]
        
        test = "def test():\n"
        for test_case in test_case_list:
            test = test + "    " + test_case + "\n" 
        test += "\ntest()"

        task = "python function requirement: "
        task = task + text + " test cases:"
        for test_case in test_case_list:
            task = task + " " + test_case 

        list_data_dict.append({"task": task, "test": test})
    return list_data_dict

def mbpp_answer_process(answers:List[Message]):
    final_answer = ""
    for answer in answers:
        if not answer.EOF:
            final_answer = answer.textual_output

    final_answer = final_answer.lstrip("```python\n").rstrip("\n```")
    return final_answer