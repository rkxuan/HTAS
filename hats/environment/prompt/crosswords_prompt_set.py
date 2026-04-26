from typing import Dict, Any

from hats.environment.prompt.prompt_set import PromptSet
from hats.environment.prompt.prompt_set_registry import PromptSetRegistry

@PromptSetRegistry.register('crosswords')
class CrosswordsPromptSet(PromptSet):

    @staticmethod
    def get_domain_description():
        return "a multi-agent collaboration environment for solving 5 x 5 mini crossword puzzles."

    @staticmethod
    def get_propose_prompt(board, num_candidates=5):
        return f'''Let's play a 5 x 5 mini crossword, where each word should have exactly 5 letters.

{board}

Given the current status, list all possible answers for unfilled or changed words, and your confidence levels (certain/high/medium/low).
Use "certain" cautiously and only when you are 100% sure this is the correct word.
You can list more then one possible answer for each word.
Each answer line should follow the format "position. word (confidence)".
For example, you answer can be:
v5. drier (certain) 
v5. dryer (medium)
h1. apple (low)
h2. poet (certain)
h2. lyric (medium)
'''
    @staticmethod
    def get_if_correct_prompt(word, meaning):
        return f'Does {word} has meaning "{meaning}"? Responde only Yes or No.'
    
    @staticmethod
    def get_suggest_prompt(board, impossible_words, correct_words, incorrect_words):
        feedback_words = {}
        if len(impossible_words) > 0:
            feedback_words['Impossible Words'] = impossible_words
        if len(correct_words) > 0:
            feedback_words['Correct Words'] = correct_words
        if len(incorrect_words) > 0:
            feedback_words['Incorrect Words'] = incorrect_words
        feedback_words_str = '\n'.join([f'{key}:\n{value}\n---' for key, value in feedback_words.items()])
        prompt = f'''You are playing a 5 x 5 mini crossword, where each word should have exactly 5 letters.
Given the current status:
{board}

The target words are classified as'''
        word_classes = list(feedback_words.keys())
        for word_class in word_classes[:-1]:
            prompt += f' {word_class},'
        prompt += f' and {word_classes[-1]}.\n---'
        prompt += feedback_words_str
        prompt += '''You will retry the game. Write a plan for the next time. 
Respond at most five sentences, one sentence per line.
Do not include the phrase "next time" in your response.
'''
        return prompt
    
    @staticmethod
    def get_value_prompt(input):
        return f'''Evaluate if there exists a five letter word of some meaning that fit some letter constraints (sure/maybe/impossible).

Incorrect; to injure: w _ o _ g
The letter constraint is: 5 letters, letter 1 is w, letter 3 is o, letter 5 is g.
Some possible words that mean "Incorrect; to injure":
wrong (w r o n g): 5 letters, letter 1 is w, letter 3 is o, letter 5 is g. fit!
sure

A person with an all-consuming enthusiasm, such as for computers or anime: _ _ _ _ u
The letter constraint is: 5 letters, letter 5 is u.
Some possible words that mean "A person with an all-consuming enthusiasm, such as for computers or anime":
geek (g e e k): 4 letters, not 5
otaku (o t a k u): 5 letters, letter 5 is u
sure

Dewy; roscid: r _ _ _ l
The letter constraint is: 5 letters, letter 1 is r, letter 5 is l.
Some possible words that mean "Dewy; roscid":
moist (m o i s t): 5 letters, letter 1 is m, not r
humid (h u m i d): 5 letters, letter 1 is h, not r
I cannot think of any words now. Only 2 letters are constrained, it is still likely
maybe

A woodland: _ l _ d e
The letter constraint is: 5 letters, letter 2 is l, letter 4 is d, letter 5 is e.
Some possible words that mean "A woodland":
forest (f o r e s t): 6 letters, not 5
woods (w o o d s): 5 letters, letter 2 is o, not l
grove (g r o v e): 5 letters, letter 2 is r, not l
I cannot think of any words now. 3 letters are constrained, and _ l _ d e seems a common pattern
maybe

An inn: _ d _ w f
The letter constraint is: 5 letters, letter 2 is d, letter 4 is w, letter 5 is f.
Some possible words that mean "An inn":
hotel (h o t e l): 5 letters, letter 2 is o, not d
lodge (l o d g e): 5 letters, letter 2 is o, not d
I cannot think of any words now. 3 letters are constrained, and it is extremely unlikely to have a word with pattern _ d _ w f to mean "An inn"
impossible

Chance; a parasitic worm; a fish: w r a k _
The letter constraint is: 5 letters, letter 1 is w, letter 2 is r, letter 3 is a, letter 4 is k.
Some possible words that mean "Chance; a parasitic worm; a fish":
fluke (f l u k e): 5 letters, letter 1 is f, not w
I cannot think of any words now. 4 letters are constrained, and it is extremely unlikely to have a word with pattern w r a k _ to mean "Chance; a parasitic worm; a fish"
impossible

{input}
'''
    
    @staticmethod
    def get_role(operation:str) -> str:
        return NotImplementedError

    @staticmethod
    def get_role_prompt(operation:str) -> str:
        return NotImplementedError
    
    @staticmethod
    def get_thought_constraint():
        return NotImplementedError
    
    @staticmethod
    def get_answer_constraint():
        return NotImplementedError

    @staticmethod
    def get_adversarial_thought_constraint():
        return NotImplementedError

    @staticmethod
    def get_adversarial_answer_constraint():
        return NotImplementedError




