from src.llm_sdk import Small_LLM_Model
from typing import Dict


def choose_constant_token(logits, token):
    for i in range(len(logits)):
        if (i != token):
            logits[i] = float("-inf")


def get_max_logits_index(logits):
    max_i = 0
    for i, v in enumerate(logits):
        if logits[max_i] < v:
            max_i = i
    return (max_i)


def choose_func(logits, func_dict, tokens):
    func_tokens = [llm.encode(i)[0].tolist() for i in func_dict.keys()]
    count = 0
    choice = []
    while (len(func_tokens)):
        candidates = [seq for seq in func_tokens if len(
            seq) > count and seq[:count] == choice]
        if not candidates:
            raise RuntimeError(
                "No candidates left (cannot complete a function name)")
        allowed = {seq[count] for seq in candidates}
        logits = llm.get_logits_from_input_ids(tokens)
        for i in range(len(logits)):
            if (i not in allowed):
                logits[i] = float("-inf")
            elif (llm.decode(i)[0] in prompt):
                logits[i] *= 10
        max = get_max_logits_index(logits)
        print(llm.decode(max), end="")
        choice.append(max)
        tokens.append(max)
        count += 1
        if (choice in func_tokens):
            break


schema = """
{
\t"prompt": "<tool_call>",
\t"name": "<tool_call>",
\t"parameters": {<parameters and value depending on function>}
}
<|endoftext|>
"""

llm = Small_LLM_Model()
prompt = (
    "What is the sum of 40 and 2?"
)
func_dict = {
    "fn_add_numbers": {
        "description": "Add two numbers together and return their sum.",
        "parameters": {
            "a": {
                "type": "number"
            },
            "b": {
                "type": "number"
            }
        },
        "returns": {
            "type": "number"
        }
    },
    "fn_greet": {
        "description": "Generate a greeting message for a person by name.",
        "parameters": {
            "name": {
                "type": "string"
            }
        },
        "returns": {
            "type": "string"
        }
    },
    "fn_reverse_string": {
        "description": "Reverse a string and return the reversed result.",
        "parameters": {
            "s": {
                "type": "string"
            }
        },
        "returns": {
            "type": "string"
        }
    },
    "fn_get_square_root": {
        "description": "Calculate the square root of a number.",
        "parameters": {
            "a": {
                "type": "number"
            }
        },
        "returns": {
            "type": "number"
        }
    },
    "fn_substitute_string_with_regex": {
        "description": "Replace all occurrences matching a regex pattern in a string.",
        "parameters": {
            "source_string": {
                "type": "string"
            },
            "regex": {
                "type": "string"
            },
            "replacement": {
                "type": "string"
            }
        },
        "returns": {
            "type": "string"
        }
    }
}

test = llm.encode(prompt)
print("\n")
print(test)
print(llm.decode(test))
tokens_prompt = test[0].tolist()
tokens_base = test[0].tolist()
print()
test = llm.encode(schema)
tokens = test[0].tolist()
print("\n")
print(test)
count = 0
# test = [llm.decode(i) for i in test[0].tolist()]
for i in tokens:
    logits = llm.get_logits_from_input_ids(tokens_prompt)
    if (i != 151657):
        choose_constant_token(logits, i)
        max = get_max_logits_index(logits)
        # print(max)
        print(llm.decode(max), end="")
        tokens_prompt.append(max)
    elif (count == 0):
        for j in tokens_base:
            logits = llm.get_logits_from_input_ids(tokens_prompt)
            choose_constant_token(logits, j)
            max = get_max_logits_index(logits)
            print(llm.decode(max), end="")
            tokens_prompt.append(max)
        count += 1
    elif (count == 1):
        choose_func(logits, func_dict, tokens_prompt)
    if (i == 151643):
        break
print()
