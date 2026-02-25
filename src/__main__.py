from src.llm_sdk import Small_LLM_Model
from typing import Dict
from math import log, exp
import re

INT_REGEX = "^-?\\d+$"
FLOAT_REGEX = "^-?(?:\\d+\\.\\d+|\\d+)$"
STRING_REGEX = "^\"(?:[^\"\\\\]|\\\\.)*\"$"
BOOL_REGEX = "^(true|false)$"
llm = Small_LLM_Model()


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


def logsumexp(xs):
    m = max(xs)
    return m + log(sum(exp(x - m) for x in xs))


def compute_avg_logprob(prob_list):
    if (len(prob_list) == 0):
        return (0)
    sum = 0
    for i in prob_list:
        sum += i
    return (sum / len(prob_list))


def choose_func(func_dict, tokens):
    best_name = ""
    best_prob = float("-inf")
    for name, value in func_dict.items():
        logprob_list = []
        name_tokens = llm.encode(name)[0].tolist()
        tmp_tokens = list(tokens)
        for t in range(len(name_tokens)):
            logits = llm.get_logits_from_input_ids(tmp_tokens)
            logprob_list.append(
                logits[name_tokens[t]] - logsumexp(logits))
            tmp_tokens.append(name_tokens[t])
        avg_prob = compute_avg_logprob(logprob_list)
        if (avg_prob > best_prob):
            best_prob = avg_prob
            best_name = name
    print(best_name, end="")
    return (best_name)


def get_param(context, param_type, prompt_token):
    nbrs = set()
    for i in "0123456789.-":
        nbrs.add(llm.encode(i)[0].tolist()[0])
    allowed = nbrs & set(prompt_token)
    param_tokens = []
    while (True):
        logits = llm.get_logits_from_input_ids(context)
        max_t = float("-inf")
        i = -1
        for t in allowed:
            if (logits[t] > max_t or i == -1):
                i = t
                max_t = logits[t]
        if (llm.decode(param_tokens + [i]) not in llm.decode(prompt_token)):
            break
        print(llm.decode(i), end="")
        context.append(i)
        param_tokens.append(i)


schema = """
{
\t"prompt": "<tool_call>",
\t"name": "<tool_call>",
\t"parameters": {<tool_call>}
}
<|endoftext|>
"""

# Ajouter les description dans le contexte
# prompt = "What is Mickael Jackson location ?"
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
    "fn_find_people": {
        "description": "Find people location",
        "parameters": {
            "a": {
                "type": "string"
            }
        },
        "returns": {
            "type": "string"
        }
    },
    "fn_substitute_string_with_regex": {
        "description": "Replace all occurrences matching \
a regex pattern in a string.",
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

# prompt = "What is the sum of 40 and 2?"
prompt = "Where is Mickael Jackson ?"
context = llm.encode(prompt)[0].tolist()
func_string = "functions:\n"
for name, value in func_dict.items():
    func_string += f"- {name} ({value['parameters']})\n"
    # print(func_string)
context += llm.encode(func_string)[0].tolist()
tokens_base = llm.encode(prompt)[0].tolist()
test = llm.encode(schema)
tokens = test[0].tolist()
count = 0
for i in tokens:
    logits = llm.get_logits_from_input_ids(context)
    if (i != 151657):
        choose_constant_token(logits, i)
        nxt = get_max_logits_index(logits)
        # print(nxt)
        print(llm.decode(nxt), end="")
        context.append(nxt)
    elif (count == 0):
        for j in tokens_base:
            logits = llm.get_logits_from_input_ids(context)
            choose_constant_token(logits, j)
            nxt = get_max_logits_index(logits)
            print(llm.decode(nxt), end="")
            context.append(nxt)
        count += 1
    elif (count == 1):
        func_name = choose_func(func_dict, context)
        context += llm.encode(func_name + " " +
                              f"\n{str(func_dict[func_name])}")[0].tolist()
        count += 1
    elif (count == 2):
        for c, param in enumerate(func_dict[func_name]["parameters"].items()):
            name, param_type = param
            parameters = f'"{name}": '
            param_tokens = llm.encode(parameters)[
                0].tolist()
            # param_tokens += get_param(tokens, param_type, tokens_base)
            for j in param_tokens:
                logits = llm.get_logits_from_input_ids(context)
                choose_constant_token(logits, j)
                nxt = get_max_logits_index(logits)
                print(llm.decode(nxt), end="")
                context.append(nxt)
            get_param(context, param_type, tokens_base)
            if (c < len(func_dict[func_name]["parameters"]) - 1):
                param_tokens = llm.encode(", ")[0].tolist()
                for j in param_tokens:
                    logits = llm.get_logits_from_input_ids(context)
                    choose_constant_token(logits, j)
                    nxt = get_max_logits_index(logits)
                    print(llm.decode(nxt), end="")
                    context.append(nxt)
        count += 1
    if (i == 151643):
        break
print()
