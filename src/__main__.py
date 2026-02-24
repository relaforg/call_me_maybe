from src.llm_sdk import Small_LLM_Model
from typing import Dict
from math import log, exp


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
        name_tokens = llm.encode(
            "The best function to answer the question is " + name)[0].tolist()
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


schema = """
{
\t"prompt": "<tool_call>",
\t"name": "<tool_call>",
\t"parameters": {<parameters and value depending on function>}
}
<|endoftext|>
"""

llm = Small_LLM_Model()
prompt = "What is the sum of 40 and 2?"
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
        nxt = get_max_logits_index(logits)
        # print(nxt)
        print(llm.decode(nxt), end="")
        tokens_prompt.append(nxt)
    elif (count == 0):
        for j in tokens_base:
            logits = llm.get_logits_from_input_ids(tokens_prompt)
            choose_constant_token(logits, j)
            nxt = get_max_logits_index(logits)
            print(llm.decode(nxt), end="")
            tokens_prompt.append(nxt)
        count += 1
    elif (count == 1):
        choose_func(func_dict, tokens_prompt)
    if (i == 151643):
        break
print()
