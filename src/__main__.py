from src.llm_sdk import Small_LLM_Model
from math import log, exp
from typing import List


llm = Small_LLM_Model()
out: List[int] = []


def choose_constant_token(logits, token):
    for i in range(len(logits)):
        if (i != token):
            logits[i] = float("-inf")


def choose_constrained_token(logits, allowed) -> int:
    max_t = float("-inf")
    nxt = -1
    for t in allowed:
        if (logits[t] > max_t or nxt == -1):
            nxt = t
            max_t = logits[t]
    return (nxt)


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


def choose_func(func_dict, tokens, out):
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
    out += llm.encode(best_name)[0].tolist()
    return (best_name)


def get_number_allowed():
    nbrs = set()
    for i in "0123456789.-":
        nbrs.add(llm.encode(i)[0].tolist()[0])
    return (nbrs)


NUMBER_ALLOWED = get_number_allowed()


def get_number_param(context, prompt_token):
    allowed = NUMBER_ALLOWED & set(prompt_token)
    param_tokens = []
    while (True):
        logits = llm.get_logits_from_input_ids(context)
        max_t = float("-inf")
        nxt = -1
        for t in allowed:
            if (logits[t] > max_t or nxt == -1):
                nxt = t
                max_t = logits[t]
        if (llm.decode(param_tokens + [nxt]) not in llm.decode(prompt_token)):
            break
        out.append(nxt)
        context.append(nxt)
        param_tokens.append(nxt)


def get_string_param(context, prompt_token):
    param_tokens = []
    quote_token = llm.encode("\"")[0].tolist()[0]
    context.append(quote_token)
    out.append(quote_token)
    allowed = set(prompt_token)
    while (True):
        logits = llm.get_logits_from_input_ids(context)
        max_t = float("-inf")
        nxt = -1
        for t in allowed:
            if (logits[t] > max_t or nxt == -1):
                nxt = t
                max_t = logits[t]
        if (llm.decode(param_tokens + [nxt]) not in llm.decode(prompt_token)):
            break
        out.append(nxt)
        context.append(nxt)
        param_tokens.append(nxt)
    context.append(quote_token)
    out.append(quote_token)


def get_bool_allowed():
    allowed = set()
    for i in ["true", "false"]:
        allowed.add(llm.encode(i)[0].tolist()[0])
    return (allowed)


BOOL_ALLOWED = get_bool_allowed()


def get_bool_param(context):
    logits = llm.get_logits_from_input_ids(context)
    max_t = float("-inf")
    nxt = -1
    for t in BOOL_ALLOWED:
        if (logits[t] > max_t or nxt == -1):
            nxt = t
            max_t = logits[t]
    out.append(nxt)
    context.append(nxt)


def get_param(context, param_type, prompt_token):
    match param_type["type"]:
        case "number":
            return get_number_param(context, prompt_token)
        case "bool" | "boolean":
            return get_bool_param(context)
        case "string" | _:
            return get_string_param(context, prompt_token)


schema = """
{
\t"prompt": "<tool_call>",
\t"name": "<tool_call>",
\t"parameters": {<tool_call>}
}
<|endoftext|>
"""

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

prompt = "What is the sum of 40 and 2?"
context = llm.encode(prompt)[0].tolist()
tokens_base = list(context)
func_string = "functions:\n"
for name, value in func_dict.items():
    func_string += f"- {name} ({value['parameters']})\n"
context += llm.encode(func_string)[0].tolist()
test = llm.encode(schema)
tokens = test[0].tolist()
count = 0
for i in tokens:
    logits = llm.get_logits_from_input_ids(context)
    if (i != 151657):
        nxt = choose_constrained_token(logits, {i})
        out.append(nxt)
        context.append(nxt)
    elif (count == 0):
        for j in tokens_base:
            logits = llm.get_logits_from_input_ids(context)
            nxt = choose_constrained_token(logits, {j})
            out.append(nxt)
            context.append(nxt)
        count += 1
    elif (count == 1):
        func_name = choose_func(func_dict, context, out)
        context += llm.encode(func_name + " " +
                              f"\n{str(func_dict[func_name])}")[0].tolist()
        count += 1
    elif (count == 2):
        COMMA_SPACE = llm.encode(", ")[0].tolist()
        for c, param in enumerate(func_dict[func_name]["parameters"].items()):
            name, param_type = param
            parameters = f'"{name}": '
            param_tokens = llm.encode(parameters)[0].tolist()
            for j in param_tokens:
                logits = llm.get_logits_from_input_ids(context)
                nxt = choose_constrained_token(logits, {j})
                out.append(nxt)
                context.append(nxt)
            get_param(context, param_type, tokens_base)
            if (c < len(func_dict[func_name]["parameters"]) - 1):
                param_tokens = COMMA_SPACE
                for j in param_tokens:
                    logits = llm.get_logits_from_input_ids(context)
                    nxt = choose_constrained_token(logits, {j})
                    out.append(nxt)
                    context.append(nxt)
        count += 1
    if (i == 151643):
        break
print(llm.decode(out))
