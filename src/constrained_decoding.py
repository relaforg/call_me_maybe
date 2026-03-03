from llm_sdk import Small_LLM_Model
from typing import Set, List, Dict
from time import time
import numpy as np
import json


class ConstrainedDecoding:
    def __init__(self, func_dict: List[Dict]) -> None:
        self.llm = Small_LLM_Model()
        self.func_dict = func_dict
        self.out: List[int] = []
        self.context: List[int] = []
        self.prompt_tokens: List[int] = []
        self.NUMBER_ALLOWED = self._get_number_allowed()
        self.QUOTE_TOKEN = self.llm.encode("\"")[0].tolist()[0]
        self.BOOL_ALLOWED = self._get_bool_allowed()
        self.SCHEMA = """
{
\t"prompt": "<tool_call>",
\t"name": "<tool_call>",
\t"parameters": {<tool_call>}
}
"""
        self.SCHEMA_TOKENS = self.llm.encode(self.SCHEMA)[0].tolist()
        self.TOOL_CALL_TOKEN = 151657
        self._func_name_tokens: dict[str, list[int]] = {
            f["name"]: self.llm.encode(f["name"])[0].tolist()
            for f in self.func_dict
        }

    def _get_bool_allowed(self):
        allowed = set()
        for i in ["true", "false"]:
            allowed.add(self.llm.encode(i)[0].tolist()[0])
        return (allowed)

    def _get_number_allowed(self):
        nbrs = set()
        for i in "0123456789.-":
            nbrs.add(self.llm.encode(i)[0].tolist()[0])
        return (nbrs)

    def _choose_constrained_token(self, logits: List[float], allowed: Set):
        logits_np = np.asarray(logits, dtype=np.float64)
        allowed_np = np.fromiter(allowed, dtype=np.int64)
        return int(allowed_np[np.argmax(logits_np[allowed_np])])

    def _logsumexp(self, xs) -> float:
        x = np.asarray(xs, dtype=np.float64)
        m = x.max()
        return float(m + np.log(np.exp(x - m).sum()))

    def _compute_avg_logprob(self, prob_list):
        if (len(prob_list) == 0):
            return (0)
        sum = 0
        for i in prob_list:
            sum += i
        return (sum / len(prob_list))

    def _choose_func(self):
        best_name = ""
        best_prob = float("-inf")

        root_logits = self.llm.get_logits_from_input_ids(self.context)
        lse_root = self._logsumexp(root_logits)
        for name, tokens in self._func_name_tokens.items():
            logprob_list = [float(root_logits[tokens[0]]) - lse_root]

            if len(tokens) > 1:
                tmp_tokens = list(self.context) + [tokens[0]]
                for t in tokens[1:]:
                    logits = self.llm.get_logits_from_input_ids(tmp_tokens)
                    logprob_list.append(
                        float(logits[t]) - self._logsumexp(logits))
                    tmp_tokens.append(t)

            avg_prob = self._compute_avg_logprob(logprob_list)
            if (avg_prob > best_prob):
                best_prob = avg_prob
                best_name = name
        self.out += self.llm.encode(best_name)[0].tolist()
        return (best_name)

    def _is_subsequence(self, sub: list[int], seq: list[int]) -> bool:
        n, m = len(sub), len(seq)
        if n == 0:
            return (True)
        for i in range(m - n + 1):
            if seq[i:i+n] == sub:
                return (True)
        return (False)

    def _get_number_param(self):
        allowed = self.NUMBER_ALLOWED & set(self.prompt_tokens)
        if (not len(allowed)):
            allowed = set(self.prompt_tokens)
        param_tokens = []
        while (True):
            logits = self.llm.get_logits_from_input_ids(self.context)
            nxt = self._choose_constrained_token(logits, allowed)
            if not self._is_subsequence(param_tokens
                                        + [nxt], self.prompt_tokens):
                break
            self.out.append(nxt)
            self.context.append(nxt)
            param_tokens.append(nxt)

    def get_max_logits_index(self, logits):
        max_i = 0
        for i, v in enumerate(logits):
            if logits[max_i] < v:
                max_i = i
        return (max_i)

    def _get_string_param(self):
        self.context.append(self.QUOTE_TOKEN)
        self.out.append(self.QUOTE_TOKEN)
        while (True):
            logits = self.llm.get_logits_from_input_ids(self.context)
            nxt = self.get_max_logits_index(logits)
            if ('"' in self.llm.decode(nxt)):
                tmp = self.llm.decode(nxt).split("\"")[0]
                nxt = self.llm.encode(tmp)[0].tolist()
                self.out += nxt
                self.context += nxt
                break
            self.out.append(nxt)
            self.context.append(nxt)
        self.context.append(self.QUOTE_TOKEN)
        self.out.append(self.QUOTE_TOKEN)

    def _get_bool_param(self):
        logits = self.llm.get_logits_from_input_ids(self.context)
        nxt = self._choose_constrained_token(logits, self.BOOL_ALLOWED)
        self.out.append(nxt)
        self.context.append(nxt)

    def _get_param(self, param_type):
        match param_type["type"]:
            case "number":
                return self._get_number_param()
            case "bool" | "boolean":
                return self._get_bool_param()
            case "string" | _:
                return self._get_string_param()

    def _add_string(self, s: str):
        token_list = self.llm.encode(s)[0].tolist()
        for i in token_list:
            self.out.append(i)
            self.context.append(i)

    def run(self, prompt: str) -> str:
        prompt = json.dumps(prompt)[1:-1]
        self.context = self.llm.encode(prompt)[0].tolist()
        self.out = []
        self.prompt_tokens = []
        self.prompt_tokens = list(self.context)
        count = 0
        for i in self.SCHEMA_TOKENS:
            if (i != self.TOOL_CALL_TOKEN):
                self.out.append(i)
                self.context.append(i)
            elif (count == 0):
                for j in self.prompt_tokens:
                    self.out.append(j)
                    self.context.append(j)
                count += 1
            elif (count == 1):
                self.context += self.llm.encode(
                    "\nBased on the prompt, the correct function is:\n"
                )[0].tolist()
                t1 = time()
                func_name = self._choose_func()
                print(f"Function choosing done in {time() - t1:.3f} seconds")
                count += 1
            elif (count == 2):
                func_context = [
                    f for f in self.func_dict if f["name"] == func_name][0]
                for c, param in enumerate(func_context["parameters"].items()):
                    name, param_type = param
                    parameters = f'"{name}": '
                    self._add_string(parameters)
                    t1 = time()
                    self._get_param(param_type)
                    print(
                        f"Get param {c + 1} done in {time() - t1:.3f} seconds")
                    if (c < len(func_context["parameters"]) - 1):
                        self._add_string(", ")
                count += 1
        result = self.llm.decode(self.out)
        return (result)
