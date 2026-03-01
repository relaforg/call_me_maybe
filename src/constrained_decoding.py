from llm_sdk import Small_LLM_Model
from typing import Set, List
from math import exp, log
import numpy as np


class ConstrainedDecoding:
    def __init__(self, func_dict):
        self.llm = Small_LLM_Model()
        self.func_dict = func_dict
        self.out = []
        self.context = []
        self.prompt_tokens = []
        self.NUMBER_ALLOWED = self._get_number_allowed()
        self.QUOTE_TOKEN = self.llm.encode("\"")[0].tolist()[0]
        self.BOOL_ALLOWED = self._get_bool_allowed()
        self.SCHEMA = """
{
\t"prompt": "<tool_call>",
\t"name": "<tool_call>",
\t"parameters": {<tool_call>}
}
<|endoftext|>
        """
        self.SCHEMA_TOKENS = self.llm.encode(self.SCHEMA)[0].tolist()
        self.TOOL_CALL_TOKEN = 151657
        self.EOT_TOKEN = 151643
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

    def _choose_constrained_token(self, logits: List[float], allowed: Set) -> int:
        max_t = float("-inf")
        nxt = -1
        for t in allowed:
            if (logits[t] > max_t or nxt == -1):
                nxt = t
                max_t = logits[t]
        return (nxt)

    def _logsumexp(self, xs):
        m = max(xs)
        return (m + log(sum(exp(x - m) for x in xs)))

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
        for name in [f["name"] for f in self.func_dict]:
            logprob_list = []
            name_tokens = self.llm.encode(name)[0].tolist()
            tmp_tokens = list(self.context)
            for t in range(len(name_tokens)):
                logits = self.llm.get_logits_from_input_ids(tmp_tokens)
                logprob_list.append(
                    logits[name_tokens[t]] - self._logsumexp_np(logits))
                tmp_tokens.append(name_tokens[t])
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
        param_tokens = []
        while (True):
            logits = self.llm.get_logits_from_input_ids(self.context)
            max_t = float("-inf")
            nxt = -1
            for t in allowed:
                if (logits[t] > max_t or nxt == -1):
                    nxt = t
                    max_t = logits[t]
            if not self._is_subsequence(param_tokens
                                        + [nxt], self.prompt_tokens):
                break
            self.out.append(nxt)
            self.context.append(nxt)
            param_tokens.append(nxt)

    def _get_string_param(self):
        param_tokens = []
        self.context.append(self.QUOTE_TOKEN)
        self.out.append(self.QUOTE_TOKEN)
        allowed = set(self.prompt_tokens)
        while (True):
            logits = self.llm.get_logits_from_input_ids(self.context)
            max_t = float("-inf")
            nxt = -1
            for t in allowed:
                if (logits[t] > max_t or nxt == -1):
                    nxt = t
                    max_t = logits[t]
            if not self._is_subsequence(param_tokens
                                        + [nxt], self.prompt_tokens):
                break
            self.out.append(nxt)
            self.context.append(nxt)
            param_tokens.append(nxt)
        self.context.append(self.QUOTE_TOKEN)
        self.out.append(self.QUOTE_TOKEN)

    def _get_bool_param(self):
        logits = self.llm.get_logits_from_input_ids(self.context)
        max_t = float("-inf")
        nxt = -1
        for t in self.BOOL_ALLOWED:
            if (logits[t] > max_t or nxt == -1):
                nxt = t
                max_t = logits[t]
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
        self.context = self.llm.encode(prompt)[0].tolist()
        self.out = []
        self.prompt_tokens = list(self.context)
        func_string = "functions:\n"
        for f in self.func_dict:
            func_string += f"- {f['name']} ({f['parameters']})\n"
        self.context += self.llm.encode(func_string)[0].tolist()
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
                func_name = self._choose_func()
                func_context = [
                    f for f in self.func_dict if f["name"] == func_name][0]
                self.context += \
                    self.llm.encode(func_name + " " +
                                    f"\n{str(func_context)}")[0].tolist()
                count += 1
            elif (count == 2):
                func_context = [
                    f for f in self.func_dict if f["name"] == func_name][0]
                for c, param in enumerate(func_context["parameters"].items()):
                    name, param_type = param
                    parameters = f'"{name}": '
                    self._add_string(parameters)
                    self._get_param(param_type)
                    if (c < len(func_context["parameters"]) - 1):
                        self._add_string(", ")
                count += 1
            if (i == self.EOT_TOKEN):
                break
        result = self.llm.decode(self.out)
        return (result)
