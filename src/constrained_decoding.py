from llm_sdk import Small_LLM_Model
from typing import Set, List, Dict, Any, Callable
from time import time
import numpy as np
import json
from tqdm import tqdm
from torch import Tensor


class ConstrainedDecoding:
    def __init__(self, func_dict: List[Dict], llm: str) -> None:
        try:
            self.llm = Small_LLM_Model(llm)
        except OSError:
            print(f"\n\33[31m[ERROR]: {llm} llm not found\33[0m\n")
            exit()
        self.encode = self._legacy_encode_wrapper
        self.decode: Callable[[Tensor | List[int]], str] = self.llm.decode
        self.TOOL_CALL_TOKEN = 151657
        self._get_tokenize_function()
        self.func_dict = func_dict
        self.out: List[int] = []
        self.context: List[int] = []
        self.prompt_tokens: List[int] = []
        self.QUOTE_TOKEN = self.encode("\"")[0]
        self.BOOL_ALLOWED = self._get_bool_allowed()
        self.SCHEMA = """
{
\t"prompt": "<tool_call>",
\t"name": "<tool_call>",
\t"parameters": {<tool_call>}
}
"""
        self.SCHEMA_TOKENS = self.encode(self.SCHEMA)
        self._func_name_tokens: dict[str, list[int]] = {
            f["name"]: self.encode(f["name"])
            for f in self.func_dict
        }
        tmp = self.encode("\n")
        self.func_context = self.encode("Function list:\n")
        for i in self._func_name_tokens.values():
            self.func_context += i
            self.func_context += tmp

    def _legacy_encode_wrapper(self, text: str) -> List[int]:
        """Wrapper to extract a list from legacy encode function"""
        return (self.llm.encode(text)[0].tolist())

    def _get_tokenize_function(self) -> None:
        """Replace legacy by custom ones if possible"""
        try:
            with open(self.llm.get_path_to_vocab_file(), "r") as f:
                self.encode_dict = json.load(f)
            self.encode_dict["<tool_call>"] = self.TOOL_CALL_TOKEN
            self.decode_dict = {v: k for k, v in self.encode_dict.items()}
            self.SPACE_CHAR = self.decode_dict[self._legacy_encode_wrapper(
                " ")[0]]
            self.TAB_CHAR = self.decode_dict[self._legacy_encode_wrapper("\t")[
                0]]
            self.NEWLINE_CHAR = self.decode_dict[self._legacy_encode_wrapper(
                "\n")[0]]
            self.max_token_len = len(max(self.encode_dict, key=len))
            self.encode = self._vocab_encode
            self.decode = self._vocab_decode
            print("\n[STATUS]: Switching to custom tokenization fonctions")
        except (FileNotFoundError, PermissionError):
            print("\n\33[33m[WARN]: Cannot load vocabulary file\33[0m")
            print("[STATUS]: Switching to legacy tokenization fonctions\n")
            return

    def _vocab_decode(self, tokens: Tensor | List[int]) -> str:
        """Custom decode function"""
        buf = ""
        for token in tokens:
            buf += self.decode_dict[token]
        buf = buf.translate(str.maketrans(self.SPACE_CHAR +
                                          self.TAB_CHAR + self.NEWLINE_CHAR,
                                          " \t\n"))
        return (buf)

    def _vocab_encode(self, text: str) -> List[int]:
        """Custom encode function"""
        text = text.translate(str.maketrans(" \t\n", self.SPACE_CHAR +
                                            self.TAB_CHAR + self.NEWLINE_CHAR))
        out: List[int] = []
        dp = [float("inf")] * (len(text) + 1)
        dp[0] = 0
        back = [0] * (len(text) + 1)
        for i in range(0, len(dp)):
            if dp[i] == float("inf"):
                continue
            for j in range(i + 1, len(dp)):
                if (text[i:j] in self.encode_dict):
                    if (dp[i] + 1 < dp[j]):
                        dp[j] = dp[i] + 1
                        back[j] = i
        i = len(text)
        while (i != 0):
            out.append(self.encode_dict[text[back[i]:i]])
            i = back[i]
        return (out[::-1])

    def _get_bool_allowed(self) -> Set:
        """Get bool token"""
        allowed = set()
        for i in ["true", "false"]:
            allowed.add(self.encode(i)[0])
        return (allowed)

    def _choose_constrained_token(self, logits: List[float],
                                  allowed: Set) -> int:
        """Choose the highest probable token from allowed"""
        logits_np = np.asarray(logits, dtype=np.float64)
        allowed_np = np.fromiter(allowed, dtype=np.int64)
        return int(allowed_np[np.argmax(logits_np[allowed_np])])

    def _logsumexp(self, xs: List[float]) -> float:
        """Calculate the log of the sum of the exponantial of the xs list"""
        x = np.asarray(xs, dtype=np.float64)
        m = x.max()
        return float(m + np.log(np.exp(x - m).sum()))

    def _compute_avg_logprob(self, prob_list: List[float]) -> float:
        """Compute the average log probabilities"""
        if (len(prob_list) == 0):
            return (0)
        sum: float = 0
        for i in prob_list:
            sum += i
        return (sum / len(prob_list))

    def _choose_func(self) -> str:
        """Choose the best function in the context"""
        best_name = ""
        best_prob = float("-inf")

        root_logits = self.llm.get_logits_from_input_ids(self.context)
        for name, tokens in self._func_name_tokens.items():
            logprob_list = [float(root_logits[tokens[0]])]

            if len(tokens) > 1:
                tmp_tokens = list(self.context) + [tokens[0]]
                for t in tokens[1:]:
                    logits = self.llm.get_logits_from_input_ids(tmp_tokens)
                    logprob_list.append(float(logits[t]))
                    tmp_tokens.append(t)

            avg_prob = self._compute_avg_logprob(logprob_list)
            if (avg_prob > best_prob):
                best_prob = avg_prob
                best_name = name
        self.out += self.encode(best_name)
        return (best_name)

    def _get_number_param(self, cast: Callable) -> None:
        """Get a numerical parameter for the choosen function"""
        buf = ""
        while (True):
            logits = self.llm.get_logits_from_input_ids(self.context)
            nxt = self._get_max_logits_index(logits)
            try:
                cast(buf + self.decode([nxt]))
            except ValueError:
                if (len(buf) != 0 or self.decode([nxt]) != "-"):
                    break
            self.out.append(nxt)
            self.context.append(nxt)
            buf += self.decode([nxt])

    def _get_max_logits_index(self, logits: List[float]) -> int:
        """Get the index of the most probable token"""
        max_i = 0
        for i, v in enumerate(logits):
            if logits[max_i] < v:
                max_i = i
        return (max_i)

    def _get_string_param(self) -> None:
        """Get a string parameter for the choosen function"""
        self.context.append(self.QUOTE_TOKEN)
        self.out.append(self.QUOTE_TOKEN)
        while (True):
            logits = self.llm.get_logits_from_input_ids(self.context)
            nxt = self._get_max_logits_index(logits)
            if ('"' in self.decode([nxt])):
                tmp = self.decode([nxt]).split("\"")[0]
                tmp2 = self.encode(tmp)
                self.out += tmp2
                self.context += tmp2
                break
            self.out.append(nxt)
            self.context.append(nxt)
        self.context.append(self.QUOTE_TOKEN)
        self.out.append(self.QUOTE_TOKEN)

    def _get_bool_param(self) -> None:
        """Get a boolean parameter for the choosen function"""
        logits = self.llm.get_logits_from_input_ids(self.context)
        nxt = self._choose_constrained_token(logits, self.BOOL_ALLOWED)
        self.out.append(nxt)
        self.context.append(nxt)

    def _get_param(self, param_type: Dict[str, Any]) -> None:
        """Get a parameter for the choosen function"""
        match param_type["type"]:
            case "number" | "float":
                self._get_number_param(float)
            case "int" | "integer":
                self._get_number_param(int)
            case "bool" | "boolean":
                self._get_bool_param()
            case "string" | _:
                self._get_string_param()

    def _add_string(self, s: str) -> None:
        """Add a constant string to context and output"""
        token_list = self.encode(s)
        for i in token_list:
            self.out.append(i)
            self.context.append(i)

    def run(self, prompt: str) -> str:
        """Constrained decoding entry point"""
        prompt = json.dumps(prompt)[1:-1]
        self.context = self.encode(prompt)
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
                self.context += self.encode(
                    "\nBased on the prompt, the correct function is:\n"
                )
                t1 = time()
                func_name = self._choose_func()
                tqdm.write(
                    "[STATUS]: Function choosing done in"
                    f"{time() - t1: .3f} seconds")
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
                    tqdm.write(
                        f"[STATUS]: Get param {c + 1} done in "
                        f"{time() - t1:.3f} seconds")
                    if (c < len(func_context["parameters"]) - 1):
                        self._add_string(", ")
                count += 1
        result = self.decode(self.out)
        return (result)
