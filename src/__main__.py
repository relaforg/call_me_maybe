from .constrained_decoding import ConstrainedDecoding
import json
from time import time
from .parser import Parser
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tqdm import tqdm
from typing import Dict


def export_result(output_file: str) -> None:
    output = Path(output_file)
    directory = output.parent
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4, ensure_ascii=False)
    except FileNotFoundError:
        print(f"{output} file not found")
    except PermissionError:
        print(f"{output} connot be writen")


if (__name__ == "__main__"):
    parser = ArgumentParser()

    parser.add_argument("-functions_definition", type=str,
                        default="data/input/functions_definition.json")
    parser.add_argument("-input", type=str,
                        default="data/input/function_calling_tests.json")
    parser.add_argument("-output", type=str,
                        default="data/output/function_calling_results.json")
    parser.add_argument("-llm", type=str, default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()

    data = Parser(args.functions_definition, args.input).run()
    funcs, prompts = data.function_defs, data.prompts

    out = []
    test = ConstrainedDecoding(funcs, args.llm)
    print(f"\nProcessing using \033[33m'{args.llm}'\033[0m\n")
    for p in tqdm(prompts, desc="Processing", leave=True):
        if (not p.get("prompt") or not isinstance(p.get("prompt"), str)):
            tqdm.write(f"\033[33m[WARN]: Cannot process {p}\033[0m\n")
            continue
        tqdm.write("\033[38;5;177m[PROMPT]: " + p["prompt"] + "\033[0m")
        t1 = time()
        result = test.run(p["prompt"])
        tqdm.write(result)
        try:
            out.append(json.loads(str(result)))
            tqdm.write("\033[32m[SUCCESS]: Valid JSON format\033[0m")
        except Exception:
            tqdm.write("\033[31m[FAILED]: Invalid JSON format\033[0m")
        tqdm.write(f"Done in {time() - t1:.3f} seconds\n")
    export_result(args.output)
