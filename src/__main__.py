from .constrained_decoding import ConstrainedDecoding
import json
from time import time
from .parser import Parser
from argparse import ArgumentParser
from pathlib import Path


def get_files_paths(args):
    file_paths = {}
    if (args.functions_definition):
        file_paths["definitions"] = args.functions_definition
    else:
        file_paths["definitions"] = "data/input/functions_definition.json"
    if (args.input):
        file_paths["input"] = args.input
    else:
        file_paths["input"] = "data/input/function_calling_tests.json"
    if (args.output):
        file_paths["output"] = args.output
    else:
        file_paths["output"] = "data/output/function_calling_results.json"
    return (file_paths)


def export_result(file_paths) -> None:
    output = Path(file_paths["output"])
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

    parser.add_argument("--functions_definition", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    file_paths = get_files_paths(args)

    parser = Parser(file_paths["definitions"], file_paths["input"])
    funcs, prompts = parser.run()

    out = []
    test = ConstrainedDecoding(funcs)
    t = time()
    for p in prompts:
        if (not p.get("prompt")):
            print(f"Cannot process {p}")
            continue
        print(p["prompt"])
        t1 = time()
        result = test.run(p["prompt"])
        print(result)
        try:
            out.append(json.loads(str(result)))
            print("\033[32mSUCCESS: Valid JSON format\033[0m")
        except Exception:
            print("\033[31mFAILED: Invalid JSON format\033[0m")
        print(f"Done in {time() - t1:.3f} seconds\n")
    process_time = time() - t
    print(f"\nTotal processing time: {process_time // 60:.0f}m "
          f"{process_time % 60:.0f}s")
    export_result(file_paths)
