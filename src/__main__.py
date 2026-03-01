from .constrained_decoding import ConstrainedDecoding
import json
from typing import Dict
from time import time
import os


def exit_parsing(function_context: Dict):
    print("Function definition is incorect\n" + str(function_context))
    exit(1)


def validate_param(param: Dict) -> int:
    if (not isinstance(param, dict)):
        return (1)
    if (not param.get("type")):
        return (1)
    if (not isinstance(param["type"], str)):
        return (1)
    return (0)


def validate_parameters(params: Dict) -> int:
    for name, value in params.items():
        if (validate_param(value)):
            return (1)
    return (0)


with open("data/input/functions_definition.json", "r") as file:
    data = json.load(file)

for f in data:
    if (
        not f.get("name") or
        not f.get("description") or
        not f.get("parameters") or
        not f.get("returns")
    ):
        exit_parsing(f)
    if (
        not isinstance(f["name"], str) or
        not isinstance(f["description"], str)
    ):
        exit_parsing(f)
    if (validate_parameters(f["parameters"])):
        exit_parsing(f)
    if (validate_param(f["returns"])):
        exit_parsing(f)

with open("data/input/function_calling_tests.json") as file:
    prompts = json.load(file)


out = []
test = ConstrainedDecoding(data)
t = time()
for p in prompts:
    if (not p.get("prompt")):
        print(f"Cannot process {p}")
        continue
    print(p["prompt"])
    t1 = time()
    result = test.run(p["prompt"])
    print(result)
    out.append(json.loads(str(result)))
    print(f"Done in {time() - t1:.3f} seconds\n")
process_time = time() - t
print(f"\nTotal processing time: {process_time // 60:.0f}m "
      f"{process_time % 60:.0f}s")

os.makedirs("data/output", exist_ok=True)
with open("data/output/function_calling_results.json", "w",
          encoding="utf-8") as f:
    json.dump(out, f, indent=4, ensure_ascii=False)
