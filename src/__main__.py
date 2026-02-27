from .constrained_decoding import ConstrainedDecoding
import json
from typing import Dict


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

prompt = "What is the sum of 40 and 2?"
test = ConstrainedDecoding(data)
print(test.run(prompt))
