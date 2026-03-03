from typing import Dict
import json


class Parser:
    def __init__(self, functions_defs_path: str, prompt_path: str) -> None:
        self.functions_defs_path = functions_defs_path
        self.prompt_path = prompt_path

    def exit_parsing(self, function_context: Dict):
        print("Function definition is incorect\n" + str(function_context))
        exit(1)

    def validate_param(self, param: Dict) -> int:
        if (not isinstance(param, dict)):
            return (1)
        if (not param.get("type")):
            return (1)
        if (not isinstance(param["type"], str)):
            return (1)
        return (0)

    def validate_parameters(self, params: Dict) -> int:
        for name, value in params.items():
            if (self.validate_param(value)):
                return (1)
        return (0)

    def extract_functions(self):
        try:
            with open(self.functions_defs_path, "r") as file:
                data = json.load(file)
        except FileNotFoundError:
            print(f"{self.functions_defs_path} file not found")
            exit()
        except PermissionError:
            print(f"{self.functions_defs_path} file cannot be opened")
            exit()

        for f in data:
            if (
                not f.get("name") or
                not f.get("description") or
                not f.get("parameters") or
                not f.get("returns")
            ):
                self.exit_parsing(f)
            if (
                not isinstance(f["name"], str) or
                not isinstance(f["description"], str)
            ):
                self.exit_parsing(f)
            if (self.validate_parameters(f["parameters"])):
                self.exit_parsing(f)
            if (self.validate_param(f["returns"])):
                self.exit_parsing(f)
        return (data)

    def extract_prompt(self):
        try:
            with open(self.prompt_path, "r") as file:
                prompts = json.load(file)
        except FileNotFoundError:
            print(f"{self.prompt_path} file not found")
            exit()
        except PermissionError:
            print(f"{self.prompt_path} file cannot be opened")
            exit()
        return (prompts)

    def run(self):
        return (self.extract_functions(), self.extract_prompt())
