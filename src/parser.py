from typing import Dict, NoReturn, Any, Tuple, List
import json
from pydantic import BaseModel, Field, model_validator


class Validate(BaseModel):
    function_defs: List[Dict[str, Any]] = Field(...)
    prompts: List[Dict[str, Any]] = Field(...)

    @model_validator(mode="after")
    def validate(self):
        self.validate_defs()
        return (self)

    def validate_defs(self):
        for f in self.function_defs:
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

    def exit_parsing(self, function_context: Dict) -> NoReturn:
        print("Function definition is incorect\n" + str(function_context))
        exit()

    def validate_param(self, param: Dict) -> int:
        if (not isinstance(param, dict)):
            return (1)
        if (not param.get("type")):
            return (1)
        if (not isinstance(param["type"], str)):
            return (1)
        if (param["type"] not in ["number", "string", "bool", "boolean"]):
            return (1)
        return (0)

    def validate_parameters(self, params: Dict) -> int:
        for name, value in params.items():
            if (self.validate_param(value)):
                return (1)
        return (0)


class Parser:
    def __init__(self, functions_defs_path: str, prompt_path: str) -> None:
        self.functions_defs_path = functions_defs_path
        self.prompt_path = prompt_path

    def extract_functions(self) -> List[Dict[str, Any]]:
        try:
            with open(self.functions_defs_path, "r") as file:
                try:
                    data: List[Dict[str, Any]] = json.load(file)
                except Exception:
                    print(f"{self.functions_defs_path} is not a valid JSON")
                    exit()
        except FileNotFoundError:
            print(f"{self.functions_defs_path} file not found")
            exit()
        except PermissionError:
            print(f"{self.functions_defs_path} file cannot be opened")
            exit()

        return (data)

    def extract_prompt(self) -> List[Dict[str, Any]]:
        try:
            with open(self.prompt_path, "r") as file:
                prompts: List[Dict[str, Any]] = json.load(file)
        except FileNotFoundError:
            print(f"{self.prompt_path} file not found")
            exit()
        except PermissionError:
            print(f"{self.prompt_path} file cannot be opened")
            exit()
        return (prompts)

    def run(self) -> Validate:
        return (Validate(
            function_defs=self.extract_functions(),
            prompts=self.extract_prompt()
        ))
