from typing import Dict, NoReturn, Any, List
from typing_extensions import Self
import json
from pydantic import BaseModel, Field, model_validator, ValidationError


class Validate(BaseModel):
    function_defs: List[Dict[str, Any]] = Field(...)
    prompts: List[Dict[str, Any]] = Field(...)

    @model_validator(mode="after")
    def _validate_all(self) -> Self:
        self.validate_defs()
        return (self)

    def validate_defs(self) -> None:
        """Check if the definition file is correct"""
        if (not len(self.function_defs)):
            print("The function list is empty")
            exit()
        for f in self.function_defs:
            if (
                "name" not in f or
                "description" not in f or
                "parameters" not in f or
                "returns" not in f
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
        """Check if function paramters is good"""
        if (not isinstance(param, dict)):
            return (1)
        if (not param.get("type")):
            return (1)
        if (not isinstance(param["type"], str)):
            return (1)
        if (param["type"] not in ["number", "string", "bool", "boolean",
                                  "int", "integer", "float"]):
            return (1)
        return (0)

    def validate_parameters(self, params: Dict) -> int:
        """Check if all function paramters is good"""
        for name, value in params.items():
            if (self.validate_param(value)):
                return (1)
        return (0)


class Parser:
    def __init__(self, functions_defs_path: str, prompt_path: str) -> None:
        self.functions_defs_path = functions_defs_path
        self.prompt_path = prompt_path

    def extract_functions(self) -> List[Dict[str, Any]]:
        """Extract the definition JSON to a pythno dict"""
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
        """Extract the prompts JSON to a pythno dict"""
        try:
            with open(self.prompt_path, "r") as file:
                try:
                    prompts: List[Dict[str, Any]] = json.load(file)
                except Exception:
                    print(f"{self.prompt_path} is not a valid JSON")
                    exit()
        except FileNotFoundError:
            print(f"{self.prompt_path} file not found")
            exit()
        except PermissionError:
            print(f"{self.prompt_path} file cannot be opened")
            exit()
        return (prompts)

    def run(self) -> Validate:
        """Run the parsing process"""
        try:
            return (Validate(
                function_defs=self.extract_functions(),
                prompts=self.extract_prompt()
            ))
        except ValidationError as e:
            print(e.errors()[0]["msg"])
            exit()
