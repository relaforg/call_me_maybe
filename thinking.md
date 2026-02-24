# Call Me Maybe

> [!IMPORTANT]
>The function calling system doesn’t answer the question directly.
>Instead, it provides the tools to solve it: the right function name and
>the correct arguments with proper types.

## Constrained Decoding

>A technique that guides the model’s output token-by-token to guarantee
>valid structure, without relying on prompting alone.

[A Guide to Structured Generation Using Constrained Decoding](https://www.aidancooper.co.uk/constrained-decoding/)
Generate a schema

```python
schema = """
{\n
\t"prompt": <The current prompt>,
\t"name": <The function name>
\t"parameters": {<parameters and value depending on function choosen>}
}\n
"""
```

To choose a function:
I have can do something like backtracking ?
Or I can just go with the best and go from there

## Input Files

In data/input :

- *function_calling_tests.json* -> contains a JSON array of natural language
                                 prompts that your system must process
    Example:

    ```json
    [
        "What is the sum of 2 and 3?",
        "Reverse the string 'hello'",
        "Calculate the factorial of 5"
    ]
    ```

- function_definitions.json -> contains the available function your system
                               can call
    Example:

    ```json
        [
            {
                "name": "fn_add_numbers",
                "description": "Add two numbers",
                "parameters":
                {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "returns": {"type": "number"}
            },
            {
                "name": "fn_reverse_string",
                "description": "Reverse a string",
                "parameters":
                {
                    "s": {"type": "string"}
                },
                "returns": {"type": "string"}
            }
        ]
    ```

## Output

The output must rescpect a schema:

- "prompt": str -> The original natural-language request
- "fn_name": str -> The name of the function to call
- args: object -> All required arguments with the correct types

    Example :

    ```json
    [
        {
            "prompt": "What is the sum of 2 and 3?",
            "fn_name": "fn_add_numbers",
            "args": {"a": 2.0, "b": 3.0}
        },
        {
            "prompt": "Reverse the string 'hello'",
            "fn_name": "fn_reverse_string",
            "args": {"s": "hello"}
        }
    ]
    ```

- The file generated must valid JSON  (no trailing commas, no comments)
- Keys and types must match the schema in function_definitions.json exactly
- No extra keys or prose are allowed anywhere in the output
- All required arguments must be present
- Argument types must match the function definition (number, string, boolean, etc.)
