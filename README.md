This project has been created as part of the 42 curriculum by *relaforg*

# Call Me Maybe

## Description

Call Me Maybe serve as an introduction in LLM integration in programming project.
The goal of the project is to create a calling function system using constrained
decoding.

> [!NOTE]
> Constrained decoding is a technique that guides the model’s output
> token-by-token to guarantee valid structure, without relying on prompting alone.

## Instructions

To install depedencies :

```bash
make install
```

To run the program :

```bash
make run ARGS="<possible arguments>"
```

## Ressources

- [A Guide to Structured Generation Using Constrained Decoding](https://www.aidancooper.co.uk/constrained-decoding/)
- [Building Intuition on Log Probabilities in Language Models](https://medium.com/ai-assimilating-intelligence/building-intuition-on-log-probabilities-in-language-models-8fd00f34c03c)
- ChatGPT

## Algorithm explanation

In order to garantee 100% valid JSON generation, I force the LLm to follow a
schema. The schema contains a particular token: "<tool_call>".
When a "<tool_call>" token is encountered the LLM can exit the schema to
generate the variable content such as the function_name or arguments.

```python
SCHEMA = """
{
\t"prompt": "<tool_call>",
\t"name": "<tool_call>",
\t"parameters": {<tool_call>}
}
"""
```

## Design decisions

In order to make the LLM choose the correct function to answer the prompt,
for each function, I feed the LLM with the function name and calculate the
average probality with which the LLM would have choosen that function name.
Thus, the chosen function is the one that was most likely to have been selected
directly by the LLM.

## Performance analysis

The program achieve 100% valid JSON output and 99%+ valid function and parameters.
In average, 1 minute and 10 secondes to process the 11 test prompts.
So, with simple input, the program can be considered reliable.

## Challenges faced

I found it very dificult to understand where were the problem as the LLM act
like a black-0box where it is impossible to know what is happening.
But it was really intresting to discover this programming challenge.

## Testing strategy

To test the program I used the test given in the subject and some personal edge
cases described in the subject (big numbers, ambiguous prompts, ...).

## Example usage

```bash
make run ARGS="--input input.json --output output.json --functions_definition func_defs.json"
```
