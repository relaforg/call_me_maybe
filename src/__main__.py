from .constrained_decoding import ConstrainedDecoding
from pprint import pprint
import json


# func_dict = {
#     "fn_add_numbers": {
#         "description": "Add two numbers together and return their sum.",
#         "parameters": {
#             "a": {
#                 "type": "number"
#             },
#             "b": {
#                 "type": "number"
#             }
#         },
#         "returns": {
#             "type": "number"
#         }
#     },
#     "fn_greet": {
#         "description": "Generate a greeting message for a person by name.",
#         "parameters": {
#             "name": {
#                 "type": "string"
#             }
#         },
#         "returns": {
#             "type": "string"
#         }
#     },
#     "fn_reverse_string": {
#         "description": "Reverse a string and return the reversed result.",
#         "parameters": {
#             "s": {
#                 "type": "string"
#             }
#         },
#         "returns": {
#             "type": "string"
#         }
#     },
#     "fn_get_square_root": {
#         "description": "Calculate the square root of a number.",
#         "parameters": {
#             "a": {
#                 "type": "number"
#             }
#         },
#         "returns": {
#             "type": "number"
#         }
#     },
#     "fn_substitute_string_with_regex": {
#         "description": "Replace all occurrences matching \
# a regex pattern in a string.",
#         "parameters": {
#             "source_string": {
#                 "type": "string"
#             },
#             "regex": {
#                 "type": "string"
#             },
#             "replacement": {
#                 "type": "string"
#             }
#         },
#         "returns": {
#             "type": "string"
#         }
#     }
# }

with open("data/input/functions_definition.json", "r") as file:
    data = json.load(file)

prompt = "What is the sum of 40 and 2?"
test = ConstrainedDecoding(data)
print(test.run(prompt))
