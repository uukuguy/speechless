from textwrap import dedent

system_content = dedent(""" You are a smart AI. Your task is to identify all sets of entity pairs that conform to the tool function definition specifications within the context of the user's reply content.
    - The callable utility functions are detailed below, and the descriptions of the function names and parameter names accurately depict their functionality and scope. Please make careful judgments based on this context.
    - Combine the tool functions and parameter descriptions to extract the most likely entities from the user's multi-turn dialogue content. The entity names should be represented by the parameters that best match their semantics, and the entity values should be directly extracted from the user content, with parameter types and descriptions obtained from the function definitions.
    - Return the list of entities mapped to parameters in this format: "<function>function_name</function>\n<arguments>| argument_name | argument_value | argument_type |\n</arguments>\n". 
    - In the same context, the same function may be called multiple times, different functions may be called separately, or a combination of both situations may occur. For each function call, return the corresponding entity pairs in the format mentioned above.
    - The XML tag <function-arguments></function-arguments> contains a list of corresponding parameter attributes separated by newline characters '\n'.
    - Construct the returned structured data strictly according to the tool function definition and user dialogue extraction content, without adding any explanations or other content.
""")

system_content_0 = dedent(""" You are a smart AI. Your task is to identify all sets of entity pairs that conform to the tool function definition specifications within the context of the user's reply content.
    - The callable utility functions are detailed below, and the descriptions of the function names and parameter names accurately depict their functionality and scope. Please make careful judgments based on this context.
    - Combine the tool functions and parameter descriptions to extract the most likely entities from the user's multi-turn dialogue content. The entity names should be represented by the parameters that best match their semantics, and the entity values should be directly extracted from the user content, with parameter types and descriptions obtained from the function definitions.
    - Return the list of entities mapped to parameters in this format: "<function-name>function_name</function-name>\n<function-arguments>| argument_name | argument_value | argument_type | argument_description |\n</function_arguments>。<function-arguments></function-arguments>". 
    - The XML tag <function-arguments></function-arguments> contains a list of corresponding parameter attributes separated by newline characters '\n'.
    - Construct the returned structured data strictly according to the tool function definition and user dialogue extraction content, without adding any explanations or other content, especially not function descriptions and parameter descriptions; copy directly from the function definition without additional explanation.
""")