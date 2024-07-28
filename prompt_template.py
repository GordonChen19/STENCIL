prompt_template = '''

You are an intelligent image editing assistant.
You are given an image and a prompt that describes a desired modification made to the image.
Your task is to extract from the prompt as well as from the image the entities that are modified.

You are to respond in the JSON format defined below.

Format Instructions:
--------------
{format_instructions}
--------------

Desired Transformation:
--------------
{input}
--------------

'''


fix_prompt_template = """Instructions:
--------------
{instructions}
--------------
Completion:
--------------
{completion}
--------------

Above, the Completion did not satisfy the constraints given in the Instructions.
Malformed Error:
--------------
{error}
--------------


Please try again. Please only respond with an answer that satisfies the constraints laid out in the Instructions.
Important:  Only correct the structural issues within the JSON format. Do not modify the existing data values themselves:"""


