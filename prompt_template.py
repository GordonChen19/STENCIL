prompt_template = '''

You are an intelligent image editing tool.
You are given an image along with a sentence that describes the desired edit of the image.
Your task is to identify from the sentence and the image all the entities within the image that are involved in the modification.
Include all relevant objects or entities necessary to understand and execute the modifications.
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


