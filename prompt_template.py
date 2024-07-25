prompt_template = '''

You are an intelligent image editing tool.
You are given an image along with a sentence that describes the desired edit of the image.
Your task is to identify from the sentence as well as from the image all the entities within the image that are involved in the modification.
Include all relevant objects or entities necessary to understand and execute the modifications, even if they are not directly mentioned in the sentence.
Some objects or entities from the image may be necessary for the context of the modification even if they are not modified.  

Example: I have an image of a horse on a grass field. The sentence is "The horse stands on two feet." The entities involved in the modification are the horse and the grass field. The grass field is necessary for the context of the modification since the horse is standing on it.

In addition, provide a concise description of the image before and after the modification.
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


