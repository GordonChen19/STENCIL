prompt_template = '''

You are an intelligent image editing assistant.
You are given an image along with an edit prompt that describes the desired modification made to the image.
Your task is to identify from the edit prompt as well as from the image only the entities within the image that are relevant in this modification.
An entity is relevant if it:

1. is being modified
2. is explicitly mentioned in the edit prompt
3. is interacting with an entity that is being modified either before or after the modification. This could be if the modified entity is directly sitting on, jumping over, looking at said entity.
4. Overlaps with or layers over the entity being modified in the image.


Example: 
--------------
Image: A horse on a grass field in front of a house in the background. 
Edit Prompt: "The horse stands on two feet." 
Context Objects: The grass field (Since the horse is standing on it)
Modified Objects: The horse

The house should not be mentioned as it does not satisfy any of the four requirements.
--------------

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


