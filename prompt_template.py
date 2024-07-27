prompt_template = '''

You are an intelligent image editing assistant.
You are given an image and a prompt that describes the desired modification made to the image.
Your task is to extract from the prompt as well as from the image the elements relevant to the modification.
In addition, you are to rewrite the prompt such that it also includes the relevant elements.
An relevant element is either:

1. directly modified by the prompt
2. interacting with the element being modified PRIOR TO modification. The could be an element that the modified element IS CURRENTLY sitting on, jumping over, looking at, etc.
3. interacting with the element being modified AFTER modification. This could be an element that the modified element WILL BE sitting on, jumping over, looking at, etc.
4. explicitly mentioned in the prompt
5. overlaps with or layers over the entity being modified in the image.

Example: 
--------------
Image: A horse on a grass field in front of a house in the background. 
Edit Prompt: "The horse stands on two feet." 
Context Objects: The grass field (Since the horse is standing on it)
Modified Objects: The horse

The house should not be mentioned as it does not satisfy any of the four requirements.
--------------

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


