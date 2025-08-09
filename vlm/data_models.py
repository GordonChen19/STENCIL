from pydantic import BaseModel, Field
from typing import List

class Entity(BaseModel):
    augmented_prompt: str = Field(description = '''Augmented Text Prompt''')