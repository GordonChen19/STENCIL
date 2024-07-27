from pydantic import BaseModel, Field
from typing import List

class Entity(BaseModel):
    modified_objects: List[str] = Field(description = '''Description of each entity that is modified in the image''')
    context_objects: List[str] = Field(description = '''Description  of each entity that is necessary for the context of the modification''')
    description: str = Field(description = '''Rewrite of the prompt to include the relevant elements''')