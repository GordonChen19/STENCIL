from pydantic import BaseModel, Field
from typing import List

class Entity(BaseModel):
    modified_objects: List[str] = Field(description = '''Description of each element that is modified in the image''', examples = ["Dog", "Bench"])