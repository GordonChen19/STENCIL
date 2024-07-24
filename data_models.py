from pydantic import BaseModel, Field
from typing import List

class Entity(BaseModel):
    name: List[str] = Field(description = '''Names of each entity''')
    