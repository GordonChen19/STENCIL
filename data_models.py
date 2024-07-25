from pydantic import BaseModel, Field
from typing import List

class Entity(BaseModel):
    modified_objects: List[str] = Field(description = '''Names of each entity that are modified in the image''')
    context_objects: List[str] = Field(description = '''Names of each entity that are necessary for the context of the modification''')
    before: str = Field(description = '''Concise description of the image prior to it being edited''')
    after: str = Field(description = '''Concise description of the image after it has been edited''')
