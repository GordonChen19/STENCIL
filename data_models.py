from pydantic import BaseModel, Field
from typing import List

class Entity(BaseModel):
    name: List[str] = Field(description = '''Names of each entity''')
    before: str = Field(description = '''Concise description of the image prior to it being edited''')
    after: str = Field(description = '''Concise description of the image after it has been edited''')
