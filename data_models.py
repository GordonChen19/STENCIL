from pydantic import BaseModel, Field

class Entity(BaseModel):
    name: str = Field(description = '''Name of the entity''', example = 'Dog')
    