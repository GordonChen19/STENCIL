
from langchain.chains import TransformChain
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import PromptTemplate
from gpt_api import chat_completion, vision_completion
from prompt_template import fix_prompt_template

def fix_chain_fun(inputs):
        fix_prompt = PromptTemplate.from_template(fix_prompt_template)
        fix_prompt_str = fix_prompt.invoke({'instructions':inputs['instructions'],
                                            'completion':inputs['completion'],
                                            'error':inputs['error']}).text
    
        completion = chat_completion(fix_prompt_str)
    
        return {"completion": completion}

fix_chain = TransformChain(
    input_variables=["instructions", "completion", "error"], output_variables=["completion"], transform=fix_chain_fun
)  


def extraction_chain(input, data_model, prompt_template, file_path):

    parser = PydanticOutputParser(pydantic_object=data_model)
    fix_parser = OutputFixingParser(parser=parser, retry_chain=fix_chain, max_retries=1)

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["input"],
        partial_variables={"format_instructions": parser.get_format_instructions()})
    
    prompt_str = prompt.invoke({"input":input}).to_string()

    response = vision_completion(prompt_str, file_path)
    return fix_parser.invoke(response).dict()
