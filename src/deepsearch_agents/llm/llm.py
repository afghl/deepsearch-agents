import asyncio
from typing import Type, TypeVar, cast, overload
from agents import TResponseInputItem
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from deepsearch_agents.conf import ModelConfig, get_configuration

T = TypeVar("T", bound=BaseModel)


config = get_configuration()

client = AsyncOpenAI(
    base_url=config.openai_base_url if config else None,
    api_key=config.get_openai_api_key(),
)


@overload
async def get_response(
    model: str,
    input: str | list[TResponseInputItem],
    output_type: None = None,
    system_instructions: str | None = None,
) -> str: ...
@overload
async def get_response(
    model: str,
    input: str | list[TResponseInputItem],
    output_type: Type[T],
    system_instructions: str | None = None,
) -> T: ...
async def get_response(
    model: str,
    input: str | list[TResponseInputItem],
    output_type: Type[T] | None = None,
    system_instructions: str | None = None,
) -> T | str:
    messages = []
    if system_instructions:
        messages.append({"role": "system", "content": system_instructions})
    if isinstance(input, str):
        messages.append({"role": "user", "content": input})
    else:
        messages.extend(input)
    model_conf = config.get_model_config(model)

    if openai_model(model_conf.model_name):
        return await _openai_chat_completion(model_conf, messages, output_type)
    else:
        raise ValueError(f"Unsupported model: {model}")


async def _openai_chat_completion(
    model_conf: ModelConfig,
    messages: list[dict[str, str]],
    output_type: Type[T] | None = None,
) -> T | str:
    if output_type is not None:
        response = await client.beta.chat.completions.parse(
            model=model_conf.model_name,
            messages=messages,  # type: ignore
            response_format=output_type,
            temperature=model_conf.temperature,
            max_tokens=model_conf.max_tokens,
            top_p=model_conf.top_p,
        )
        return cast(T, response.choices[0].message.parsed)
    else:
        response = await client.chat.completions.create(
            model=model_conf.model_name,
            messages=messages,  # type: ignore
            temperature=model_conf.temperature,
            max_tokens=model_conf.max_tokens,
            top_p=model_conf.top_p,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Empty response from OpenAI")
        return content


def openai_model(model_name: str) -> bool:
    return model_name.startswith("gpt-")


class Answer(BaseModel):
    answer: str = Field(description="Answer to the question")
    source: str = Field(description="Source of the answer")


async def main():
    res = await get_response("gpt-4o", "hello world")
    print(res.capitalize())
    ans = await get_response("gpt-4o", "fix this code", Answer)
    print(ans.answer)


if __name__ == "__main__":
    asyncio.run(main())
