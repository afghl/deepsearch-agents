import asyncio
from typing import Type, TypeVar, cast, overload
from agents import TResponseInputItem
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from deepsearch_agents.conf import get_configuration

T = TypeVar("T", bound=BaseModel)


config = get_configuration()

client = AsyncOpenAI(
    base_url=config.openai_base_url,
    api_key=config.openai_api_key,
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
    if output_type is not None:
        res = await client.beta.chat.completions.parse(
            model=model,
            messages=messages,  # type: ignore
            response_format=output_type,
        )
        return cast(T, res.choices[0].message.parsed)
    else:
        res = await client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
        )
        return cast(str, res.choices[0].message.content)


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
