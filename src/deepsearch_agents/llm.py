import asyncio
from dataclasses import dataclass
from typing import Generic, Type, TypeVar, cast, overload
from agents import (
    AgentOutputSchema,
    ModelResponse,
    ModelSettings,
    ModelTracing,
    OpenAIChatCompletionsModel,
    TResponseInputItem,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, TypeAdapter

from deepsearch_agents.conf import MY_OPENAI_API_KEY, OPENAI_BASE_URL

T = TypeVar("T", bound=BaseModel)


client = AsyncOpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=MY_OPENAI_API_KEY,
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
