import asyncio
from typing import Type, TypeVar, Union, cast, overload, Generic
from agents import TResponseInputItem, Usage
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from deepsearch_agents.log import logger
from deepsearch_agents.conf import ModelConfig, get_configuration
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from openai.types.chat.chat_completion import ChatCompletion

# M = TypeVar("M", bound=BaseModel)
T = TypeVar("T", bound=Union[str, BaseModel])


config = get_configuration()

client = AsyncOpenAI(
    base_url=config.openai_base_url if config else None,
    api_key=config.openai_api_key,
)


class LLMResponse(BaseModel, Generic[T]):
    response: T
    usage: Usage


@overload
async def get_response(
    model: str,
    input: str | list[TResponseInputItem],
    output_type: None = None,
    system_instructions: str | None = None,
) -> LLMResponse[str]: ...


@overload
async def get_response(
    model: str,
    input: str | list[TResponseInputItem],
    output_type: Type[T],
    system_instructions: str | None = None,
) -> LLMResponse[T]: ...


async def get_response(
    model: str,
    input: str | list[TResponseInputItem],
    output_type: Type[T] | None = None,
    system_instructions: str | None = None,
) -> LLMResponse[T]:
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
) -> LLMResponse[T]:
    if output_type is not None:
        response = await client.beta.chat.completions.parse(
            model=model_conf.model_name,
            messages=messages,  # type: ignore
            response_format=output_type,
            temperature=model_conf.temperature,
            max_tokens=model_conf.max_tokens,
            top_p=model_conf.top_p,
        )
        # Extract usage information from the response
        usage = _get_usage(response)
        # Return the parsed response
        return LLMResponse(response=response.choices[0].message.parsed, usage=usage)  # type: ignore
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
        usage = _get_usage(response)
        return LLMResponse(response=content, usage=usage)  # type: ignore


def _get_usage(response: ParsedChatCompletion | ChatCompletion) -> Usage:
    if response.usage is None:
        logger.error("response.usage is None")
        return Usage()
    else:
        return Usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )


_openai_models = [
    "gpt-4o",
    "gpt-4o-mini",
    "o1-preview",
    "o1",
    "o1-mini",
    "o1-preview",
    "o3-mini",
    "o3-preview",
]


def openai_model(model_name: str) -> bool:
    return model_name in _openai_models


class Answer(BaseModel):
    answer: str = Field(description="Answer to the question")
    source: str = Field(description="Source of the answer")


async def main():
    res = await get_response("rewrite", "hello world")
    print(res.response.capitalize())
    ans = await get_response("rewrite", "Why is the sky blue?", Answer)
    print(ans.response.answer)


if __name__ == "__main__":
    asyncio.run(main())
