from typing import List

from agents import Usage
from openai import AsyncOpenAI
from pydantic import BaseModel

from deepsearch_agents.conf import get_configuration

client = AsyncOpenAI()


config = get_configuration()

client = AsyncOpenAI(
    base_url=config.openai_base_url if config else None,
    api_key=config.openai_api_key,
)


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    usage: Usage


async def get_embedding(model: str, text: str) -> EmbeddingResponse:
    model_conf = config.get_model_config(model)
    response = await client.embeddings.create(
        model=model_conf.model_name,
        input=text,
    )
    return EmbeddingResponse(
        embedding=response.data[0].embedding,
        usage=Usage(
            input_tokens=response.usage.prompt_tokens,
            total_tokens=response.usage.total_tokens,
        ),
    )
