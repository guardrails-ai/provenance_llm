import numpy as np
import pytest
from guardrails import Guard
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from validator import ProvenanceLLM

# Setup text sources
SOURCES = [
    "The sun is a star.",
    "The sun rises in the east and sets in the west.",
    "Sun is the largest object in the solar system, and all planets revolve around it.",
]

# Load model for embedding function
MODEL = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Create embed function
def embed_function(sources: list[str]) -> np.array:
    return MODEL.encode(sources)

def test_sources_one_chunk():
    validator = ProvenanceLLM(
        validation_method="full",
        llm_callable="gpt-4o-mini",
        top_k=1,
    )
    value = "The sun is a dog."
    sources = ["The sun is a star."]
    response = validator.validate(value, metadata={"sources": sources, "embed_function": embed_function})

def test_sources_two_chunks():
    validator = ProvenanceLLM(
        validation_method="full",
        llm_callable="gpt-4o-mini",
        top_k=1,
    )
    value = "The sun is a dog."
    sources = ["The sun is a star.", "Jupiter is the largest planet in the solar system."]
    response = validator.validate(value, metadata={"sources": sources, "embed_function": embed_function})
