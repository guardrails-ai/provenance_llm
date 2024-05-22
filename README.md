## Overview

| Developed by | Guardrails AI |
| Date of development | Feb 15, 2024 |
| Validator type | RAG |
| Blog | https://www.guardrailsai.com/blog/reduce-ai-hallucinations-provenance-guardrails |
| License | Apache 2 |
| Input/Output | Output |

## Description

This validator uses an LLM callable to evaluate the generated text against the provided contexts. In order to use this validator, you must provide one of the following in the `metadata` field while calling the validator:
- A `query_function`: `query_function` takes the LLM-generated text string as input and returns a list of relevant chunks. The list should be sorted in ascending order by the distance between the chunk and the LLM-generated text.
- `sources` and `embed_function`: `sources` is a list of strings containing the text that the LLM attribute is attributed against. The `embed_function` should take a string or a list of strings as input and return a np array of floats. The vector should be normalized to unit length.

Below is a step-wise breakdown of how the validator works:

1. The list of sources is chunked based on user's parameters.
2. Each source chunk is embedded and stored in a vector database or an in-memory embedding store.
3. The LLM generated output is chunked based on user-specified parameters.
4. Each LLM output chunk is embedded by the same model used for embedding source chunks.
5. The `k` nearest source chunks are determined for the LLM output chunk.
6. To evaluate whether the `k` nearest source chunks support the LLM output chunk, a call to an LLM is made to get a prediction.

### Intended use

The primary intended use is for RAG applications to check if a text is hallucinated by establishing a source (i.e. provenance) for any LLM generated text. Out-of-scope use cases are general question answering without RAG or text grounding. Use this in combination with traditional RAG to achieve Retrieval-Augmented (Validated) Generation [RAVG].

## Requirements

* Dependencies:
    - `litellm`
    - `numpy`
    - `nltk`
    - `tenacity`
	- guardrails-ai>=0.4.0

* To use in an example: 
    - `sentence-transformers`
    - `chromadb`

* Foundation model access keys: 
    - Yes (depending on which model is used for embeddings)

## Installation

```bash
guardrails hub install hub://guardrails/provenance_llm
```

## Usage Examples

### Validating string output via Python

In this example, we apply the validator to a string output generated by an LLM.

```python
# Import Guard and Validator
from guardrails.hub import ProvenanceLLM
from guardrails import Guard
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "This example requires the `sentence-transformers` package. "
        "Install it with `pip install sentence-transformers`, and try again."
    )

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

# Use the Guard with the validator
guard = Guard().use(
    ProvenanceLLM,
    validation_method="sentence",
    llm_callable="gpt-3.5-turbo",
    top_k=3,
    on_fail="exception",
)


# Test passing response
guard.validate(
    """
    The sun is a star that rises in the east and sets in the west.
    """,
    metadata={"sources": SOURCES, "embed_function": embed_function, "pass_on_invalid": True},
)

try:
    # Test failing response
    guard.validate(
        """
        Pluto is the farthest planet from the sun.
        """,  # This sentence is not "false", but is still NOT supported by the sources
        metadata={"sources": SOURCES, "embed_function": embed_function},
    )
except Exception as e:
    print(e)
```
Output:
```console
Validation failed for field with errors: None of the following sentences in your response are supported by provided context:
- Pluto is the farthest planet from the sun.
```

# API Reference

**`__init__(self, validation_method="sentence", llm_callable="gpt-3.5-turbo", top_k=3, max_tokens=2, on_fail="noop")`**
<ul>

Initializes a new instance of the Validator class.

**Parameters:**

- **`validation_method`** _(str:)_ Whether to validate at the sentence level or over the full text. Must be one of sentence or full. Defaults to sentence.
- **`llm_callable`** _(str, Callable):_ Either the name of the LiteLLM model string to use, or a callable that accepts a string prompt and returns a string response: "yes" or "no". Defaults to `gpt-3.5-turbo`.
- **`top_k`** _(int):_ The number of chunks to return from the query function. Defaults to 3.
- **`on_fail`** *(str, Callable):* The policy to enact when a validator fails. If `str`, must be one of `reask`, `fix`, `filter`, `refrain`, `noop`, `exception` or `fix_reask`. Otherwise, must be a function that is called when the validator fails.

</ul>

<br>

**`__call__(self, value, metadata={}) → ValidationResult`**

<ul>

Validates the given `value` using the rules defined in this validator, relying on the `metadata` provided to customize the validation process. This method is automatically invoked by `guard.parse(...)`, ensuring the validation logic is applied to the input data.

Note:

1. This method should not be called directly by the user. Instead, invoke `guard.parse(...)` where this method will be called internally for each associated Validator.
2. When invoking `guard.parse(...)`, ensure to pass the appropriate `metadata` dictionary that includes keys and values required by this validator. If `guard` is associated with multiple validators, combine all necessary metadata into a single dictionary.

**Parameters:**

- **`value`** *(Any):* The input value to validate.
- **`metadata`** *(dict):* A dictionary containing metadata required for validation. Keys and values must match the expectations of this validator.
    
    | Key | Type | Description | Default |
    | --- | --- | --- | --- |
    | `query_function` | _Optional[Callable]_ | A callable that takes a string and returns a list of (chunk, score) tuples. In order to use this validator, you must provide either a `query_function` or `sources` with an `embed_function` in the metadata. The query_function should take a string as input and return a list of (chunk, score) tuples. The chunk is a string and the score is a float representing the cosine distance between the chunk and the input string. The list should be sorted in ascending order by score. | None |
    | `sources` | *Optional[List[str]]* | The source text. In order to use this validator, you must provide either a `query_function` or `sources` with an `embed_function` in the metadata. | None |
    | `embed_function` | *Optional[Callable]* | A callable that creates embeddings for the sources. Must accept a list of strings and return an np.array of floats. | sentence-transformer's `paraphrase-MiniLM-L6-v2` |
    | `pass_on_invalid` | *Optional[bool]* | Whether to pass the validator if LLM returns an invalid response | False |
    | `chunk_strategy` | *Optional[str]* | The strategy to use for chunking the input and sources. Must be one of `sentence`, `word`, `char` or `token`. | `sentence` |
    | `chunk_size` | *Optional[int]* | The number of sentences, words, characters or tokens in each chunk. Depends on the `chunk_strategy` used | 5 |
    | `chunk_overlap` | *Optional[int]* | The number of sentences, words, characters or tokens to overlap between chunks. Depends on the `chunk_strategy` used | 2 |

</ul>
