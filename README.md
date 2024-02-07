## Details

| Developed by | Guardrails AI |
| --- | --- |
| Date of development | Feb 15, 2024 |
| Validator type | RAG |
| Blog | https://www.guardrailsai.com/blog/reduce-ai-hallucinations-provenance-guardrails |
| License | Apache 2 |
| Input/Output | Output |

## Description

This validator uses an LLM callable to evaluate the generated text against the provided contexts (LLM-ception). In order to use this validator, you must provide either:

1. a `query_function` in the metadata. That function should take a string as input (the LLM-generated text) and return a list of relevant chunks. The list should be sorted in ascending order by the distance between the chunk and the LLM-generated text.
2. `sources` with an `embed_function` in the metadata. The embed_function should take a string or a list of strings as input and return a np array of floats. The vector should be normalized to unit length.

## Example Usage Guide

### Installation

```bash
$ gudardrails hub install ProvenanceV1
```

### Initialization

```python
guard = Guard.from_string(validators=[
    ProvenanceV1(llm_callable="gpt-3.5-turbo", ...)
])
```

### Invocation

```python
def embed_function(text: Union[str, List[str]]) -> np.ndarray:
    return np.array([[0.1, 0.2, 0.3]])

guard.parse(
    llm_output=...,
    metadata={"query_function": query_function}
)
```

## Intended use

- Primary intended uses: For RAG applications, checking if a text is hallucinated by establishing a source (i.e. provenance) for any LLM generated text.
- Out-of-scope use cases: General question answering without RAG or text grounding.

## Expected deployment metrics

|  | CPU | GPU |
| --- | --- | --- |
| Latency | 1.5 seconds | - |
| Memory | n/a | - |
| Cost | token cost for LLM invocation | - |
| Expected quality | 80% | - |

## Resources required

- Dependencies: Foundation model library, Embedding library
- Foundation model access keys: Yes (depending on which model is used)
- Compute: N/A

## Validator Performance

### Evaluation Dataset

https://huggingface.co/datasets/miracl/hagrid

### Model Performance Measures

| Accuracy | 80% |
| --- | --- |
| F1 Score | 0.9 |

### Decision thresholds

0.5
