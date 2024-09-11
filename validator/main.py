import itertools
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import nltk
import numpy as np
from guardrails.utils.docs_utils import get_chunks_from_text
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_random_exponential
from sentence_transformers import SentenceTransformer


PROVENANCE_LLM_PROMPT = """You will be given a context and a claim. Your task is to determine whether the claim is supported by the information provided in the context or not.

First, carefully read the following context:

<context>
{context}
</context>

Now, consider this claim:

<claim>
{claim}
</claim>

To determine if the claim is supported by the context, follow these steps:

1. Analyze the claim and break it down into its key components.
2. Search for evidence in the context that either supports or contradicts these components.
3. Consider both explicit statements and information that can be reasonably inferred from the context.
4. Be aware of any potential contradictions or inconsistencies between the claim and the context.

Before making your final determination, provide your reasoning in detail. Explain how you arrived at your conclusion, citing specific evidence from the context where applicable.

After presenting your reasoning, state your final determination. Use "NO" if the claim is not supported by the context or if there are any ambiguities in pronoun references that make the claim's validity uncertain.

Remember to base your analysis solely on the information provided in the context. Do not introduce external knowledge or make assumptions beyond what can be reasonably inferred from the given information.

Present your complete response in the following format:
<analysis>
[Your detailed reasoning and analysis here]
</analysis>

<supported>
[YES] or [NO]
</supported>
"""


@register_validator(name="guardrails/provenance_llm", data_type="string")
class ProvenanceLLM(Validator):
    """Validates that the LLM-generated text is supported by the provided
    context.

    This validator uses an LLM callable to evaluate the generated text against the
    provided context (from vector similarity).

    **Key Properties**

    | Property                      | Description                         |
    | ----------------------------- | ----------------------------------- |
    | Name for `format` attribute   | `guardrails/provenance_llm`         |
    | Supported data types          | `string`                            |
    | Programmatic fix              | None                                |

    Args:
        validation_method (str, optional): The method to use for validation.
            Must be either "sentence" or "full". Defaults to "sentence".
        llm_callable (Union[str, Callable], optional): The LLM callable to use.
            This can be either the model name string for LiteLLM, or a callable
            that accepts a prompt and returns a response. Defaults to "gpt-3.5-turbo".
        top_k (int, optional): The number of chunks to consider for similarity
            comparison. Defaults to 3.
        on_fail (Optional[Callable], optional): A callable to execute when the
            validation fails. Defaults to None.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        validation_method: str = "sentence",
        llm_callable: Union[str, Callable] = "gpt-3.5-turbo",
        top_k: int = 3,
        on_fail: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            on_fail,
            validation_method=validation_method,
            llm_callable=llm_callable,
            top_k=top_k,
            **kwargs,
        )

        if validation_method not in ["sentence", "full"]:
            raise ValueError("validation_method must be 'sentence' or 'full'.")
        self._validation_method = validation_method

        if not isinstance(llm_callable, (str, Callable)):
            raise ValueError(
                "llm_callable must be either a string (model name for LiteLLM) "
                " or a callable that accepts a prompt and returns a response."
            )
        # Set the LLM callable
        self.set_callable(llm_callable)
        self._top_k = int(top_k)

    def set_callable(self, llm_callable: Union[str, Callable]) -> None:
        """Set the LLM callable.

        Args:
            llm_callable: Either the model name string for LiteLLM,
                or a callable that accepts a prompt and returns a response.
        """
        if isinstance(llm_callable, str):
            # Setup a LiteLLM callable
            def litellm_callable(prompt: str) -> str:
                # Get the LLM response
                messages = [{"content": prompt, "role": "user"}]

                try:
                    # We should allow users to pass kwargs to this somehow
                    val_response = completion(model=llm_callable, messages=messages)
                    # Get the response and strip and lower it
                    val_response = val_response.choices[0].message.content  # type: ignore
                    if val_response is None:
                        raise RuntimeError("Error getting response from the LLM.")
                    val_response = val_response.strip().lower()
                except Exception as e:
                    raise RuntimeError(
                        f"Error getting response from the LLM: {e}"
                    ) from e
                return val_response

            self._llm_callable = litellm_callable
        elif isinstance(llm_callable, Callable):
            self._llm_callable = llm_callable

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def call_llm(self, prompt: str) -> str:
        """Call `self._llm_callable` with the given prompt.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            response (str): String representing the LLM response.
        """
        return self._llm_callable(prompt)

    def evaluate_with_llm(self, text: str, query_function: Callable) -> str | None:
        """Validate that the LLM-generated text is supported by the provided
        contexts.

        Args:
            value (Any): The LLM-generated text.
            query_function (Callable): The query function.

        Returns:
            eval_response (str): The evaluation response from the LLM.
        """

        def extract_tagged_text(text, tag_name):
            start_tag = f"<{tag_name}>"
            end_tag = f"</{tag_name}>"
            start_index = text.find(start_tag)
            end_index = text.find(end_tag)
            if start_index != -1 and end_index != -1:
                return text[start_index + len(start_tag) : end_index].strip()
            return None

        def remove_non_alphabetic(text):
            return "".join(char for char in text if char.isalpha())

        def process_output(text, tag_name):
            extracted_text = extract_tagged_text(text, tag_name)
            if extracted_text is None:
                return None
            return remove_non_alphabetic(extracted_text).strip().lower()


        # Get the relevant chunks using the query function
        relevant_chunks = query_function(text=text, k=self._top_k)

        # Create the prompt to ask the LLM
        context = "\n".join(relevant_chunks)
        prompt = PROVENANCE_LLM_PROMPT.format(context=context, claim=text)

        # Get evaluation response
        eval_response = self.call_llm(prompt)
        return process_output(eval_response, "supported")

    def validate_each_sentence(
        self, value: Any, query_function: Callable, metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate each sentence in the response."""
        pass_on_invalid = metadata.get("pass_on_invalid", False)  # Default to False

        # Split the value into sentences using nltk sentence tokenizer.
        sentences = nltk.sent_tokenize(value)

        unsupported_sentences, supported_sentences = [], []
        for sentence in sentences:
            eval_response = self.evaluate_with_llm(sentence, query_function)
            if eval_response == "yes":
                supported_sentences.append(sentence)
            elif eval_response == "no":
                unsupported_sentences.append(sentence)
            else:
                if pass_on_invalid:
                    warn(
                        "The LLM returned an invalid response. Considering the sentence as supported..."
                    )
                    supported_sentences.append(sentence)
                else:
                    warn(
                        "The LLM returned an invalid response. Considering the sentence as unsupported..."
                    )
                    unsupported_sentences.append(sentence)

        if unsupported_sentences:
            unsupported_sentences = "- " + "\n- ".join(unsupported_sentences)
            return FailResult(
                metadata=metadata,
                error_message=(
                    f"None of the following sentences in your response "
                    "are supported by the provided context:"
                    f"\n{unsupported_sentences}"
                ),
                fix_value="\n".join(supported_sentences),
            )
        return PassResult(metadata=metadata)

    def validate_full_text(
        self, value: Any, query_function: Callable, metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate the entire LLM text."""
        pass_on_invalid = metadata.get("pass_on_invalid", False)  # Default to False

        # Self-evaluate LLM with entire text
        eval_response = self.evaluate_with_llm(value, query_function)
        if eval_response == "yes":
            return PassResult(metadata=metadata)
        if eval_response == "no":
            return FailResult(
                metadata=metadata,
                error_message=(
                    "The following text in your response is not "
                    "supported by the provided context:\n" + value
                ),
            )
        if pass_on_invalid:
            warn("The LLM returned an invalid response. Passing the validation...")
            return PassResult(metadata=metadata)
        return FailResult(
            metadata=metadata,
            error_message=(
                "The LLM returned an invalid response. Failing the validation..."
            ),
        )

    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        """Validation method for the `ProvenanceLLM` validator."""

        query_function = self.get_query_function(metadata)
        if self._validation_method == "sentence":
            return self.validate_each_sentence(value, query_function, metadata)
        return self.validate_full_text(value, query_function, metadata)

    def get_query_function(self, metadata: Dict[str, Any]) -> Callable:
        """Get the query function from metadata.

        If `query_function` is provided, it will be used. Otherwise, `sources` and
        `embed_function` will be used to create a default query function.
        """
        query_fn = metadata.get("query_function", None)
        sources = metadata.get("sources", None)

        # Check that query_fn or sources are provided
        if query_fn is not None:
            if sources is not None:
                warnings.warn(
                    "Both `query_function` and `sources` are provided in metadata. "
                    "`query_function` will be used."
                )
            return query_fn

        if sources is None:
            raise ValueError(
                "You must provide either `query_function` or `sources` in metadata."
            )

        # Check chunking strategy, size and overlap
        chunk_strategy = metadata.get("chunk_strategy", "full")
        if chunk_strategy not in ["full", "sentence", "word", "char", "token"]:
            raise ValueError(
                "`chunk_strategy` must be one of 'sentence', 'word', "
                "'char', or 'token'."
            )
        chunk_size = metadata.get("chunk_size", 5)
        chunk_overlap = metadata.get("chunk_overlap", 2)

        # Check embed model
        embed_function = metadata.get("embed_function", None)
        if embed_function is None:
            # Load model for embedding function
            MODEL = SentenceTransformer("paraphrase-MiniLM-L6-v2")

            # Create embed function
            def st_embed_function(sources: list[str]):
                return MODEL.encode(sources)

            embed_function = st_embed_function
        return partial(
            self.query_vector_collection,
            sources=metadata["sources"],
            embed_function=embed_function,
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @staticmethod
    def query_vector_collection(
        text: str,
        k: int,
        sources: List[str],
        embed_function: Callable,
        chunk_strategy: str = "sentence",
        chunk_size: int = 5,
        chunk_overlap: int = 2,
    ) -> List[Tuple[str, float]]:
        """Query a collection of vectors using given text, return the top k chunks."""
        chunks = [
            get_chunks_from_text(source, chunk_strategy, chunk_size, chunk_overlap)
            for source in sources
        ]
        chunks = list(itertools.chain.from_iterable(chunks))

        # Create embeddings
        source_embeddings = np.array(embed_function(chunks))
        query_embedding = embed_function(text)

        # Ensure source_embeddings is 2D
        if source_embeddings.ndim == 1:
            source_embeddings = source_embeddings.reshape(1, -1)

        # Ensure query_embedding is 1D
        query_embedding = query_embedding.squeeze()

        # Compute distances using cosine similarity
        # and return top k nearest chunks
        cos_sim = 1 - (
            np.dot(source_embeddings, query_embedding)
            / (
                np.linalg.norm(source_embeddings, axis=1)
                * np.linalg.norm(query_embedding)
            )
        )
        top_indices = np.argsort(cos_sim)[:k]
        top_chunks = [chunks[j] for j in top_indices]
        return top_chunks
