import logging
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter, Retry
from transformers import AutoTokenizer

from guidance.llms import LLM, LLMSession, SyncSession

logger = logging.getLogger(__name__)


def _get_session() -> requests.Session:
    """Get a session with retries."""
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s


def _remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def _output_parser(response, prompt):
    output = {"choices": []}
    if isinstance(response["text"], str):
        response["text"] = [response["text"]]
    for text in response["text"]:
        assert text.startswith(prompt)
        output["choices"].append(
            {"text": _remove_prefix(text, prompt), "logprobs": None}
        )
    return output


class VLLMWrapper(LLM):
    llm_name: str = "vllm"

    def __init__(
        self,
        server_uri,
        model,
        caching=True,
        temperature=0.0,
        **kwargs,
    ):
        super().__init__()
        self.server_uri = server_uri
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self.caching = caching
        self.temperature = temperature

    def session(self, asynchronous=False):
        if asynchronous:
            return VLLMSession(self)
        else:
            return SyncSession(VLLMSession(self))


class VLLMSession(LLMSession):
    async def __call__(
        self,
        prompt,
        stop=None,
        stop_regex=None,
        temperature=None,
        n=1,
        max_tokens=1000,
        logprobs=None,
        top_p=1.0,
        echo=False,
        logit_bias=None,
        token_healing=None,
        pattern=None,
        stream=False,
        cache_seed=0,
        caching=None,
        **generate_kwargs,
    ):
        self._log_on_unused_argument(
            token_healing=token_healing,
            logit_bias=logit_bias,
            stop=stop,
            stop_regex=stop_regex,
            logprobs=logprobs,
            pattern=pattern,
        )

        if temperature is None:
            temperature = self.llm.temperature

        kwargs = dict(
            prompt=prompt,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        args = locals().copy()
        cache_params = self._cache_params(args)
        llm_cache = self.llm.cache
        key = llm_cache.create_key(self.llm.llm_name, **cache_params)

        in_cache = key in llm_cache
        not_caching = (caching is not True and not self.llm.caching) or caching is False
        if not in_cache or not_caching:
            with _get_session() as session:
                response = session.post(
                    url=urljoin(self.llm.server_uri, "generate"),
                    json=kwargs,
                )
                response.raise_for_status()
                llm_cache[key] = _output_parser(response.json(), prompt)
        return llm_cache[key]

    @staticmethod
    def _log_on_unused_argument(**kwargs):
        unused_args = [f"{key}={val}" for key, val in kwargs.items() if val is not None]
        unused_args_as_str = ", ".join(unused_args)
        if unused_args:
            logger.warning(f"Arguments {unused_args_as_str} are not used by VLLM.")
