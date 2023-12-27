from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter, Retry
from transformers import AutoTokenizer

from guidance.llms import LLM, LLMSession, SyncSession


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
        pretrained_model_name_or_path,
        caching=True,
        temperature=0.0,
        **kwargs,
    ):
        super().__init__()
        self.server_uri = server_uri
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
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
        assert (
            token_healing is None or token_healing is False
        ), "Token healing is not supported for VLLM"
        assert (
            stop is None and stop_regex is None
        ), "Stop and stop_regex are not supported for VLLM"
        assert logit_bias is None, "Logit bias is not supported for VLLM"

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
