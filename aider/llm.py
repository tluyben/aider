import importlib
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

AIDER_SITE_URL = "https://aider.chat"
AIDER_APP_NAME = "Aider"

os.environ["OR_SITE_URL"] = AIDER_SITE_URL
os.environ["OR_APP_NAME"] = AIDER_APP_NAME

# `import litellm` takes 1.5 seconds, defer it!


class LazyLiteLLM:
    _lazy_module = None

    def __getattr__(self, name):
        self._load_litellm()
        return getattr(self._lazy_module, name)

    def _load_litellm(self):
        if self._lazy_module is not None:
            return

        self._lazy_module = importlib.import_module("litellm")

        self._lazy_module.suppress_debug_info = True
        self._lazy_module.set_verbose = False
        self._lazy_module.drop_params = True


litellm = LazyLiteLLM()

__all__ = [litellm]
def send_custom_claude_compatible_request(model, messages, temperature=0, stream=False, functions=None, extra_headers=None, max_tokens=None):
    """
    Make a request to the custom Claude-compatible API.
    """
    api_url = "https://api.custom-claude-compatible.com/v1/chat/completions"
    api_key = os.environ.get("CUSTOM_CLAUDE_COMPATIBLE_API_KEY")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if functions:
        payload["functions"] = functions

    response = requests.post(api_url, headers=headers, json=payload, stream=stream)
    response.raise_for_status()

    if stream:
        return (chunk.decode() for chunk in response.iter_content(chunk_size=2048))
    else:
        return response.json()
