import os
from typing import List, Literal, Dict, Any

from dotenv import load_dotenv

load_dotenv()

Role = Literal["system", "user", "assistant"]
Message = Dict[str, Any]


class DeepSeekClient:
    """
    Thin wrapper for an LLM.

    For now:
      - If DEEPSEEK_API_KEY is missing, always use a stub.
      - If DEEPSEEK_API_KEY exists, we still default to stub,
        but you can set use_real=True once you wire an API.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-V3.2-Exp",
        use_real: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.use_real = use_real if use_real is not None else bool(self.api_key)

    def complete(
        self,
        messages: List[Message],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        """
        messages: list of {"role": "system" | "user" | "assistant", "content": str}
        Returns: plain text completion
        """
        if not self.api_key or not self.use_real:
            last_user = next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                "",
            )
            return f"[LLM stub] I received your request:\n\n{last_user}"

        return self._call_real_api(messages, max_tokens=max_tokens, temperature=temperature)

    def _call_real_api(
        self,
        messages: List[Message],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        """
        Placeholder for a real DeepSeek or HF call.
        """
        raise NotImplementedError("Real DeepSeek API call is not implemented yet.")

