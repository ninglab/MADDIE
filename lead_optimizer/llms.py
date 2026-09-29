import os


class HFLocalLLM:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        device_map: str | None = "auto",
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device_map = device_map
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
        )

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        if self.tokenizer is None or self.model is None:
            self.load()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
        else:
            # Override sampling values inherited from some model generation configs.
            generate_kwargs["temperature"] = 1.0
            generate_kwargs["top_p"] = 1.0
        if self.tokenizer.eos_token_id is not None:
            generate_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

        output_ids = self.model.generate(**inputs, **generate_kwargs)
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


class OpenAICompatibleLLM:
    """Chat-completions client for vLLM or another OpenAI-compatible server."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        model_kwargs: dict | None = None,
        client=None,
    ):
        self.model = model or os.getenv("LLM_MODELNAME") or "gpt-4.1-mini"
        resolved_base_url = (
            base_url
            or os.getenv("LLM_BASEURL")
            or os.getenv("OPENAI_BASE_URL")
        )
        self.base_url = resolved_base_url.rstrip("/") if resolved_base_url else None
        resolved_api_key = (
            api_key
            or os.getenv("LLM_APIKEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.api_key = resolved_api_key or ("EMPTY" if self.base_url else None)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model_kwargs = model_kwargs or {}
        self.client = client
        self.responses = []
        # Preserve attribute names used by the MADDIE OpenAILLM wrapper.
        self.url = self.base_url
        self.key = self.api_key
        self.kwargs = self.model_kwargs

    def load(self) -> None:
        if self.client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAICompatibleLLM requires the 'openai' package. "
                "Install it in the lead optimizer environment with: pip install openai"
            ) from exc
        client_kwargs = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = OpenAI(**client_kwargs)

    def get_single_response(self, user: str, system: str | None = None):
        self.load()
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        request_kwargs.update(self.model_kwargs)
        response = self.client.chat.completions.create(**request_kwargs)
        self.responses.append(response)
        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI-compatible server returned an empty response.")
        return content.strip(), response

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        content, _response = self.get_single_response(
            user=user_prompt,
            system=system_prompt,
        )
        return content


OpenAILLM = OpenAICompatibleLLM
