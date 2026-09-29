from openai import Client
from typing import Union
from .base import LLM
import os

class OpenAILLM(LLM):
    def __init__(self, model_kwargs={}):
        self.url = os.environ.get("LLM_BASEURL", None)
        self.key = os.environ.get("LLM_APIKEY", None)
        self.model = os.environ.get("LLM_MODELNAME", "gpt-5.4")
        self.client = Client(base_url=self.url, api_key=self.key)
        self.kwargs = model_kwargs

    def get_single_response(self, user, system=None):
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.kwargs
        )
        return response.choices[0].message.content, response
