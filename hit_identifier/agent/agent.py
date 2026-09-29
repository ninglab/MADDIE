import logging
import time

runtime_logger = logging.getLogger("hit_identifier")


class OpenAIAgent():
    def __init__(self, key, model):
        self.timestamp = 0
        self.model = model
        self.key = key
        self.history = []
        self.responses = []

    def get_response(self, messages, retry=True):
        from openai import OpenAI
        client = OpenAI(api_key=self.key or None)
        self.timestamp = time.time()
        try:
            completion = client.chat.completions.create(
                messages=messages,
                model=self.model,
            )
            self.responses.append(completion)
            result = completion.choices[0].message.content
            self.history.append(messages)
            return result
        except Exception as e:
            if retry:
                runtime_logger.warning("OpenAI API error (%s); retrying", e)
                time.sleep(0.5)
                return self.get_response(messages, retry=False)
            else:
                raise e

    def count_tokens(self, text):
        return 0
