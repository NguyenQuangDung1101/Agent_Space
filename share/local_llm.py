import os
import base64
import requests

from dotenv import load_dotenv


class Copilot:

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        load_dotenv()

        self.provider = provider or os.getenv("LLM_PROVIDER", "OLLAMA")
        self.provider_api_key = os.getenv("PROVIDER_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "gemma4:31b-cloud")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.timeout = timeout or 300

    def infer_ollama(
        self,
        user_prompt,
        system_prompt=None,
        image_path=None,
        think=None,
    ):
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        user_message = {
            "role": "user",
            "content": user_prompt,
        }

        if image_path:
            with open(image_path, "rb") as image_file:
                user_message["images"] = [
                    base64.b64encode(
                        image_file.read()
                    ).decode("utf-8")
                ]

        messages.append(user_message)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        if think is not None:
            payload["think"] = think

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()["message"]["content"]


    # Main Infer Funct ========================================================================================
    def infer(
        self,
        user_prompt,
        system_prompt=None,
        image_path=None,
        think=True,
    ):
        if self.provider.upper() == "OLLAMA":
            return self.infer_ollama(
                user_prompt,
                system_prompt,
                image_path,
                think,
            )

        raise ValueError(
            f"Unsupported provider: {self.provider}"
        )
    # =========================================================================================================
    
if __name__ == "__main__":
    copilot = Copilot()
    user_prompt = "Write a short poem about the ocean."
    system_prompt = "You are a creative poet."
    response = copilot.infer(user_prompt, system_prompt)
    print(response)