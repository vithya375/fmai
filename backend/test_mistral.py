import mistralai
import mistralai.client
print("Mistral initialized")
client = mistralai.client.Mistral(api_key="dummy")
print("has complete_async:", hasattr(client.chat, "complete_async"))
