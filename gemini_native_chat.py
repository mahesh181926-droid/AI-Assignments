from google import genai

class GeminiChatCompletion:
    """
    Mimics openai.ChatCompletion.create(), but uses Gemini Native SDK directly.
    """
    @staticmethod
    def create(model, messages, temperature=0.7, max_tokens=1024, api_key=None, **kwargs):
        """
        Args:
            model (str): Gemini model name (e.g., 'gemini-pro')
            message (list): List of dicts with 'role' and 'content' keys
            temperature (float): Sampling temperature
            max_tokens (int): Max tokens in response
            api_key (str): Google Generative AI API Key
            kwargs: other parameters(ignored)
        Returns:
            dict: Mimics OpenAI ChatCompletion response
        """

        #Convert OpenAI-style messages to Gemini format
        gemini_messages = []
        for msg in messages:
            if msg['role'] == 'user':
                gemini_messages.append({"role": "user", "parts": [{"text": msg['content']}]})
            elif msg['role'] == 'system':
                #Gemini may not use 'system' role, but can prepend as context
                gemini_messages.append({"role": "user", "parts": [{"text": msg['content']}]})
            # 'assistant' role is not sent as input

        #Initialize the Client with your API key
        client = genai.Client(api_key=api_key) 

        #Call Gemini API
        response = client.models.generate_content(model=model,contents=gemini_messages)

        #Extract response text
        content = response.candidates[0].content.parts[0].text 
        
        #Format response to mimic OpenAI
        return {
            'id': 'chatcmpl=gemini-native',
            'object': 'chat.completion',
            'created': 0,
            'model': model,
            'choices': [
                {
                    'index': 0,
                    'messages': {
                        'role': 'assistant',
                        'content': content
                    },
                    'finish_reason': 'stop'
                }
            ],
            'usage': {
                'prompt_tokens': None,
                'completion_tokens': None,
                'total_tokens': None
            }
        }

response = GeminiChatCompletion.create(
    model="gemini-2.5-flash",
    messages= [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who won the world cup in 2022?"}
    ],
    api_key="AIzaSyABwoDPRHAo0he5CazlsZ_WasGLrj7-p3c"
)

print(response['choices'][0]['messages']['content'])