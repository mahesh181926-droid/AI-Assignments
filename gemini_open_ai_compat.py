from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage

class ChatCompletion:
    """
    Mimics openai.ChatCompletion.create(), but routes to Gemini via LangChain.
    """
    @staticmethod
    def create(model, messages,temperature=0.7, max_tokens=1024, api_key=None, **kwargs):
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
        #Prepare LangChain Gemini Chat Model
        chat = ChatGoogleGenerativeAI(model=model,
                                      temperature=temperature,
                                      max_output_tokens=max_tokens,
                                      google_api_key=api_key)
        
        #Convert OpenAI-style messages to LangChain messages
        lc_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                lc_messages.append(SystemMessage(content=msg['content']))
            elif msg['role'] == 'user':
                lc_messages.append(HumanMessage(content=msg['content']))
            else:
                #Gemini may not support 'assistant' as input, skip or handle as needed
                pass

        #Get Gemini response
        response = chat.invoke(lc_messages)

        #Format response to mimic OpenAI
        return {
            'id': 'chatcmpl-gemini',
            'object': 'chat-completion',
            'created': 0,
            'model': model,
            'choices': [
                {
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': response.content
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