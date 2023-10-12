from langchain.llms import Ollama
from opencc import OpenCC


def convert_s2tw(str):
    cc = OpenCC("s2twp")  # convert from Simplified Chinese to Traditional Chinese
    converted = cc.convert(str)
    return converted


def convert_tw2s(str):
    cc = OpenCC("tw2sp")  # convert from Traditional Chinese to Simplified Chinese
    converted = cc.convert(str)
    return converted


def Taiwan_llama(query):
    Taiwan_llama_query = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {query} ASSISTANT:"
    ollama = Ollama(base_url="http://localhost:11434", model="Taiwan-LLaMa:13b-v1")
    result = ollama(Taiwan_llama_query)
    print("Taiwan_llama:")
    print(result)


def llama2_chinese(query):
    query = convert_tw2s(query)
    ollama = Ollama(
        base_url="http://localhost:11434", model="llama2-chinese:13b-chat-fp16"
    )
    result = convert_s2tw(ollama(query))
    print("llama2_chinese:")
    print(result)


def mistral(query):
    mistral_query = "用中文寫1個 " + query
    mistral_query = convert_tw2s(mistral_query)
    ollama = Ollama(base_url="http://localhost:11434", model="mistral")
    result = convert_s2tw(ollama(mistral_query))
    print("mistral:")
    print(result)


if __name__ == "__main__":
    query = "連身洋裝的銷售文案"
    # Taiwan_llama(query)
    # llama2_chinese(query)
    # mistral(query)

    # https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/llms/ollama.py
