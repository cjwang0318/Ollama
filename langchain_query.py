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
    # ollama = Ollama(base_url="http://localhost:11434", model="Taiwan-LLaMa:7b-v2-chat")
    ollama = Ollama(
        base_url="http://192.168.50.26:11434", model="Taiwan-LLaMa:7b-v2-chat"
    )
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


def breeze(query):
    mistral_query = (
        "<s>You are a helpful AI assistant built by MediaTek Research. The user you are helping speaks Traditional Chinese and comes from Taiwan. [INST] 用中文寫1個 "
        + query
        + "[/INST]"
    )
    # mistral_query = convert_tw2s(mistral_query)
    ollama = Ollama(
        base_url="http://192.168.50.26:11434",
        model="ycchen/breeze-7b-instruct-v1_0",
        temperature=0.8,
        top_k=30,
        repeat_penalty=1.5,
    )
    result = convert_s2tw(ollama(mistral_query))
    # print("mistral:")
    # print(result)
    return result


def taide(query):
    system_meg = "你是一個來自台灣的AI助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。"
    taide_query = (
        f"""<|start_header_id|>system<|end_header_id|>
        {system_meg}<|eot_id|><|start_header_id|>user<|end_header_id|>
        {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    )
    # print(taide_query)
    # mistral_query = convert_tw2s(mistral_query)
    ollama = Ollama(
        base_url="http://192.168.50.26:11434",
        model="cwchang/llama3-taide-lx-8b-chat-alpha1",
        temperature=0.8,
        top_k=30,
        repeat_penalty=1.5,
    )
    result = convert_s2tw(ollama(taide_query))
    # print("taid:")
    # print(result)
    return result


def Taiwan_llama3(query):
    system_meg = "You are an AI assistant called Twllm, created by TAME (TAiwan Mixture of Expert) project."
    query = f"""<|start_header_id|>system<|end_header_id|>
        {system_meg}<|eot_id|><|start_header_id|>user<|end_header_id|>
        {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    # print(taide_query)
    # mistral_query = convert_tw2s(mistral_query)
    ollama = Ollama(
        base_url="http://192.168.50.26:11434",
        model="cwchang/llama-3-taiwan-70b-instruct",
        temperature=0.8,
        top_k=30,
        repeat_penalty=1.5,
    )
    result = convert_s2tw(ollama(query))
    # print("taid:")
    # print(result)
    return result


def do_query(model_type, query):
    if model_type == "S":
        result_str = breeze(query)
    elif model_type == "M":
        result_str = taide(query)
    elif model_type == "L":
        result_str = Taiwan_llama3(query)
    else:
        result_str = "Please select a Model"
    return result_str


if __name__ == "__main__":
    query = "西裝褲的標題，銷售對像是年輕人，有耐穿好穿搭的特色"
    # Taiwan_llama(query)
    # breeze(query)
    # llama2_chinese(query)
    # mistral(query)
    # taide(query)
    Taiwan_llama3(query)
    # https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/llms/ollama.py
