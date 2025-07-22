from openai import OpenAI


client = OpenAI(api_key="<API_KEY>", base_url="https://api.siliconflow.cn/v1")

response1 = client.chat.completions.create(
    model='Qwen/Qwen3-8B',
    messages=[
        {'role': 'user',
         'content': "tell me a story in chinese within 200 words."}
    ],
    stream=True
)

response2 = client.embeddings.create(
    model='BAAI/bge-m3',
    dimensions=256,
    encoding_format='float',
    input=["Hello, world!", "How are you?"]
)


print(response2)

for chunk in response1:
    if not chunk.choices:
        continue
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)