import base64
import requests
import google.generativeai as genai
from PIL import Image
import replicate
import dashscope
from http import HTTPStatus
import os

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
    
def gpt_4v(image_path, prompt, api, temperature, top_p):
    with open(image_path, "rb") as image:
        image_base64 = base64.b64encode(image.read()).decode("utf-8")
        
    headers = {
        "Authorization": f"Bearer {api}",
        # "Content-Type": "application/json"
    }
    payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1024,
                "temperature": temperature,
                "top_p": top_p
            }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload)
    response = response.json()
    print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']

def gpt_4o(image_path, prompt, api, temperature, top_p):
    with open(image_path, "rb") as image:
        image_base64 = base64.b64encode(image.read()).decode("utf-8")
        
    headers = {
        "Authorization": f"Bearer {api}",
        # "Content-Type": "application/json"
    }
    payload = {
                "model": "gpt-4o-2024-05-13",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1024,
                "temperature": temperature,
                "top_p": top_p
            }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload)
    response = response.json()
    print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']


def gemini(image_path, prompt, api, temperature, top_p):
    genai.configure(api_key=api)

    def upload_to_gemini(path, mime_type=None):
        """Uploads the given file to Gemini.

        See https://ai.google.dev/gemini-api/docs/prompting_with_media
        """
        file = genai.upload_file(path, mime_type=mime_type)
        print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file

    generation_config = {
    "temperature": temperature,
    "top_p": top_p,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
    model_name="gemini-pro-vision",
    generation_config=generation_config,
    )

    # TODO Make these files available on the local file system
    # You may need to update the file paths
    files = [
    upload_to_gemini(image_path, mime_type="image/jpeg"),
    ]

    chat_session = model.start_chat(
    history=[
        {
        "role": "user",
        "parts": [
            files[0]
        ],
        },
    ]
    )

    response = chat_session.send_message(prompt)
    print(response.text)

    
def llava_1_6_34b(image_path, prompt, api, temperature, top_p):
    os.environ['REPLICATE_API_TOKEN'] = api
    collected_output = ""
    output = replicate.run(
        "yorickvp/llava-v1.6-34b:510c16b590b7e32a1ccd905bd4df6a2873a760c7cb4fec6715c96353fa6e302d",
        input={
            "image": f"data:image/jpeg;base64,{encode_image(image_path)}",
            "top_p": top_p,
            "prompt": prompt,
            "max_tokens": 1024,
            "temperature": temperature
        }
    )

    for item in output:
        print(item, end="")
        collected_output += item
    
    return collected_output

def llava_1_6_13b(image_path, prompt, api, temperature, top_p):
    os.environ['REPLICATE_API_TOKEN'] = api
    collected_output = ""
    output = replicate.run(
        "yorickvp/llava-v1.6-vicuna-13b:0603dec596080fa084e26f0ae6d605fc5788ed2b1a0358cd25010619487eae63",
        input={
            "image": f"data:image/jpeg;base64,{encode_image(image_path)}",
            "top_p": top_p,
            "prompt": prompt,
            "max_tokens": 1024,
            "temperature": temperature
        }
    )

    for item in output:
        print(item, end="")
        collected_output += item
    
    return collected_output
        

def llava_1_6_7b(image_path, prompt, api, temperature, top_p):
    os.environ['REPLICATE_API_TOKEN'] = api
    collected_output = ""
    output = replicate.run(
        "yorickvp/llava-v1.6-mistral-7b:19be067b589d0c46689ffa7cc3ff321447a441986a7694c01225973c2eafc874",
        input={
            "image": f"data:image/jpeg;base64,{encode_image(image_path)}",
            "top_p": top_p,
            "prompt": prompt,
            "max_tokens": 1024,
            "temperature": temperature
        }
    )

    for item in output:
        print(item, end="")
        collected_output += item
    
    return collected_output
        

def qwen_vl_plus(image_path, prompt, api, temperature, top_p):
    dashscope.api_key = api
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"file://{image_path}"},
                {"text": prompt}
            ]
        }
    ]
    response = dashscope.MultiModalConversation.call(model='qwen-vl-plus',
                                                     messages=messages, temperature=temperature, top_p=top_p)
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    
    if response.status_code == HTTPStatus.OK:
        print(response.output.choices[0].message.content.text)
        return response.output.choices[0].message.content.text
    else:
        raise Exception(f"Request failed with code {response.status_code}: {response.message}")
    
def qwen_vl_max(image_path, prompt, api, temperature, top_p):
    dashscope.api_key = api
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"file://{image_path}"},
                {"text": prompt}
            ]
        }
    ]
    response = dashscope.MultiModalConversation.call(model='qwen-vl-max',
                                                     messages=messages, temperature=temperature, top_p=top_p)
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    
    if response.status_code == HTTPStatus.OK:
        print(response.output.choices[0].message.content.text)
        return response.output.choices[0].message.content.text
    else:
        raise Exception(f"Request failed with code {response.status_code}: {response.message}")


if __name__ == "__main__":
    qwen_vl_plus("/media/ssd/cdp/LMM-as-a-Judge/MLLM-Judge-Code/Dataset/image/0.jpg", "What is this image about", "sk-af159c39ff7b45d4adff9be1847b82a1", 0.9, 0.9)