import openai

# 设置API密钥
openai.api_key = "your-key"

jsonl_file_path = './MELD/MELD.jsonl'

response = openai.File.create(
  file=open(jsonl_file_path, "rb"),
  purpose='fine-tune'
)

file_id = response['id']
print(f"File uploaded successfully. File ID: {file_id}")