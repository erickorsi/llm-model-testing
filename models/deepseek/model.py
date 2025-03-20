from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

model_name = "deepseek-ai/DeepSeek-V3-Base" # lightweight
#model_name = "deepseek-ai/DeepSeek-V3" # heavier but better for complex NLP
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
