# import unsloth
# # Modules for fine-tuning
# from unsloth import FastLanguageModel
# import torch # Import PyTorch
# from trl import SFTTrainer # Trainer for supervised fine-tuning (SFT)
# from unsloth import is_bfloat16_supported # Checks if the hardware supports bfloat16 precision
# # Hugging Face modules
# from huggingface_hub import login # Lets you login to API
# from transformers import TrainingArguments # Defines training hyperparameters
# from datasets import load_dataset # Lets you load fine-tuning datasets
# # Import weights and biases
# import wandb
# # Import kaggle secrets
# from kaggle_secrets import UserSecretsClient

# # Initialize Hugging Face & WnB tokens
# user_secrets = UserSecretsClient() # from kaggle_secrets import UserSecretsClient
# hugging_face_token = user_secrets.get_secret("HF_TOKEN")
# wnb_token = user_secrets.get_secret("wnb")

# # Login to Hugging Face
# login(hugging_face_token) # from huggingface_hub import login

# # Login to WnB
# wandb.login(key=wnb_token) # import wandb
# run = wandb.init(
#     project='Fine-tune-DeepSeek-R1-Distill-Llama-8B on Customer Chatbot', 
#     job_type="training", 
#     anonymous="allow"
# )

# # Set parameters
# max_seq_length = 2048 # Define the maximum sequence length a model can handle (i.e. how many tokens can be processed at once)
# dtype = None # Set to default 
# load_in_4bit = True # Enables 4 bit quantization — a memory saving optimization 

# # Load the DeepSeek R1 model and tokenizer using unsloth — imported using: from unsloth import FastLanguageModel
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",  # Load the pre-trained DeepSeek R1 model (8B parameter version)
#     max_seq_length=max_seq_length, # Ensure the model can process up to 2048 tokens at once
#     dtype=dtype, # Use the default data type (e.g., FP16 or BF16 depending on hardware support)
#     load_in_4bit=load_in_4bit, # Load the model in 4-bit quantization to save memory
#     token=hugging_face_token, # Use hugging face token
# )

# # Define a system prompt under prompt_style 
# prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
# Write a response that appropriately completes the request. 
# Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

# ### Instruction:
# You are a quality assurance expert with in-depth knowledge of the University of Nottingham Quality Manual, its academic policies, and quality assurance framework. 
# Please answer the following question related to university quality assurance or academic standards.

# ### Question:
# {}

# ### Response:
# <think>{}"""

# # Creating a test medical question for inference
# # question = """A 61-year-old woman with a long history of involuntary urine loss during activities like coughing or 
# #               sneezing but no leakage at night undergoes a gynecological exam and Q-tip test. Based on these findings, 
# #               what would cystometry most likely reveal about her residual volume and detrusor contractions?"""
# question = """What is the university of nottingham quality manual for?"""

# # Enable optimized inference mode for Unsloth models (improves speed and efficiency)
# FastLanguageModel.for_inference(model)  # Unsloth has 2x faster inference!

# # Format the question using the structured prompt (`prompt_style`) and tokenize it
# inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")  # Convert input to PyTorch tensor & move to GPU

# # Generate a response using the model
# outputs = model.generate(
#     input_ids=inputs.input_ids, # Tokenized input question
#     attention_mask=inputs.attention_mask, # Attention mask to handle padding
#     max_new_tokens=1200, # Limit response length to 1200 tokens (to prevent excessive output)
#     use_cache=True, # Enable caching for faster inference
# )

# # Decode the generated output tokens into human-readable text
# response = tokenizer.batch_decode(outputs)

# # Extract and print only the relevant response part (after "### Response:")
# print(response[0].split("### Response:")[1])  

# # Updated training prompt style to add </think> tag 
# train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
# Write a response that appropriately completes the request. 
# Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

# ### Instruction:
# You are a quality assurance expert with in-depth knowledge of the University of Nottingham Quality Manual, its academic policies, and quality assurance framework. 
# Please answer the following question related to university quality assurance or academic standards.

# ### Question:
# {}

# ### Response:
# <think>
# {}
# </think>
# {}"""

# dataset = load_dataset("json", data_files="/kaggle/input/d/nigelseah/quality-manual-dataset/structured_quality_manual.json", field=None, split="train")
# print(dataset[0])

# # We need to format the dataset to fit our prompt training style 
# EOS_TOKEN = tokenizer.eos_token  # Define EOS_TOKEN which the model when to stop generating text during training
# EOS_TOKEN

# # Apply LoRA (Low-Rank Adaptation) fine-tuning to the model 
# model_lora = FastLanguageModel.get_peft_model(
#     model,
#     r=16,  # LoRA rank: Determines the size of the trainable adapters (higher = more parameters, lower = more efficiency)
#     target_modules=[  # List of transformer layers where LoRA adapters will be applied
#         "q_proj",   # Query projection in the self-attention mechanism
#         "k_proj",   # Key projection in the self-attention mechanism
#         "v_proj",   # Value projection in the self-attention mechanism
#         "o_proj",   # Output projection from the attention layer
#         "gate_proj",  # Used in feed-forward layers (MLP)
#         "up_proj",    # Part of the transformer’s feed-forward network (FFN)
#         "down_proj",  # Another part of the transformer’s FFN
#     ],
#     lora_alpha=16,  # Scaling factor for LoRA updates (higher values allow more influence from LoRA layers)
#     lora_dropout=0,  # Dropout rate for LoRA layers (0 means no dropout, full retention of information)
#     bias="none",  # Specifies whether LoRA layers should learn bias terms (setting to "none" saves memory)
#     use_gradient_checkpointing="unsloth",  # Saves memory by recomputing activations instead of storing them (recommended for long-context fine-tuning)
#     random_state=3407,  # Sets a seed for reproducibility, ensuring the same fine-tuning behavior across runs
#     use_rslora=False,  # Whether to use Rank-Stabilized LoRA (disabled here, meaning fixed-rank LoRA is used)
#     loftq_config=None,  # Low-bit Fine-Tuning Quantization (LoFTQ) is disabled in this configuration
# )

# def formatting_prompts_func(examples):
#     urls = examples["url"]
#     headers = examples["header"]
#     intros = examples["intro"]
#     contents = examples["content"]
    
#     texts = []  # Initialize list to store the formatted documents

#     for url, header, intro, content in zip(urls, headers, intros, contents):
#         # Create a single coherent document for each JSON entry
#         document = (
#             f"URL: {url}\n"
#             f"Header: {header}\n"
#             f"Intro: {intro}\n"
#             f"Content: {content}"
#         )
#         texts.append(document)
    
#     return texts  # Return the list of processed strings

# # Initialize the fine-tuning trainer — Imported using from trl import SFTTrainer
# trainer = SFTTrainer(
#     model=model_lora,  # The model to be fine-tuned
#     tokenizer=tokenizer,  # Tokenizer to process text inputs
#     # train_dataset=dataset_finetune,  # Dataset used for training
#     train_dataset=dataset,  # Dataset used for training
#     dataset_text_field="text",  # Specifies which field in the dataset contains training text
#     max_seq_length=max_seq_length,  # Defines the maximum sequence length for inputs
#     dataset_num_proc=2,  # Uses 2 CPU threads to speed up data preprocessing
#     formatting_func=formatting_prompts_func,
#     # Define training arguments
#     args=TrainingArguments(
#         per_device_train_batch_size=2,  # Number of examples processed per device (GPU) at a time
#         gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps before updating weights
#         num_train_epochs=1, # Full fine-tuning run
#         warmup_steps=5,  # Gradually increases learning rate for the first 5 steps
#         max_steps=60,  # Limits training to 60 steps (useful for debugging; increase for full fine-tuning)
#         learning_rate=2e-4,  # Learning rate for weight updates (tuned for LoRA fine-tuning)
#         fp16=not is_bfloat16_supported(),  # Use FP16 (if BF16 is not supported) to speed up training
#         bf16=is_bfloat16_supported(),  # Use BF16 if supported (better numerical stability on newer GPUs)
#         logging_steps=10,  # Logs training progress every 10 steps
#         optim="adamw_8bit",  # Uses memory-efficient AdamW optimizer in 8-bit mode
#         weight_decay=0.01,  # Regularization to prevent overfitting
#         lr_scheduler_type="linear",  # Uses a linear learning rate schedule
#         seed=3407,  # Sets a fixed seed for reproducibility
#         output_dir="outputs",  # Directory where fine-tuned model checkpoints will be saved
#     ),
# )

# # Start the fine-tuning process
# trainer_stats = trainer.train()

# # Save the fine-tuned model
# wandb.finish()

import os 
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "nigelseah/nottingham-qa-model"
# filenames = [
#     "adapter_config.json",
#     "adapter_model.safetensors",
#     "tokenizer.json",
#     "tokenizer_config.json",
#     "special_tokens_map.json",
# ]
# # for filename in filenames:
# #     hf_hub_download(
# #         repo_id=model_id,
# #         filename=filename,
# #         token=HUGGING_FACE_API_KEY,
# #     )
# #     print(f"Downloaded {filename} to model_files directory")

tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Define your prompt style with placeholders for the question and chain-of-thought.
prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a quality assurance expert with in-depth knowledge of the University of Nottingham Quality Manual, its academic policies, and quality assurance framework. 
Please answer the following question related to university quality assurance or academic standards.

### Question:
{}

### Response:
<think>{}"""

# Create a text-generation pipeline
text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=1000)

# Define your question and optionally the chain-of-thought (empty in this example)
question = "what are the university's regulations about missing an exam?"
chain_of_thought = ""  # You can add any initial chain-of-thought here if desired.

# Format the prompt by inserting the question and chain-of-thought into the template
formatted_prompt = prompt_style.format(question, chain_of_thought)

# Generate a response using the pipeline
output = text_gen_pipeline(formatted_prompt)
print(output)