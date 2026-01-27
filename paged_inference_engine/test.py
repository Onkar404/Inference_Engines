
from paged_engine import PagedGPTInferenceEngine

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

save_dir = "./gpt2_saved"

tokenizer = GPT2Tokenizer.from_pretrained(save_dir)
model = GPT2LMHeadModel.from_pretrained(
    save_dir,
    dtype=torch.float16,   # match inference dtype
    low_cpu_mem_usage=True
).to(device)

model.eval()
for p in model.parameters():
    p.requires_grad_(False)


BATCH_SIZE = 1
BLOCK_SIZE = model.config.n_ctx    
DTYPE = torch.float16                
engine = PagedGPTInferenceEngine(
    model,
    512,4
)



prompt = "lion is the king of"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]

# Reference
# with torch.no_grad():
#     ref_logits = model(input_ids).logits[:, -1, :]
#     ref_token = torch.argmax(ref_logits, dim=-1)

# # Engine
# eng_logits = engine.prefill(input_ids)
# eng_token = torch.argmax(eng_logits, dim=-1)

# print("Reference token:", tokenizer.decode(ref_token))
# print("Engine token:   ", tokenizer.decode(eng_token))

# assert ref_token.item() == eng_token.item()
# print("Prefill parity PASSED ✅")

# engine.prefill(input_ids)
# generated=engine.decode()
# print(tokenizer.decode(generated))

# reference
# with torch.no_grad():
#     ref = model.generate(input_ids, max_new_tokens=20, do_sample=False)

# # engine
engine.prefill(input_ids)
out = engine.decode(max_new_tokens=20)

# print("ref:", tokenizer.decode(ref[:][-1]))
print("eng:", tokenizer.decode(out))


# print(tokenizer.decode(engine.generate(input_ids,max_new_tokens=50)))
# ref = model.generate(
#     input_ids,
#     max_new_tokens=20,
#     do_sample=False,
#     pad_token_id=tokenizer.eos_token_id
# )

# eng = engine.generate(
#     input_ids,
#     max_new_tokens=20
# )

# print("REF:", tokenizer.decode(ref[0]))
# print("ENG:", tokenizer.decode(input_ids[0].tolist() + eng))