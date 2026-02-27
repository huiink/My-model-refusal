import random
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi, get_token, login, snapshot_download
from tqdm import tqdm

# 確保在 Inference Mode 下執行，節省記憶體
torch.inference_mode()

# 1. 設定模型與參數
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
SAVE_PATH = "D:\Ai\deepseek-r1-1.5B_refusal"  # 修改後模型的儲存路徑
REPO_ID = "huiink/deepseek-r1-1.5B-abliterated" # 你的 HF 帳號/模型名稱

# 💡 強烈建議使用 bfloat16：這是大模型原生訓練的精度，能避免精度轉換帶來的效能退化與記憶體浪費
print(f"Loading model: {MODEL_ID} in bfloat16...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    dtype=torch.bfloat16, 
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# settings:
instructions = 64
# 設定提取層
layer_idx = int(len(model.model.layers) * 0.6) 
pos = -1

print("Instruction count: " + str(instructions))
print("Extraction Layer index: " + str(layer_idx))

# 2. 讀取資料與計算 Refusal Vector
with open("harmful.txt", "r", encoding="utf-8") as f:
    harmful = f.readlines()

with open("harmless.txt", "r", encoding="utf-8") as f:
    harmless = f.readlines()

harmful_instructions = random.sample(harmful, instructions)
harmless_instructions = random.sample(harmless, instructions)

harmful_toks = [
    tokenizer.apply_chat_template(conversation=[{"role": "user", "content": insn}],
        add_generation_prompt=True,
        return_tensors="pt") for insn in harmful_instructions]
harmless_toks = [
    tokenizer.apply_chat_template(conversation=[{"role": "user", "content": insn}],
        add_generation_prompt=True,
        return_tensors="pt") for insn in harmless_instructions]

max_its = instructions * 2
bar = tqdm(total=max_its, desc="Generating Activations")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def generate(toks):
    bar.update(n=1)
    
    # 檢查輸入型別並提取 input_ids
    if isinstance(toks, dict) or hasattr(toks, 'input_ids'):
        input_ids = toks['input_ids']
    else:
        input_ids = toks

    # 移動到 GPU/CPU
    input_ids = input_ids.to(model.device)
    
    # 手動建立 Attention Mask
    attention_mask = torch.ones(input_ids.shape, device=model.device, dtype=torch.long)
    
    # 執行生成
    return model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_hidden_states=True,
        pad_token_id=tokenizer.pad_token_id
    )

harmful_outputs = [generate(toks) for toks in harmful_toks]
harmless_outputs = [generate(toks) for toks in harmless_toks]

bar.close()

harmful_hidden = [output.hidden_states[0][layer_idx][:, pos, :] for output in harmful_outputs]
harmless_hidden = [output.hidden_states[0][layer_idx][:, pos, :] for output in harmless_outputs]

harmful_mean = torch.stack(harmful_hidden).mean(dim=0)
harmless_mean = torch.stack(harmless_hidden).mean(dim=0)

refusal_dir = harmful_mean - harmless_mean
refusal_dir = refusal_dir / refusal_dir.norm()

print("\nRefusal Vector Calculated.")
print(f"Vector norm: {refusal_dir.norm().item()}")

# ---------------------------------------------------------
# 3. 執行權重正交化 (Weight Orthogonalization)
# ---------------------------------------------------------

print(f"\nStarting Ablation (Orthogonalization)...")
print(f"Targeting layers from {layer_idx} to {len(model.model.layers)}")

def orthogonalize_weight(weight, direction):
    # 保存原始形狀（重要！防止變成 4D）
    orig_shape = weight.shape
    
    # 強制轉成 2D 處理
    if weight.ndim > 2:
        weight_2d = weight.view(orig_shape[0], -1)
    else:
        weight_2d = weight
    
    direction = direction.view(-1).to(weight_2d.device, dtype=weight_2d.dtype)
    
    v = direction.unsqueeze(1)
    P = torch.matmul(v, v.T)
    I = torch.eye(P.shape[0], device=weight_2d.device, dtype=weight_2d.dtype)
    Q = I - P
    
    with torch.no_grad():
        new_weight = torch.matmul(weight_2d, Q)
        weight.copy_(new_weight.view(orig_shape))  # 還原原始形狀

target_layers = model.model.layers[layer_idx:] 

for i, layer in enumerate(tqdm(target_layers, desc="Ablating Layers")):
    modules_to_modify = []
    
    # Attention Projections
    if hasattr(layer.self_attn, "q_proj"): modules_to_modify.append(layer.self_attn.q_proj)
    if hasattr(layer.self_attn, "k_proj"): modules_to_modify.append(layer.self_attn.k_proj)
    if hasattr(layer.self_attn, "v_proj"): modules_to_modify.append(layer.self_attn.v_proj)
    
    # # MLP Projections
    if hasattr(layer.mlp, "gate_proj"): modules_to_modify.append(layer.mlp.gate_proj)
    if hasattr(layer.mlp, "up_proj"): modules_to_modify.append(layer.mlp.up_proj)
    
    # 舊架構預防
    if hasattr(layer.self_attn, "query_key_value"): modules_to_modify.append(layer.self_attn.query_key_value)
    if hasattr(layer.mlp, "dense_h_to_4h"): modules_to_modify.append(layer.mlp.dense_h_to_4h)

    for module in modules_to_modify:
        orthogonalize_weight(module.weight, refusal_dir)

print("\nAblation Complete.")

# ---------------------------------------------------------
# 4. 儲存模型並完美還原原廠 Config
# ---------------------------------------------------------
print(f"\n[1/3] 下載原始 repo 的所有非權重檔案（config、modeling_*.py、tokenizer）...")

os.makedirs(SAVE_PATH, exist_ok=True)

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=SAVE_PATH,
    local_dir_use_symlinks=False,
    ignore_patterns=[
        "*.safetensors", "*.bin", "*.pt", "*.msgpack", 
        "optimizer*", "rng_state_*", "trainer_state.json", "*.h5"
    ]
)

print(f"[2/3] 儲存修改後的權重 (safetensors)...")
model.save_pretrained(
    SAVE_PATH, 
    safe_serialization=True,      # 強烈建議用 safetensors（更安全、更快）
    max_shard_size="4GB"        # 1.5B 模型其實不用分片，可以註解掉更乾淨
)

print(f"[3/3] 重新儲存 tokenizer（確保 chat_template 正確）...")
tokenizer.save_pretrained(SAVE_PATH)

print(f"\n 模型已完美儲存到 {SAVE_PATH}")
print("   - 權重：已 ablate（修改過）")
print("   - config / modeling_*.py / configuration_*.py：原始官方")
print("   - tokenizer（含 chat_template）：最新狀態")
# ---------------------------------------------------------
# 5. 上傳至 Hugging Face (Optional)
# ---------------------------------------------------------
upload_choice = input("\nDo you want to upload to Hugging Face now? (y/n): ")

if upload_choice.lower() == 'y':
    token = get_token()
    if token is None:
        print("Token not found. Initiating login...")
        login()
        token = get_token()

    if token:
        try:
            api = HfApi()
            print(f"Creating repo: {REPO_ID} ...")
            api.create_repo(repo_id=REPO_ID, exist_ok=True, repo_type="model")
            
            # 因為我們在第 4 步已經把資料夾整理得非常完美，所以直接上傳整個資料夾即可！
            print(f"Uploading the entire perfectly mixed folder...")
            api.upload_folder(
                folder_path=SAVE_PATH,
                repo_id=REPO_ID,
                repo_type="model",
                commit_message="Upload ablated weights with original configs via Representation Engineering"
            )
            
            print(f"\n🎉 Success! Perfect model available at https://huggingface.co/{REPO_ID}")
            
        except Exception as e:
            print(f"Upload failed: {e}")
else:
    print("Skipping upload. Your perfect local model is ready.")
