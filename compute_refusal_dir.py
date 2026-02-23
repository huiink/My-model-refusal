import random
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# 引入 Hugging Face Hub API (用於上傳)
from huggingface_hub import HfApi, get_token, login , hf_hub_download

from tqdm import tqdm

# 確保在 Inference Mode 下執行，節省記憶體
torch.inference_mode()

# 1. 設定模型與參數
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
SAVE_PATH = "./Deeepseek-r1_refusal"  # 修改後模型的儲存路徑
REPO_ID = "huiink/depseek-r1-1.5B-abliterated" # 修改這裡：你的 HF 帳號/模型名稱

print(f"Loading model: {MODEL_ID} in float32...")


# 這是因為量化後的權重無法直接修改並儲存為標準格式
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    dtype=torch.float32,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# settings:
instructions = 32
# 設定提取層 (原腳本設定)
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
    
    # 1. 檢查輸入型別並提取 input_ids
    # 如果 toks 是字典 (BatchEncoding)，取出 'input_ids'
    if isinstance(toks, dict) or hasattr(toks, 'input_ids'):
        input_ids = toks['input_ids']
    else:
        # 如果 toks 已經是 Tensor
        input_ids = toks

    # 2. 移動到 GPU/CPU
    input_ids = input_ids.to(model.device)
    
    # 3. 手動建立 Attention Mask
    # 形狀與 input_ids 相同，全為 1
    attention_mask = torch.ones(input_ids.shape, device=model.device, dtype=torch.long)
    
    # 4. 執行生成
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
    # W_new = W @ (I - v * v.T)
    
    # 修正：強制將方向向量轉為 1D [hidden_size]，避免形狀錯誤
    direction = direction.view(-1).to(weight.device, dtype=weight.dtype)
    
    # 計算投影矩陣 P = v * v.T
    # v shape: [hidden_size, 1]
    v = direction.unsqueeze(1)
    
    # P shape: [hidden_size, hidden_size]
    P = torch.matmul(v, v.T)
    
    # 計算 (I - P)
    I = torch.eye(P.shape[0], device=weight.device, dtype=weight.dtype)
    Q = I - P
    
    # 執行投影: W_new = W @ Q
    with torch.no_grad():
        weight.copy_(torch.matmul(weight, Q))
# 選擇要修改的層
# 策略：通常修改提取層及其之後的所有層
target_layers = model.model.layers[layer_idx:] 

for i, layer in enumerate(tqdm(target_layers, desc="Ablating Layers")):
    # 我們需要找出層中的 Linear 模組
    # 常見的目標是 "Input Projections" (Q, K, V, Up, Gate)
    # 這會移除模型 "讀取" 該特徵的能力
    
    modules_to_modify = []
    
    # Attention Projections
    if hasattr(layer.self_attn, "q_proj"): modules_to_modify.append(layer.self_attn.q_proj)
    if hasattr(layer.self_attn, "k_proj"): modules_to_modify.append(layer.self_attn.k_proj)
    if hasattr(layer.self_attn, "v_proj"): modules_to_modify.append(layer.self_attn.v_proj)
    
    # MLP Projections
    if hasattr(layer.mlp, "gate_proj"): modules_to_modify.append(layer.mlp.gate_proj)
    if hasattr(layer.mlp, "up_proj"): modules_to_modify.append(layer.mlp.up_proj)
    
    # 如果是 Falcon 舊架構 (只是預防萬一，Falcon3 應該是 Llama 架構)
    if hasattr(layer.self_attn, "query_key_value"): modules_to_modify.append(layer.self_attn.query_key_value)
    if hasattr(layer.mlp, "dense_h_to_4h"): modules_to_modify.append(layer.mlp.dense_h_to_4h)

    for module in modules_to_modify:
        orthogonalize_weight(module.weight, refusal_dir)

print("\nAblation Complete.")

# ---------------------------------------------------------
# 4. 儲存模型
# ---------------------------------------------------------
print(f"Saving ONLY model weights to {SAVE_PATH} ...")
# 只存模型，不要呼叫 tokenizer.save_pretrained()
model.save_pretrained(SAVE_PATH, max_shard_size="4GB") 
print("Weights saved locally.")

# ---------------------------------------------------------
# 5. 上傳至 Hugging Face (Optional)
# ---------------------------------------------------------

upload_choice = input("\nDo you want to upload to Hugging Face now? (y/n): ")

if upload_choice.lower() == 'y':
    # 檢查是否已登入
    token = get_token()
    if token is None:
        print("Token not found. Initiating login...")
        login() # 這會跳出輸入框讓你直接在程式中登入
        token = get_token() # 再次檢查登入是否成功

    if token: # 確保登入成功再繼續
        try:
            api = HfApi()
            print(f"Creating repo: {REPO_ID} ...")
            api.create_repo(repo_id=REPO_ID, exist_ok=True, repo_type="model")
            
            # ==========================================
            # 關鍵修改 1：只上傳消融後的「權重」，不傳本地 JSON
            # ==========================================
            print(f"Uploading ablated weights only...")
            api.upload_folder(
                folder_path=SAVE_PATH,
                repo_id=REPO_ID,
                repo_type="model",
                allow_patterns=["*.safetensors", "*.safetensors.index.json"], # 只放行權重檔
                commit_message="Upload ablated model weights via Representation Engineering"
            )
            
            # ==========================================
            # 關鍵修改 2：直接從原始基礎模型搬運「配置與分詞器」
            # ==========================================
            print(f"Copying original configs and tokenizer from {MODEL_ID}...")
            config_files = [
                "config.json", 
                "generation_config.json", 
                "tokenizer.json", 
                "tokenizer_config.json"
            ]
            
            for file_name in config_files:
                try:
                    # 步驟 A: 將原廠檔案快取到本地
                    cached_file = hf_hub_download(repo_id=MODEL_ID, filename=file_name)
                    # 步驟 B: 原封不動上傳到你的倉庫
                    api.upload_file(
                        path_or_fileobj=cached_file,
                        path_in_repo=file_name,
                        repo_id=REPO_ID,
                        repo_type="model",
                        commit_message=f"Restore original {file_name} from base model"
                    )
                    print(f"  -> Successfully copied {file_name}")
                except Exception as e:
                    # 若基礎模型本來就沒有某些檔案 (例如 merges.txt 不一定會有)，可以直接忽略
                    pass
            
            print(f"\nSuccess! Perfect model available at https://huggingface.co/{REPO_ID}")
            
        except Exception as e:
            print(f"Upload failed: {e}")
else:
    print("Skipping upload. You can upload manually later.")