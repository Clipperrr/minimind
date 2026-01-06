import argparse
import random
import warnings
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, Logger
from transformers import AutoModel
import re
from collections import Counter
warnings.filterwarnings('ignore')


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer, tokenizer, args):
    """整合所有奖励函数计算总奖励"""
    def reasoning_model_reward(rewards):
        # 1. 格式奖励（仅针对训练推理模型时使用）
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"

        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern:
                format_rewards.append(0.5)
            elif match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(-1.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        # 2. 标记奖励（防止严格奖励稀疏，仅针对训练推理模型时使用）
        def mark_num(text):
            reward = 0
            print(text)
            print(text.count("<think>"), text.count("</think>"), text.count("<answer>"), text.count("</answer>"))
            if text.count("<think>") == 1 and text.count("</think>") == 1:
                reward += 0.50
            elif text.count("<think>") > 1 or text.count("</think>") > 1:
                reward -= 1.0
            if text.count("<answer>") == 1 and text.count("</answer>") == 1:
                reward += 0.50
            elif text.count("<answer>") > 1 or text.count("</answer>") > 1:
                reward -= 1.0
            if reward > 0:
                reward += 0.5  # 额外的成对奖励
            else:
                reward -= 1.0
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    def ngram_penalty(responses, tokenizer, n=3, weight=-1.0):
        rewards = []

        for text in responses:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < n:
                rewards.append(0.0)
                continue

            ngrams = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
            counts = Counter(ngrams)

            repeated = sum(c - 1 for c in counts.values() if c > 1)
            rewards.append(weight * (repeated / len(ngrams)))

        return torch.tensor(rewards, device=args.device)

    rewards = ngram_penalty(responses, tokenizer, weight=-1.0)

    # 格式奖励
    rewards = reasoning_model_reward(rewards)

    # 使用reward model计算整个response的奖励
    with torch.no_grad():
        reward_model_scores = []
        for prompt, response in zip(prompts, responses):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            tmp_chat = messages + [{"role": "assistant", "content": response}]
            score = reward_model.get_score(reward_tokenizer, tmp_chat)

            scale = 3.0
            score = max(min(score, scale), -scale)

            # 当args.reasoning=1时，额外计算<answer>内容的奖励
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()
                # 对answer内容单独计算reward
                tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                answer_score = max(min(answer_score, scale), -scale)
                score = score * 0.4 + answer_score * 0.6
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards



def merge_lora(model, path):
    try:
        state_dict = torch.load(path, map_location=model.device)
    except Exception as e:
        Logger(f"无法加载LoRA权重文件: {path}")
        return model
    # 移除可能的module前缀，确保key与模型层级名称一致
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            key_A = f"{name}.lora.A.weight"
            key_B = f"{name}.lora.B.weight"
            if key_A in state_dict and key_B in state_dict:
                # 直接合并权重: W_new = W_old + B @ A
                module.weight.data += state_dict[key_B] @ state_dict[key_A]

    return model

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            if args.merge_lora:
                merge_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
            else:
                apply_lora(model)
                load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    print(f'MiniMind模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    return model.eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--merge_lora', default=False, type=bool, help="LoRA合并（None表示不使用）")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=800, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('👶: '), '')
    for prompt in prompt_iter:
        setup_seed(2026) # or setup_seed(random.randint(0, 2048))
        if input_mode == 0: print(f'👶: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        if args.weight == 'reason': templates["enable_thinking"] = True # 仅Reason模型使用
        inputs = tokenizer.apply_chat_template(**templates) if args.weight not in ['pretrain', 'pretrain_fine'] else (tokenizer.bos_token + prompt)
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)
        print('🤖️: ', end='')
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        print('\n\n')

        # reward_model = AutoModel.from_pretrained(
        #     '../internlm2-1_8b-reward', dtype=torch.float16, trust_remote_code=True
        # )
        # reward_model = reward_model.to(args.device).eval().requires_grad_(False)
        # reward_tokenizer = AutoTokenizer.from_pretrained('../internlm2-1_8b-reward', trust_remote_code=True)
        # rewards = calculate_rewards(prompts, [response], reward_model, reward_tokenizer, tokenizer, args)  # [B]

        # print(f"奖励分数: {rewards[0]:.4f}\n")
if __name__ == "__main__":
    main()