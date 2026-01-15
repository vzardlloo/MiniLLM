from torch.utils.data import Dataset
import torch
import os
from datasets import load_dataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 预训练数据集-继承自torch.utils.data.Dataset。Dataset是PyTorch中用于创建自定义数据集的基类。
# 示例：
# 假设我们有一个句子："我爱学习"，经过tokenizer处理后得到：
# input_ids = [101, 2769, 4263, 1962, 102]  # 假设101是[CLS]，102是[SEP]
# 代码执行过程：
# # 原始序列: [101, 2769, 4263, 1962, 102]
# X = input_ids[:-1]  # 去掉最后一个标记 → [101, 2769, 4263, 1962]
# Y = input_ids[1:]   # 去掉第一个标记 → [2769, 4263, 1962, 102]
# 训练任务对应关系：
# 输入X: [101, 2769, 4263, 1962]  → 模型需要预测
# 目标Y: [2769, 4263, 1962, 102]  → 下一个标记
#
# 具体预测任务：
# - 给定[101] → 预测2769("我")
# - 给定[101, 2769] → 预测4263("爱")
# - 给定[101, 2769, 4263] → 预测1962("学习")
# - 给定[101, 2769, 4263, 1962] → 预测102([SEP])
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建输入文本
        encoding = self.tokenizer(
            str(sample['text']),  # 获取样本中的'text'字段并转换为字符串
            max_length=self.max_length,
            padding='max_length', # 如果序列长度小于max_length，则用填充标记填充到max_length
            truncation=True,  # 如果序列长度大于max_length，则截断到max_length
            return_tensors='pt' # 返回PyTorch张量格式的结果
        )
        # 从编码结果中获取input_ids（标记ID的序列），并使用squeeze()方法移除维度为1的维度。
        input_ids = encoding.input_ids.squeeze()
        # 创建一个损失掩码，其中非填充标记的位置为True，填充标记的位置为False。
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # 创建输入序列X，它是input_ids去掉最后一个标记后的张量。
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        # 创建目标序列Y，它是input_ids去掉第一个标记后的张量。
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        # 将损失掩码也去掉第一个位置，使其与Y的长度匹配，并转换为长整型张量。
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, cs):
        messages = cs.copy()
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self.generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置
        # # === 打印每个token的掩码情况 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y, m) in enumerate(zip(X, Y, loss_mask)):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([y])!r:16s} mask={m}")
        # # ================================
        return X, Y, loss_mask