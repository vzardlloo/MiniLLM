from torch.utils.data import Dataset
import torch
import os
from datasets import load_dataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 预训练数据集-继承自torch.utils.data.Dataset。Dataset是PyTorch中用于创建自定义数据集的基类。
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

