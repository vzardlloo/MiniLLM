# MiniLLM
## 项目介绍
快速(2h)训练一个超级mini版(0.025B)的LLM

## 项目运行
### 个人环境配置分享
- CPU: Intel(R) Xeon(R) Platinum 8336C CPU @ 2.30GHz
- RAM: 468 GiB
- GPU: NVIDIA A30 24GB
- Linux: Linux-5.15.0-91-generic-x86_64-with-glibc2.35
- Ubuntu: 22.04.4 LTS
- CUDA: 12.4
- Python: 3.10.12

## 运行项目
1. 克隆项目仓库
   ```bash
   git clone https://github.com/vzardlloo/MiniLLM.git
   cd MiniLLM
   ```
2. 安装依赖
   ```bash
   pip3 install uv && uv venv && uv pip install -r requirements.txt
   ```
3. 数据集下载
    ```bash
   cd dataset
   wget https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/pretrain_hq.jsonl
   wget https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/sft_mini_512.jsonl
   cd ..
    ```
4. 训练模型(预训练)
   ```bash
   cd trainer/
   python3 train_pretrain.py
   ```
5. 训练模型(SFT)
   ```bash
   python3  train_full_sft.py
   ```
6. 模型评估测试
   ```bash
   cd .. && python3 eval_llm.py 
   ```
  


