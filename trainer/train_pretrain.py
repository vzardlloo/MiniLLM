import os
import sys

__package__ = "trainer"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minillm import MiniLLMConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, \
     init_model, SkipBatchSampler

warnings.filterwarnings('ignore')



def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    # 交叉熵损失函数,自回归模型的目标是预测下一个 token 的概率分布，交叉熵损失是其标配损失函数,核心作用是量化模型预测的概率分布与真实标签分布的差异
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    #遍历数据加载器，每次获取一个批次的数据，批次包含输入序列X、目标序列Y和损失掩码loss_mask
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        # 将所有数据移动到指定设备（GPU / CPU）
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        # 动态学习率调整，采用余弦退火算法，根据当前训练步数和总训练步数，计算当前学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 更新参数组,将计算出的学习率赋值给优化器的参数组,确保优化器在训练过程中使用的学习率是动态调整的
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # 启用混合精度训练（FP16/FP32），平衡训练速度与数值稳定性
        with autocast_ctx:
            # 前向传播：返回包含 logits（模型预测）和 aux_loss（辅助损失）的结果
            res = model(X)
            # 计算loss, 通过交叉熵损失计算模型预测(logits)与真实Token(Y)的loss,loss格式和Y对齐
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            ## 计算logits_loss,忽略填充标记的损失，求和后除以有效标记数，得到平均损失
            logits_loss = (loss * loss_mask).sum() / loss_mask.sum()
            # 计算总loss, 包含logits_loss和aux_loss
            loss = logits_loss + res.aux_loss
            # 梯度累积缩放,总损失除以累积步数
            loss = loss / args.accumulation_steps
        # 由于是使用FP16进行训练，需要对损失进行缩放，避免梯度消失(梯度变为0，导致模型无法更新),注意这里进行缩放之后,计算出的梯度的结果也会被缩放,后续要记得反缩放
        # 然后进行反向传播,根据loss计算梯度
        scaler.scale(loss).backward()

        # 累积多个小批次的梯度，然后一次性更新参数
        if (step + 1) % args.accumulation_steps == 0:
            # 梯度反缩放,将之前缩放后的梯度恢复到原始值,确保在参数更新时使用的是正确的梯度
            scaler.unscale_(optimizer)
            # 对梯度进行裁剪,防止梯度爆炸(梯度变得非常大，导致模型参数更新过大，甚至无法收敛)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 混合精度下更新模型参数，在FP32精度下更新模型参数,将更新后的参数转换回FP16
            scaler.step(optimizer)
            # 更新缩放器的梯度缩放因子,根据当前批次的梯度动态调整缩放因子,以平衡训练速度与数值稳定性
            scaler.update()
            # 更新完了参数,清空梯度缓存,准备下一个批次的梯度计算
            optimizer.zero_grad(set_to_none=True)
        # 打印日志,记录当前训练进度,包括损失、学习率、训练时间等信息
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_logits_loss = logits_loss.item()
            current_aux_loss = res.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')

            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss,
                                 "learning_rate": current_lr, "epoch_time": eta_min})
        # 模型保存
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del X, Y, loss_mask, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniLLM Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniLLM-Pretrain", help="wandb项目名")
    args = parser.parse_args()

    # 1. 初始化分布式训练环境和随机种子
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 如果是分布式环境, 每个进程使用不同的随机种子，单机模式就是固定种子
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    #  2. 配置目录、模型参数、检查checkpoint(支持断点续训)
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniLLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

    # 3. 设置混合精度
    # FP32(Float32): 精度极高，计算结果准确，但是 显存占用大、计算速度慢
    # FP16(Float16): 精度较低，计算结果可能会有一定误差，但是 显存占用小、计算速度快
    # 混合精度训练 = FP16 计算 + FP32 兜底,在模型训练的「大部分计算环节」使用 FP16（提速、省显存），只在「对精度敏感的核心环节」保留 FP32（保证训练稳定、不丢模型效果）
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # 4. 国产开源swanlab代替wandb, 用于可视化训练过程
    # https://swanlab.cn
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniLLM-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # 5. 定义模型、数据、优化器
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 梯度缩放器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 6. 从checkpoint恢复状态
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # 7. DistributedDataParallel包装模型,支持并行训练
    if dist.is_initialized():
        # 忽略一些静态变量
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # 8. 开始训练
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
        else:  # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), 0, wandb)

    #  9. 清理分布进程
    if dist.is_initialized(): dist.destroy_process_group()