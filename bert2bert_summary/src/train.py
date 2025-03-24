# -*- coding: utf-8 -*-
import os
import torch
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    EncoderDecoderModel,
    BertTokenizer,
    DataCollatorForSeq2Seq,
    get_scheduler
)
from datasets import load_dataset, DatasetDict

# 设备检测（优先使用 CUDA）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 使用设备: {DEVICE}")
if DEVICE == "cuda":
    print(f"⚡ CUDA设备: {torch.cuda.get_device_name(0)}")
    print(f"💾 显存容量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

# 路径配置
DATA_PATH = r"C:\Users\76472\Desktop\bert2bert_summary\bert2bert_summary\data\lcsts.json"
MODEL_SAVE_DIR = r"C:\Users\76472\Desktop\bert2bert_summary\bert2bert_summary\models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def init_model_tokenizer(model_name="bert-base-chinese"):
    """初始化模型与分词器"""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        model_name,
        model_name,
        tie_encoder_decoder=True
    )

    model.config.update({
        "decoder_start_token_id": tokenizer.cls_token_id,
        "eos_token_id": tokenizer.sep_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "max_length": 128
    })

    return model, tokenizer


def preprocess_data(file_path, tokenizer, max_input=256, max_target=64):
    """改进后的数据预处理与分集"""
    # 加载原始数据
    full_dataset = load_dataset("json", data_files=file_path,
                                encoding="utf-8-sig", split="train")
    full_dataset =  full_dataset.select(range(min(100000, len(full_dataset))))
    # 数据集清洗（示例：过滤空数据和过短文本）
    full_dataset = full_dataset.filter(
        lambda x: len(x["text"]) > 50 and len(x["summary"]) > 5
    )

    # 划分训练集（80%）、验证集（10%）、测试集（10%）
    split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
    train_val = split_dataset["train"]
    test_dataset = split_dataset["test"]

    # 进一步划分验证集
    train_val_split = train_val.train_test_split(test_size=0.125, seed=42)  # 0.125 * 0.8=0.1
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]

    # 数据预处理函数
    def process(examples):
        inputs = tokenizer(
            examples["text"],
            max_length=max_input,
            truncation=True,
            padding="max_length"
        )
        targets = tokenizer(
            examples["summary"],
            max_length=max_target,
            truncation=True,
            padding="max_length",
            add_special_tokens=True
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    # 创建DatasetDict
    return DatasetDict({
        "train": train_dataset.map(process, batched=True, remove_columns=["text", "summary"]),
        "validation": val_dataset.map(process, batched=True, remove_columns=["text", "summary"]),
        "test": test_dataset.map(process, batched=True, remove_columns=["text", "summary"])
    })


def train():
    # 训练参数
    BATCH_SIZE = 16 if DEVICE == "cuda" else 4
    NUM_EPOCHS = 5  # 增加训练轮次
    LEARNING_RATE = 3e-5

    # 初始化模型、tokenizer 和数据集
    model, tokenizer = init_model_tokenizer()
    datasets = preprocess_data(DATA_PATH, tokenizer)

    # 数据加载器
    collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100)

    train_dataloader = DataLoader(
        datasets["train"],
        batch_size=BATCH_SIZE,
        collate_fn=collator,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    val_dataloader = DataLoader(
        datasets["validation"],
        batch_size=BATCH_SIZE,
        collate_fn=collator,
        pin_memory=True
    )

    # 优化器 & 学习率调度
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=200,
        num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    model.to(DEVICE)

    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        model.train()
        train_progress = tqdm(train_dataloader, desc=f"🚀 训练轮次 {epoch + 1}/{NUM_EPOCHS}")

        total_train_loss = 0
        for step, batch in enumerate(train_progress):
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}

            with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                outputs = model(**batch)

            scaler.scale(outputs.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            total_train_loss += outputs.loss.item()
            avg_train_loss = total_train_loss / (step + 1)
            train_progress.set_postfix(loss=f"{outputs.loss.item():.4f}", avg_loss=f"{avg_train_loss:.4f}")

        # 验证阶段
        model.eval()
        total_val_loss = 0
        val_progress = tqdm(val_dataloader, desc="🔍 验证中...")

        with torch.no_grad():
            for batch in val_progress:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                total_val_loss += outputs.loss.item()
                val_progress.set_postfix(val_loss=f"{outputs.loss.item():.4f}")

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"\n📊 Epoch {epoch + 1} | 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(MODEL_SAVE_DIR)
            tokenizer.save_pretrained(MODEL_SAVE_DIR)
            print(f"💾 发现更优模型，已保存至 {MODEL_SAVE_DIR}")

    print(f"\n🏆 训练完成！最佳验证损失: {best_val_loss:.4f}")


if __name__ == "__main__":
    torch.set_num_threads(4)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train()