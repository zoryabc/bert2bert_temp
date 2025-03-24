import json
import torch
import jieba
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    AutoModelForSeq2SeqLM,
    EncoderDecoderModel,
    BertTokenizer,
    BertConfig,
    GenerationMixin
)

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CustomEncoderDecoder(EncoderDecoderModel, GenerationMixin):
    """继承GenerationMixin解决生成警告问题"""

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return super().prepare_inputs_for_generation(*args, **kwargs)


def generate_summary(model, tokenizer, text):
    """优化后的摘要生成函数"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=128,
            num_beams=6,
            early_stopping=True,
            decoder_start_token_id=tokenizer.cls_token_id,
            bos_token_id=tokenizer.cls_token_id,
            no_repeat_ngram_size=3,
            length_penalty=0.9
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def load_models():
    """修复配置问题的模型加载函数"""
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    # 加载训练模型
    trained_model = AutoModelForSeq2SeqLM.from_pretrained(
        r"C:\Users\76472\Desktop\bert2bert_summary\bert2bert_summary\models",
        trust_remote_code=True,  # 解决继承问题
        local_files_only=True
    )

    # 配置基础模型
    decoder_config = BertConfig.from_pretrained("bert-base-chinese")
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True

    base_model = CustomEncoderDecoder.from_encoder_decoder_pretrained(
        "bert-base-chinese",
        "bert-base-chinese",
        decoder_config=decoder_config,
        tie_weights=True  # 绑定权重解决初始化警告
    )

    # 显式设置关键参数
    base_model.config.update({
        "decoder_start_token_id": tokenizer.cls_token_id,
        "bos_token_id": tokenizer.cls_token_id,
        "pad_token_id": tokenizer.pad_token_id
    })

    return base_model, trained_model, tokenizer


def evaluate_models(test_data, base_model, trained_model, tokenizer):
    """增强型评估函数"""
    # 中文分词配置（关键修复点）
    jieba.initialize()
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=False,
        tokenizer=lambda x: jieba.lcut(x)  # 使用lcut直接生成列表
    )
    smooth_func = SmoothingFunction().method1

    for model_type, model in [("base", base_model), ("trained", trained_model)]:
        model.to(DEVICE)
        model.eval()

        valid_samples = 0
        progress_bar = tqdm(test_data[:200], desc=f"评估 {model_type} 模型")

        for sample in progress_bar:
            try:
                text = sample.get("text", "")
                true_summary = sample.get("summary", "")

                if len(text) < 10 or len(true_summary) < 3:  # 过滤无效样本
                    continue

                # 生成摘要
                pred_summary = generate_summary(model, tokenizer, text)
                if len(pred_summary.strip()) < 5:  # 过滤无效输出
                    continue

                # 中文BLEU计算
                ref_tokens = list(jieba.cut(true_summary))  # 使用 jieba.cut
                hyp_tokens = list(jieba.cut(pred_summary))  # 使用 jieba.cut
                results[model_type]["bleu"] += sentence_bleu(
                    [ref_tokens], hyp_tokens,
                    weights=(0.5, 0.5, 0, 0),
                    smoothing_function=smooth_func  # 平滑处理
                )

                # ROUGE计算
                scores = scorer.score(true_summary, pred_summary)
                results[model_type]["rouge1"] += scores['rouge1'].fmeasure
                results[model_type]["rouge2"] += scores['rouge2'].fmeasure
                results[model_type]["rougeL"] += scores['rougeL'].fmeasure

                valid_samples += 1
                progress_bar.set_postfix({
                    "BLEU": f"{results[model_type]['bleu'] / valid_samples:.3f}",
                    "有效样本": valid_samples
                })

            except Exception as e:
                print(f"\n处理样本时发生错误: {type(e).__name__}-{str(e)}")
                continue

        # 计算平均指标
        if valid_samples > 0:
            for metric in ["bleu", "rouge1", "rouge2", "rougeL"]:
                results[model_type][metric] /= valid_samples
        else:
            print(f"\n{model_type}模型无有效样本")

    return results


if __name__ == "__main__":
    # 加载测试数据
    data_path = r"C:\Users\76472\Desktop\bert2bert_summary\bert2bert_summary\data\lcsts.json"
    with open(data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)[-1000:]  # 使用最新1000条测试数据

    # 初始化模型
    base_model, trained_model, tokenizer = load_models()

    # 执行评估
    results = evaluate_models(test_data, base_model, trained_model, tokenizer)

    # 打印结果
    print("\n评估结果对比：")
    print(f"| 模型类型 | BLEU-2 | ROUGE-1 | ROUGE-2 | ROUGE-L |")
    print("|----------|--------|---------|---------|---------|")
    for model_type in ["base", "trained"]:
        data = results[model_type]
        print(
            f"| {model_type.ljust(8)} | {data['bleu']:.4f} | {data['rouge1']:.4f} | {data['rouge2']:.4f} | {data['rougeL']:.4f} |")