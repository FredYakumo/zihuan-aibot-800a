#!/usr/bin/env python3
"""
LTP Model Test Script
测试 LTP 模型的基本功能
"""
import os
import sys

import torch

# Add the nn directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from nn.models import LTPModel, LTPProcessor, get_device


def test_ltp_model():
    """Test the LTP model functionality"""
    print("=" * 50)
    print("         LTP 模型功能测试")
    print("=" * 50)

    # Get device
    device = get_device()
    print(f"📱 使用设备: {device}")

    # Initialize LTP processor
    print("\n🔧 正在初始化 LTP 处理器...")
    try:
        ltp_processor = LTPProcessor(device=device)
        print("✅ LTP 处理器初始化成功")
    except Exception as e:
        print(f"❌ LTP 处理器初始化失败: {e}")
        return False

    # Test texts
    test_texts = [
        "他叫汤姆去拿外衣。",
        "我爱北京天安门。",
        "自然语言处理是人工智能的重要分支。",
    ]

    print(f"\n📝 测试文本:")
    for i, text in enumerate(test_texts, 1):
        print(f"   {i}. {text}")

    print("\n" + "=" * 50)
    print("开始执行各项功能测试")
    print("=" * 50)

    # Test word segmentation
    print("\n【测试1】中文分词功能")
    print("-" * 30)
    try:
        cws_result = ltp_processor.word_segmentation(test_texts)
        print("✅ 中文分词测试通过")
        if "cws" in cws_result and cws_result["cws"]:
            print("📋 分词结果:")
            for i, (text, words) in enumerate(zip(test_texts, cws_result["cws"]), 1):
                print(f"   测试{i}: {text}")
                print(f"   分词: {' / '.join(words) if words else '无结果'}")
                print()
        else:
            print("⚠️  未获取到分词结果")
    except Exception as e:
        print(f"❌ 中文分词测试失败: {e}")

    # Test POS tagging
    print("\n【测试2】词性标注功能")
    print("-" * 30)
    try:
        pos_result = ltp_processor.pos_tagging(test_texts)
        print("✅ 词性标注测试通过")
        if "pos" in pos_result and pos_result["pos"]:
            print("📋 词性标注结果:")
            for i, (text, pos_tags) in enumerate(zip(test_texts, pos_result["pos"]), 1):
                print(f"   测试{i}: {text}")
                print(f"   词性: {' / '.join(pos_tags) if pos_tags else '无结果'}")
                print()
        else:
            print("⚠️  未获取到词性标注结果")
    except Exception as e:
        print(f"❌ 词性标注测试失败: {e}")

    # Test NER
    print("\n【测试3】命名实体识别功能")
    print("-" * 30)
    try:
        ner_result = ltp_processor.named_entity_recognition(test_texts)
        print("✅ 命名实体识别测试通过")
        if "ner" in ner_result and ner_result["ner"]:
            print("📋 命名实体识别结果:")
            for i, (text, ner_tags) in enumerate(zip(test_texts, ner_result["ner"]), 1):
                print(f"   测试{i}: {text}")
                if ner_tags:
                    entities = []
                    for tag in ner_tags:
                        if tag != "O":  # 'O' 表示非实体
                            entities.append(tag)
                    print(f"   实体: {' / '.join(entities) if entities else '无实体'}")
                else:
                    print("   实体: 无结果")
                print()
        else:
            print("⚠️  未获取到命名实体识别结果")
    except Exception as e:
        print(f"❌ 命名实体识别测试失败: {e}")

    # Test direct model inference
    print("\n【测试4】模型直接推理功能")
    print("-" * 30)
    try:
        test_text = test_texts[0]
        print(f"🔍 正在对文本进行直接推理: {test_text}")
        result = ltp_processor.process_text(test_text)
        print("✅ 模型直接推理测试通过")
        print("📋 推理结果信息:")
        print(f"   结果字段: {list(result.keys())}")
        if "hidden_states" in result:
            print(f"   隐藏状态维度: {result['hidden_states'].shape}")
        if "embeddings" in result:
            print(f"   嵌入向量维度: {result['embeddings'].shape}")
        print()
    except Exception as e:
        print(f"❌ 模型直接推理测试失败: {e}")

    print("\n" + "=" * 50)
    print("🎉 所有测试已完成")
    print("=" * 50)
    return True


if __name__ == "__main__":
    test_ltp_model()
