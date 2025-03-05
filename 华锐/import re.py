import re
import jieba
import jieba.posseg as pseg
import sys


sys.stdout.reconfigure(encoding='utf-8')

# 示例文本
text = "单只可转债 / 可交债占净值不得超 10%。
text  = "单只基金持有同一（同一信用级别）资产支持证券不超 10%。"
text = "单只组合单股票持有量不超过股票总股本不超过 5% a 股 + cdr + 转股。"

# 1. 数据预处理
# 1.1 文本清洗
def clean_text(text):
    # 移除特殊字符（保留百分号 % 和加号 +），并处理多余空格
    cleaned_text = re.sub(r'[^%\+A-Za-z0-9\u4e00-\u9fa5]', ' ', text)  # 保留中文、英文、数字、百分号和加号
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # 处理多余空格
    cleaned_text = cleaned_text.lower()  # 转换为小写
    return cleaned_text

cleaned_text = clean_text(text)
print(f"清洗后的文本: {cleaned_text}")

# 1.2 分词处理
def segment_text(text):
    # 使用 jieba 分词
    seg_list = jieba.cut(text)
    return list(seg_list)

seg_result = segment_text(cleaned_text)
print(f"分词结果: {'/'.join(seg_result)}")

# 1.3 词性标注
def pos_tagging(text):
    # 使用 jieba 的词性标注功能
    words = pseg.cut(text)
    pos_result = []
    for word, flag in words:
        pos_result.append(f"{word}/{flag}")
    return pos_result

pos_result = pos_tagging(cleaned_text)
print(f"词性标注结果: {'/'.join(pos_result)}")

# 2. 规则提取与标准化
# 2.1 识别数量词和限制词
def extract_quantifiers_and_restrictions(pos_result):
    quantifiers = []
    restrictions = []
    
    for item in pos_result:
        word = item.split("/")[0]
        pos = item.split("/")[1]
        
        # 提取数量词（如 5%）
        if pos == "m" and "%" in word:
            quantifiers.append(word)
        
        # 提取限制词（如 不超过）
        if word in ["不超过", "不得超过", "不得超", "不得超过"]:
            restrictions.append(word)
    
    return quantifiers, restrictions

quantifiers, restrictions = extract_quantifiers_and_restrictions(pos_result)
print(f"提取的数量词: {quantifiers}")
print(f"提取的限制词: {restrictions}")

# 2.2 确定投资标的相关信息
def extract_investment_targets(seg_result):
    # 定义投资标的相关词汇
    investment_keywords = {"a股", "cdr", "转股", "股票", "债券", "资产支持证券"}
    targets = []
    for word in seg_result:
        if word in investment_keywords:
            targets.append(word)
    return list(set(targets))  # 去重

targets = extract_investment_targets(seg_result)
print(f"提取的投资标的: {targets}")

# 2.3 标准化规则表述
def standardize_rules(quantifiers, restrictions, targets):
    # 示例：投资标的：a 股、cdr、转股；限制条件：持有比例不超过 5%
    standardized_rule = f"投资标的：{', '.join(targets)}；限制条件："
    
    for i in range(len(restrictions)):
        standardized_rule += f"{restrictions[i]} {quantifiers[i]}；"
    
    return standardized_rule.rstrip("；")  # 去掉最后一个分号

standardized_rule = standardize_rules(quantifiers, restrictions, targets)
print(f"标准化规则表述: {standardized_rule}")