import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import sqlite3
import re

# 配置类
class Config:
    MODEL_PATH = Path('D:/model/BAAI-bge-large-zh-v1.5')  # 模型路径
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 计算设备
    CACHE_DIR = Path('./cache')
    BATCH_SIZE = 128  # 向量化批处理大小
    INVALID_LEVEL_WORDS = set(['核心', '非核心', '通用', '程序化', '通用词', '核', '通用词0'])

# 默认读取output/analysis目录下最新日期文件夹中的分析结果
OUTPUT_DIR = Path('./output')
ANALYSIS_DIR = OUTPUT_DIR / 'analysis'

def find_latest_analysis_file():
    """
    查找最新的分析结果文件
    """
    # 获取最新的日期文件夹
    ANALYSIS_DIR = Path('./output/analysis')
    date_dirs = sorted([d for d in ANALYSIS_DIR.iterdir() if d.is_dir()], reverse=True)
    if not date_dirs:
        print('未找到分析结果目录！')
        sys.exit(1)
    
    latest_dir = date_dirs[0]
    # 查找最新的分析文件
    result_files = sorted(latest_dir.glob('*_with_relation.xlsx'), 
                         key=lambda x: x.stat().st_mtime, 
                         reverse=True)
    if not result_files:
        print('未找到分析结果文件！')
        sys.exit(1)
    
    return result_files[0]

def split_ad_unit(unit_name):
    """
    将推广单元按照-和_分隔符拆分成层级
    返回一个列表，包含各层级名称（最多3个）
    """
    if pd.isna(unit_name):
        return []
    
    # 先按-分割
    parts = str(unit_name).split('-')
    # 对每个部分再按_分割
    levels = []
    for part in parts:
        levels.extend(part.split('_'))
    levels = [level.strip() for level in levels if level.strip()]
    
    # 只返回前三个层级
    return levels[:3]

def split_ad_unit_fixed(unit_name):
    levels = split_ad_unit(unit_name)
    return (levels + [None] * 3)[:3]

def is_invalid_level_word(word):
    if pd.isna(word):
        return True
    word = str(word).strip()
    if word in Config.INVALID_LEVEL_WORDS:
        return True
    if re.fullmatch(r'\d+', word):  # 纯数字
        return True
    return False

def find_valid_min_level(row):
    # 从最细层级往前找，直到遇到有效层级
    for i in range(3, 0, -1):
        level = row.get(f'层级{i}')
        if not is_invalid_level_word(level):
            return i, level
    return None, None

def get_vector_from_cache(text, cache_db):
    if pd.isna(text) or not str(text).strip():
        return None
    conn = sqlite3.connect(str(cache_db))
    cursor = conn.cursor()
    cursor.execute('SELECT vector, dimension FROM vectors WHERE text = ?', (text,))
    result = cursor.fetchone()
    conn.close()
    if result:
        vector = torch.from_numpy(np.frombuffer(result[0], dtype=np.float32).copy()).to(Config.DEVICE)
        if vector.size(0) != result[1]:
            return None
        return vector
    return None

def batch_calc_and_cache(texts, tokenizer, model, cache_db):
    texts = [t for t in texts if t and not pd.isna(t)]
    texts = list(set(texts))
    uncached = []
    cached = {}
    conn = sqlite3.connect(str(cache_db))
    cursor = conn.cursor()
    # 新增：确保表结构存在
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vectors (
            text TEXT PRIMARY KEY,
            vector BLOB,
            dimension INTEGER
        )
    ''')
    for t in texts:
        cursor.execute('SELECT vector, dimension FROM vectors WHERE text = ?', (t,))
        result = cursor.fetchone()
        if result:
            vector = torch.from_numpy(np.frombuffer(result[0], dtype=np.float32).copy()).to(Config.DEVICE)
            if vector.size(0) == result[1]:
                cached[t] = vector
            else:
                uncached.append(t)
        else:
            uncached.append(t)
    if uncached:
        tokenizer_inputs = tokenizer(uncached, return_tensors='pt', truncation=True, padding=True, max_length=128)
        for k in tokenizer_inputs:
            tokenizer_inputs[k] = tokenizer_inputs[k].to(Config.DEVICE)
        with torch.no_grad():
            outputs = model(**tokenizer_inputs).last_hidden_state[:, 0, :].to(torch.float32)
        for t, vec in zip(uncached, outputs):
            cursor.execute('INSERT OR REPLACE INTO vectors (text, vector, dimension) VALUES (?, ?, ?)',
                           (t, vec.cpu().numpy().tobytes(), vec.size(0)))
            cached[t] = vec
        conn.commit()
    conn.close()
    return cached

def main():
    # 查找最新的分析文件
    latest_file = find_latest_analysis_file()
    print(f'分析文件: {latest_file}')
    
    # 读取全部数据
    df = pd.read_excel(latest_file)
    print(f'原始数据行数: {len(df)}')
    print('数据列名:', list(df.columns))
    
    # 层级拆分（保证每行只有3个元素）
    levels_df = pd.DataFrame([split_ad_unit_fixed(unit) for unit in df['推广单元']], columns=['层级1', '层级2', '层级3'])
    df = pd.concat([df.reset_index(drop=True), levels_df], axis=1)
    # 有效最小层级赋值（直接在df主表上）
    valid_min_levels = df.apply(find_valid_min_level, axis=1)
    df['有效最小层级序号'] = [x[0] for x in valid_min_levels]
    df['有效最小层级名称'] = [x[1] for x in valid_min_levels]
    print(df[['推广单元', '层级1', '层级2', '层级3', '有效最小层级序号', '有效最小层级名称']].head())
    print('有效最小层级分布:')
    print(df['有效最小层级名称'].value_counts().head(20))
    # 只保留有有效最小层级的
    valid_df = df[~df['有效最小层级名称'].isna()].copy()
    all_levels = valid_df['有效最小层级名称'].dropna().unique().tolist()
    print(f'需要计算向量的唯一层级词数量: {len(all_levels)}')

    # 加载模型
    print('加载模型...')
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
    model = AutoModel.from_pretrained(Config.MODEL_PATH).to(Config.DEVICE)

    # 计算并缓存所有层级向量
    print('批量计算并缓存层级向量...')
    level_vecs = batch_calc_and_cache(all_levels, tokenizer, model, Config.CACHE_DIR / 'unit_level_vectors.db')
    print('加载关键词向量缓存...')
    # 关键词向量缓存
    keyword_cache_db = Config.CACHE_DIR / 'keyword_vectors.db'

    # 计算相似度
    print('计算相似度...')
    sim_list = []
    for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df)):
        level_word = row['有效最小层级名称']
        keyword = row['keyword_name']
        level_vec = level_vecs.get(level_word)
        keyword_vec = get_vector_from_cache(keyword, keyword_cache_db)
        if level_vec is not None and keyword_vec is not None:
            sim = torch.nn.functional.cosine_similarity(level_vec.unsqueeze(0), keyword_vec.unsqueeze(0)).item()
        else:
            sim = np.nan
        sim_list.append(sim)
    valid_df['有效最小层级相似度'] = sim_list
    # 输出
    output_dir = latest_file.parent / 'unit_level_analysis'
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    result_file = output_dir / f'unit_level_valid_min_similarity_{timestamp}.xlsx'
    valid_df.to_excel(result_file, index=False)
    print(f'已输出有效最小层级相似度分析: {result_file}')

if __name__ == '__main__':
    main() 