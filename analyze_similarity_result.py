import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

# 默认读取output目录下最新的final_auto_merged_*.xlsx
OUTPUT_DIR = Path('./output')
ANALYSIS_DIR = OUTPUT_DIR / 'analysis'

# 查找最新的合并结果文件
result_files = sorted(OUTPUT_DIR.glob('final_auto_merged_*.xlsx'), key=lambda x: x.stat().st_mtime, reverse=True)
if not result_files:
    print('未找到合并结果文件！')
    sys.exit(1)

result_file = result_files[0]
print(f'分析文件: {result_file}')

df = pd.read_excel(result_file)

# 判断用的列名
sim_col = None
for col in ['final_similarity', 'similarity']:
    if col in df.columns:
        sim_col = col
        break
if sim_col is None:
    print('未找到相似度列！')
    sys.exit(1)

# 相关性判断

def judge_relation(sim):
    if pd.isna(sim):
        return '无相似度'
    if sim >= 0.80:
        return '强相关'
    elif sim >= 0.65:
        return '较相关'
    elif sim >= 0.5:
        return '弱相关'
    else:
        return '不相关'

df['相关性判断（建议）'] = df[sim_col].apply(judge_relation)

def get_opt_action(row):
    status = row.get('账户添加状态', None)
    relation = row['相关性判断（建议）']
    if status == '未添加':
        if relation in ['强相关', '较相关']:
            return '添加至账户内'
        elif relation == '弱相关':
            return '需人工判断处理'
        elif relation == '不相关':
            return '加入否词'
        else:
            return '需人工判断处理'
    elif status == '已添加':
        return '已添加，无需操作'
    else:
        return '账户状态未知，需人工判断'

# 新增建议优化动作列（必须在保存前）
if '账户添加状态' in df.columns:
    df['建议优化动作'] = df.apply(get_opt_action, axis=1)
else:
    df['建议优化动作'] = '缺少账户添加状态字段，无法判断'

# 生成分析结果目录（带日期）
today_str = datetime.now().strftime('%Y%m%d')
output_dir = ANALYSIS_DIR / today_str
output_dir.mkdir(parents=True, exist_ok=True)

# 保存新文件，文件名与原文件一致
new_file = output_dir / (result_file.stem + '_with_relation.xlsx')
df.to_excel(new_file, index=False)
print(f'已生成带相关性判断的新文件: {new_file}') 