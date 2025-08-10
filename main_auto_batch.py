import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil
import logging
import logging.handlers
import time
import importlib.util
import glob

def setup_logger():
    """设置日志，避免截断问题"""
    # 设置更大的缓冲区，避免日志截断
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    
    logger = logging.getLogger("auto_batch")
    logger.setLevel(logging.INFO)
    
    # 清除所有handler
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])
    
    # 文件handler
    file_handler = logging.FileHandler('auto_batch_processing.log', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 控制台handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(stream_handler)
    
    # 设置其他模块的日志级别，减少冗余信息
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('numpy').setLevel(logging.WARNING)
    
    return logger

# 配置参数
BATCH_SIZE = 150000
INPUT_FILE = Path('./input/搜索词报告.xlsx')
BATCH_DIR = Path('./input/batches_auto')
OUTPUT_DIR = Path('./output')
HISTORY_DIR = OUTPUT_DIR / 'history'
CACHE_DIR = Path('./cache')

# 确保输出目录存在
OUTPUT_DIR.mkdir(exist_ok=True)

logger = setup_logger()

# 检查历史文件，返回可复用的记录
def check_history_files(df_input):
    """检查历史文件，返回已处理的记录"""
    logger.info("检查历史文件...")
    
    # 获取最近的历史文件（最近10个）
    history_files = list(HISTORY_DIR.glob('result_*.xlsx'))
    if not history_files:
        logger.info("未找到历史文件")
        return pd.DataFrame(), 0
    
    # 按修改时间排序，获取最新的10个文件
    history_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    recent_history_files = history_files[:10]
    logger.info(f"使用最近 {len(recent_history_files)} 个历史文件")
    
    # 合并所有历史文件
    all_history_records = []
    for history_file in recent_history_files:
        try:
            df_history = pd.read_excel(history_file)
            all_history_records.append(df_history)
            logger.info(f"读取历史文件 {history_file.name}: {len(df_history)} 条记录")
        except Exception as e:
            logger.warning(f"读取历史文件 {history_file.name} 失败: {e}")
    
    if not all_history_records:
        logger.info("无法读取任何历史文件")
        return pd.DataFrame(), 0
    
    # 合并所有历史记录
    df_all_history = pd.concat(all_history_records, ignore_index=True)
    # 去重，保留最新的记录
    df_all_history = df_all_history.drop_duplicates(subset=['keyword_name', 'search_word'], keep='last')
    logger.info(f"合并后历史文件包含 {len(df_all_history)} 条唯一记录")
    
    # 检查字段映射
    if '关键词/营销要点/知识问答' in df_input.columns and '搜索词' in df_input.columns:
        # 创建用于比较的键
        df_input['compare_key'] = df_input['关键词/营销要点/知识问答'].astype(str) + '|||' + df_input['搜索词'].astype(str)
        df_all_history['compare_key'] = df_all_history['keyword_name'].astype(str) + '|||' + df_all_history['search_word'].astype(str)
        
        # 找到已存在的记录
        existing_mask = df_input['compare_key'].isin(df_all_history['compare_key'])
        existing_count = existing_mask.sum()
        
        if existing_count > 0:
            logger.info(f"发现 {existing_count} 条记录已存在于历史文件中，将跳过向量计算")
            # 从历史文件中获取已存在记录的相似度
            existing_records = df_input[existing_mask].copy()
            existing_records = existing_records.merge(
                df_all_history[['keyword_name', 'search_word', 'similarity']], 
                left_on=['关键词/营销要点/知识问答', '搜索词'],
                right_on=['keyword_name', 'search_word'],
                how='left'
            )
            # 清理临时列
            existing_records = existing_records.drop(columns=['compare_key'])
            return existing_records, existing_count
        else:
            logger.info("未发现可复用的历史记录")
            return pd.DataFrame(), 0
    else:
        logger.warning("输入文件字段格式不匹配，无法检查历史记录")
        return pd.DataFrame(), 0

# 自动分批
def split_input_to_batches(input_file, batch_size=BATCH_SIZE):
    logger.info(f"读取数据文件: {input_file}")
    df = pd.read_excel(input_file, engine='openpyxl')
    total_records = len(df)
    logger.info(f"总记录数: {total_records}")
    
    # 检查历史文件
    existing_records, existing_count = check_history_files(df)
    
    # 创建需要处理的批次（保留所有记录，但标记哪些需要计算向量）
    BATCH_DIR.mkdir(exist_ok=True)
    batch_files = []
    
    if existing_count > 0:
        logger.info(f"发现 {existing_count} 条记录可复用历史相似度，{total_records - existing_count} 条需要重新计算")
        # 创建包含所有记录的批次，但标记哪些需要计算
        num_batches = (total_records + batch_size - 1) // batch_size
        logger.info(f"将分成 {num_batches} 个批次，每批最多 {batch_size} 条记录")
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_records)
            batch_df = df.iloc[start_idx:end_idx].copy()
            batch_num = i + 1
            batch_file = BATCH_DIR / f'batch_{batch_num:02d}_auto.xlsx'
            batch_df.to_excel(batch_file, index=False)
            batch_files.append(batch_file)
            logger.info(f"创建批次 {batch_num}: {len(batch_df)} 条记录 -> {batch_file}")
    else:
        logger.info("所有记录都需要重新计算")
        num_batches = (total_records + batch_size - 1) // batch_size
        logger.info(f"将分成 {num_batches} 个批次，每批最多 {batch_size} 条记录")
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_records)
            batch_df = df.iloc[start_idx:end_idx]
            batch_num = i + 1
            batch_file = BATCH_DIR / f'batch_{batch_num:02d}_auto.xlsx'
            batch_df.to_excel(batch_file, index=False)
            batch_files.append(batch_file)
            logger.info(f"创建批次 {batch_num}: {len(batch_df)} 条记录 -> {batch_file}")
    
    return batch_files, existing_records

# 处理单个批次，返回结果文件名
def process_single_batch(batch_file, batch_num, main_py="main.py"):
    logger.info(f"\n=== 开始处理批次 {batch_num}: {batch_file.name} ===")
    
    # 检查现有缓存状态
    keyword_cache_file = CACHE_DIR / 'keyword_vectors.db'
    search_cache_file = CACHE_DIR / 'search_vectors.db'
    
    if keyword_cache_file.exists():
        import sqlite3
        try:
            conn = sqlite3.connect(str(keyword_cache_file))
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM vectors')
            keyword_count = cursor.fetchone()[0]
            conn.close()
            logger.info(f"现有关键词缓存: {keyword_count} 条向量")
        except:
            logger.info("关键词缓存文件存在但无法读取")
    else:
        logger.info("关键词缓存文件不存在，将创建新缓存")
    
    if search_cache_file.exists():
        import sqlite3
        try:
            conn = sqlite3.connect(str(search_cache_file))
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM vectors')
            search_count = cursor.fetchone()[0]
            conn.close()
            logger.info(f"现有搜索词缓存: {search_count} 条向量")
        except:
            logger.info("搜索词缓存文件存在但无法读取")
    else:
        logger.info("搜索词缓存文件不存在，将创建新缓存")
    
    # 临时禁用主程序的日志配置，避免重复输出
    import logging
    import sys
    import os
    
    # 在导入主程序之前，先禁用所有可能的日志输出
    original_loggers = {}
    for name in ['transformers', 'torch', 'numpy', 'tqdm']:
        if name in logging.Logger.manager.loggerDict:
            original_loggers[name] = {
                'level': logging.getLogger(name).level,
                'propagate': logging.getLogger(name).propagate
            }
            logging.getLogger(name).setLevel(logging.CRITICAL)
            logging.getLogger(name).propagate = False
    
    # 临时禁用根logger的handlers，防止主程序重新配置日志
    original_root_handlers = logging.getLogger().handlers.copy()
    logging.getLogger().handlers.clear()
    
    # 临时重定向stdout和stderr，防止主程序输出日志
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # 创建一个空的输出流
    class NullStream:
        def write(self, text): pass
        def flush(self): pass
    
    null_stream = NullStream()
    sys.stdout = null_stream
    sys.stderr = null_stream
    
    # 动态导入主程序
    spec = importlib.util.spec_from_file_location("main_module", main_py)
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)
    
    # 设置输入目录为当前批次目录
    main_module.Config.INPUT_DIR = batch_file.parent
    
    # 优化参数 - 根据缓存情况调整
    main_module.Config.BATCH_SIZE = 512  # 使用512批次大小
    main_module.Config.TEXT_BATCH_SIZE = 15000
    main_module.Config.MEMORY_CACHE_SIZE = 150000
    main_module.Config.MEMORY_CLEANUP_INTERVAL = 3
    main_module.Config.ENABLE_DYNAMIC_BATCHING = True
    main_module.Config.MIN_BATCH_SIZE = 64  # 提高最小批次大小
    main_module.Config.MAX_BATCH_SIZE = 1024  # 提高最大批次大小
    main_module.Config.QUEUE_LIMIT = 30000
    
    # 禁用并行处理，避免重复日志和循环问题
    main_module.Config.ENABLE_PARALLEL_PROCESSING = False
    
    # 完全禁用主程序的日志输出
    if hasattr(main_module, 'logger'):
        # 清除主程序的所有handlers
        main_module.logger.handlers.clear()
        # 设置日志级别为CRITICAL，几乎不输出任何日志
        main_module.logger.setLevel(logging.CRITICAL)
        # 禁用传播到根logger
        main_module.logger.propagate = False
    
    # 禁用主程序模块中其他可能的logger
    for name in ['transformers', 'torch', 'numpy', 'tqdm']:
        if name in logging.Logger.manager.loggerDict:
            logging.getLogger(name).setLevel(logging.CRITICAL)
            logging.getLogger(name).propagate = False
    
    # 运行主程序，传递批次文件路径
    t0 = time.time()
    try:
        main_module.tqdm.pandas()
        main_module.main(str(batch_file))
    except Exception as e:
        logger.error(f"批次 {batch_num} 处理失败: {e}")
        return None
    finally:
        # 恢复日志配置
        if hasattr(main_module, 'logger'):
            main_module.logger.propagate = True
        # 恢复原始日志配置
        for name, config in original_loggers.items():
            if name in logging.Logger.manager.loggerDict:
                logging.getLogger(name).setLevel(config['level'])
                logging.getLogger(name).propagate = config['propagate']
        # 恢复根logger的handlers
        logging.getLogger().handlers = original_root_handlers
        # 恢复stdout和stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    t1 = time.time()
    
    # 检查缓存更新情况
    if keyword_cache_file.exists():
        try:
            conn = sqlite3.connect(str(keyword_cache_file))
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM vectors')
            new_keyword_count = cursor.fetchone()[0]
            conn.close()
            logger.info(f"批次 {batch_num} 后关键词缓存: {new_keyword_count} 条向量")
        except:
            pass
    
    if search_cache_file.exists():
        try:
            conn = sqlite3.connect(str(search_cache_file))
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM vectors')
            new_search_count = cursor.fetchone()[0]
            conn.close()
            logger.info(f"批次 {batch_num} 后搜索词缓存: {new_search_count} 条向量")
        except:
            pass
    
    logger.info(f"批次 {batch_num} 处理完成，用时 {t1-t0:.1f} 秒")
    
    # 查找本批次的结果文件
    output_files = list(OUTPUT_DIR.glob(f"{batch_file.stem}_with_similarity_*.xlsx"))
    if output_files:
        logger.info(f"批次 {batch_num} 结果文件: {output_files[-1]}")
        return output_files[-1]
    else:
        logger.warning(f"批次 {batch_num} 未找到结果文件")
        return None

# 合并本次批次的结果 - 针对新信息修正版
def merge_results(result_files, existing_records):
    logger.info("\n开始合并结果文件...")

    # 1. 读取input
    input_df = pd.read_excel(INPUT_FILE)
    if '关键词/营销要点/知识问答' in input_df.columns:
        input_df = input_df.rename(columns={'关键词/营销要点/知识问答': 'keyword_name', '搜索词': 'search_word'})

    # 2. 历史数据处理
    if not existing_records.empty:
        history_clean = pd.DataFrame()
        for col in existing_records.columns:
            if col == '关键词/营销要点/知识问答':
                history_clean['keyword_name'] = existing_records[col]
                continue
            if col == '搜索词':
                history_clean['search_word'] = existing_records[col]
                continue
            if col in ['keyword_name', 'search_word', 'compare_key']:
                continue
            history_clean[col] = existing_records[col]
        if 'compare_key' in history_clean.columns:
            history_clean = history_clean.drop(columns=['compare_key'])
        if 'similarity' not in history_clean.columns:
            logger.error("历史数据缺少similarity列！")
            return
        history_clean = history_clean[['keyword_name', 'search_word', 'similarity']]
    else:
        history_clean = pd.DataFrame(columns=['keyword_name', 'search_word', 'similarity'])

    # 3. 新结果处理
    dfs = []
    for f in result_files:
        if f and f.exists():
            df = pd.read_excel(f)
            if 'compare_key' in df.columns:
                df = df.drop(columns=['compare_key'])
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated(keep='first')]
            dfs.append(df)
    if dfs:
        new_results = pd.concat(dfs, ignore_index=True)
        if 'similarity' not in new_results.columns:
            logger.error("新结果缺少similarity列！")
            return
        new_results = new_results[['keyword_name', 'search_word', 'similarity']]
    else:
        new_results = pd.DataFrame(columns=['keyword_name', 'search_word', 'similarity'])

    # 4. 合并：以input为基准，优先历史
    merged = input_df.merge(history_clean, on=['keyword_name', 'search_word'], how='left', suffixes=('', '_hist'))
    merged = merged.merge(new_results, on=['keyword_name', 'search_word'], how='left', suffixes=('', '_new'))

    # 5. 选择相似度
    merged['final_similarity'] = merged['similarity']
    merged.loc[merged['final_similarity'].isna(), 'final_similarity'] = merged.loc[merged['final_similarity'].isna(), 'similarity_new']
    merged = merged.drop(columns=['similarity', 'similarity_new'])

    # 6. 列顺序
    desired_order = [
        '日期', '账户', '账户ID', '推广单元', '推广单元ID', 
        'keyword_name', '关键词/营销要点ID', 'search_word', 
        '触发模式', '账户添加状态', '匹配模式', '展现量', '点击量', 
        '消费', '激活人数成本', '激活次日留存率', '平均点击价格', '点击率', 
        '上方位展现', '上方位点击', '上方位消费', '上方首位展现', 
        '激活人数', '激活人数转化率', '次日留存人数', 'final_similarity'
    ]
    existing_cols = merged.columns.tolist()
    final_col_order = [col for col in desired_order if col in existing_cols]
    for col in existing_cols:
        if col not in final_col_order:
            final_col_order.append(col)
    merged = merged[final_col_order]

    # 7. 对所有字段做完全去重，保留第一条
    before_drop = len(merged)
    merged = merged.drop_duplicates(keep='first')
    after_drop = len(merged)
    logger.info(f"去重后总记录数: {after_drop}（去除 {before_drop - after_drop} 条完全重复行）")

    timestamp = datetime.strftime(datetime.now(), "%Y%m%d_%H%M")
    final_file = OUTPUT_DIR / f'final_auto_merged_{timestamp}.xlsx'
    merged.to_excel(final_file, index=False)
    logger.info(f"合并完成: {final_file}")
    logger.info(f"最终记录数: {len(merged)}")

if __name__ == '__main__':
    total_start_time = time.time()
    logger.info("\n==== 自动批量处理流程开始 ====")
    
    # 检查是否需要清理缓存
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--clear-cache':
        logger.info("清理现有缓存...")
        for cache_file in [CACHE_DIR / 'keyword_vectors.db', CACHE_DIR / 'search_vectors.db']:
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"已删除缓存文件: {cache_file}")
        logger.info("缓存清理完成")
    
    if not INPUT_FILE.exists():
        logger.error(f"输入文件不存在: {INPUT_FILE}")
        exit(1)
    
    batch_files, existing_records = split_input_to_batches(INPUT_FILE, batch_size=BATCH_SIZE)
    result_files = []
    
    if len(batch_files) == 0:
        logger.info("所有记录都已存在于历史文件中，无需处理新批次")
    else:
        logger.info(f"\n开始处理 {len(batch_files)} 个批次...")
        logger.info("提示: 如需清理缓存重新开始，请使用: python main_auto_batch.py --clear-cache")
        
        for i, batch_file in enumerate(batch_files, 1):
            result_file = process_single_batch(batch_file, i)
            if result_file:
                result_files.append(result_file)
    
    merge_results(result_files, existing_records)
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    logger.info(f"\n==== 全部批次处理与合并完成 ====")
    logger.info(f"总运行时间: {hours}小时 {minutes}分钟 {seconds}秒 ({total_time:.1f}秒)") 
