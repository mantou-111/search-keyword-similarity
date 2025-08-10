#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU版本主程序 - 避免GPU内存问题
"""

import pandas as pd
import torch
import logging
import logging.handlers
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
from datetime import datetime
import re
import time
import sqlite3
import queue
import threading
import psutil

# 导入CPU配置
from config_cpu import Config, set_cpu_performance_profile

# 初始化日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            Config.OUTPUT_DIR / 'cpu_similarity_analysis.log',
            maxBytes=10*1024*1024,
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 向量缓存类（CPU版本）
class VectorCache:
    def __init__(self, cache_file):
        # 初始化缓存
        self.cache_file = Config.CACHE_DIR / cache_file
        self.memory_cache = {}
        self.conn = sqlite3.connect(str(self.cache_file), check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.write_failures = []
        self.lock = threading.Lock()
        self._init_db()
        self.queue = queue.Queue(maxsize=Config.QUEUE_LIMIT)
        self.writer_thread = threading.Thread(target=self._writer)
        self.writer_thread.daemon = True
        self.writer_thread.start()
        self.cursor.execute('SELECT COUNT(*) FROM vectors')
        logger.info(f"初始化缓存 {cache_file}: 已有 {self.cursor.fetchone()[0]} 条向量")

    def _init_db(self):
        # 初始化数据库表和索引
        with self.lock:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS vectors (
                    text TEXT PRIMARY KEY,
                    vector BLOB,
                    dimension INTEGER
                )
            ''')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_text ON vectors (text)')
            self.cursor.execute('ANALYZE vectors')
            self.conn.commit()

    def _writer(self):
        # 异步写入线程
        batch = []
        last_write = time.time()
        while True:
            try:
                item = self.queue.get(timeout=3600)
                if item is None:
                    if batch:
                        self._batch_write(batch)
                    break
                text, vector = item
                batch.append((text, vector.numpy().tobytes(), vector.size(0)))
                if len(batch) >= 1000 or time.time() - last_write > 10:  # CPU模式更频繁写入
                    logger.debug(f"写入缓存 {len(batch)} 条, 队列大小: {self.queue.qsize()}")
                    self._batch_write(batch)
                    batch = []
                    last_write = time.time()
                self.queue.task_done()
            except queue.Empty:
                if batch:
                    logger.debug(f"写入缓存 {len(batch)} 条, 队列大小: {self.queue.qsize()}")
                    self._batch_write(batch)
                    batch = []
                    last_write = time.time()
                continue
            except Exception as e:
                logger.error(f"缓存写入线程错误: {str(e)}")

    def _batch_write(self, batch):
        # 批量写入数据库
        try:
            with self.lock:
                conn = sqlite3.connect(str(self.cache_file), check_same_thread=False)
                cursor = conn.cursor()
                cursor.execute('BEGIN TRANSACTION')
                cursor.executemany('INSERT OR REPLACE INTO vectors (text, vector, dimension) VALUES (?, ?, ?)', batch)
                conn.commit()
                conn.close()
        except Exception as e:
            logger.error(f"批量缓存写入失败: {str(e)}")
            self.write_failures.extend([item[0] for item in batch])
            if 'conn' in locals():
                conn.rollback()
                conn.close()

    def get(self, text):
        # 获取单个向量
        if not isinstance(text, str) or not text.strip():
            logger.debug(f"无效文本跳过查询: {text}")
            return None
        if text in self.memory_cache:
            return self.memory_cache[text]
        try:
            with self.lock:
                self.cursor.execute('SELECT vector, dimension FROM vectors WHERE text = ?', (text,))
                result = self.cursor.fetchone()
            if result:
                vector = torch.from_numpy(np.frombuffer(result[0], dtype=np.float32).copy())
                if vector.size(0) != result[1]:
                    logger.error(f"向量维度不匹配: {text}, 缓存维度={result[1]}, 实际={vector.size(0)}")
                    return None
                self.memory_cache[text] = vector
                if len(self.memory_cache) > Config.MEMORY_CACHE_SIZE:
                    self.memory_cache.pop(next(iter(self.memory_cache)))
                return vector
            return None
        except Exception as e:
            logger.error(f"缓存查询失败: {text}, 错误: {str(e)}")
            return None

    def get_batch(self, texts, desc="缓存查询"):
        # 批量获取向量
        start_time = time.time()
        vectors = [None] * len(texts)
        text_to_index = {text: i for i, text in enumerate(texts) if isinstance(text, str) and text.strip()}
        valid_texts = list(text_to_index.keys())
        cached_count = 0

        # 内存缓存查询
        for text in valid_texts:
            if text in self.memory_cache:
                vectors[text_to_index[text]] = self.memory_cache[text]
                cached_count += 1

        # SQLite缓存查询
        uncached = [text for text in valid_texts if text not in self.memory_cache]
        if uncached:
            batch_size = 500  # CPU模式较小的批量查询
            for i in range(0, len(uncached), batch_size):
                batch_uncached = uncached[i:i+batch_size]
                placeholders = ','.join('?' * len(batch_uncached))
                try:
                    with self.lock:
                        conn = sqlite3.connect(str(self.cache_file), check_same_thread=False)
                        cursor = conn.cursor()
                        cursor.execute('PRAGMA temp_store = MEMORY')
                        cursor.execute('PRAGMA cache_size = 5000')
                        cursor.execute(f'SELECT text, vector, dimension FROM vectors WHERE text IN ({placeholders})', batch_uncached)
                        
                        results = cursor.fetchall()
                        for text, vector, dim in results:
                            vector = torch.from_numpy(np.frombuffer(vector, dtype=np.float32).copy())
                            if vector.size(0) != dim:
                                logger.error(f"向量维度不匹配: {text}, 缓存维度={dim}, 实际={vector.size(0)}")
                                continue
                            self.memory_cache[text] = vector
                            if len(self.memory_cache) > Config.MEMORY_CACHE_SIZE:
                                self.memory_cache.pop(next(iter(self.memory_cache)))
                            vectors[text_to_index[text]] = vector
                            cached_count += 1
                        conn.close()
                except Exception as e:
                    logger.error(f"SQLite 查询失败 (批次 {i//batch_size + 1}): {str(e)}")
        
        hit_rate = (cached_count / len(valid_texts)) * 100 if valid_texts else 0
        query_time = time.time() - start_time
        logger.debug(f"{desc} 查询耗时: {query_time:.3f}秒, 命中率: {hit_rate:.1f}%")
        return vectors, cached_count, len(valid_texts)

    def add(self, text, vector):
        # 添加向量到缓存
        try:
            if not isinstance(text, str) or not text.strip():
                logger.debug(f"无效文本跳过缓存: {text}")
                return
            if not isinstance(vector, torch.Tensor) or vector.numel() == 0:
                logger.debug(f"无效向量跳过缓存: {text}")
                return
            self.memory_cache[text] = vector
            if len(self.memory_cache) > Config.MEMORY_CACHE_SIZE:
                self.memory_cache.pop(next(iter(self.memory_cache)))
            try:
                self.queue.put((text, vector), block=False)
            except queue.Full:
                with self.lock:
                    conn = sqlite3.connect(str(self.cache_file), check_same_thread=False)
                    cursor = conn.cursor()
                    cursor.execute('BEGIN TRANSACTION')
                    cursor.execute('INSERT OR REPLACE INTO vectors (text, vector, dimension) VALUES (?, ?, ?)', 
                                  (text, vector.numpy().tobytes(), vector.size(0)))
                    conn.commit()
                    conn.close()
        except Exception as e:
            self.write_failures.append(text)
            logger.error(f"缓存添加失败: {text}, 错误: {str(e)}")

    def close(self):
        # 关闭缓存
        self.queue.put(None)
        self.writer_thread.join()
        if self.write_failures:
            logger.warning(f"缓存写入失败 {len(self.write_failures)} 次，样例: {self.write_failures[:5]}")
        with self.lock:
            self.conn.close()

# 内存监控工具函数（CPU版本）
def monitor_memory_usage():
    """监控当前内存使用情况"""
    memory = psutil.virtual_memory()
    memory_gb = memory.used / (1024**3)
    memory_total_gb = memory.total / (1024**3)
    memory_percent = memory.percent
    logger.debug(f"系统内存使用: {memory_gb:.2f}GB / {memory_total_gb:.1f}GB ({memory_percent:.1f}%)")
    return memory_gb, memory_total_gb, memory_percent

def check_memory_safety():
    """检查内存安全性"""
    memory_gb, memory_total_gb, memory_percent = monitor_memory_usage()
    if memory_percent > 90:
        logger.error(f"系统内存使用率过高: {memory_percent:.1f}%")
        return False
    elif memory_percent > 80:
        logger.warning(f"系统内存使用率较高: {memory_percent:.1f}%")
        return True
    return True

def emergency_memory_cleanup():
    """紧急内存清理"""
    logger.warning("执行紧急内存清理...")
    import gc
    gc.collect()
    memory_gb, memory_total_gb, memory_percent = monitor_memory_usage()
    if memory_percent > 90:
        logger.error(f"系统内存使用率仍然过高: {memory_percent:.1f}%")
        return False
    logger.info(f"紧急内存清理完成，当前使用率: {memory_percent:.1f}%")
    return True

# 文本清理函数
def clean_text(text):
    # 清理文本，保留有效字符
    if pd.isna(text) or text is None or str(text).strip().lower() in ['nan', 'null', '']:
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。？！：；""''、·…—\-—～《》]', '', text)
    return text[:Config.MAX_LENGTH]

# 批量向量化（CPU版本）
def batch_embedding(texts, tokenizer, model, cache, batch_size=Config.BATCH_SIZE, desc="处理向量", max_retries=3):
    # 批量生成向量并写入缓存
    start_time = time.time()
    vectors = [None] * len(texts)
    valid_texts = [None] * len(texts)
    text_batch_size = Config.TEXT_BATCH_SIZE
    total_uncached = 0
    total_valid = 0
    invalid_count = 0
    error_count = 0
    total_batches = (len(texts) + text_batch_size - 1) // text_batch_size
    
    # 性能统计
    processing_stats = {
        'cache_hits': 0,
        'cache_misses': 0,
        'processing_time': 0,
        'memory_cleanups': 0,
        'retry_count': 0,
        'batch_size_adjustments': 0
    }
    
    # 动态批次大小调整
    current_batch_size = batch_size
    consecutive_oom_errors = 0
    consecutive_successes = 0

    logger.info(f"{desc} 开始处理 {len(texts)} 个文本，分为 {total_batches} 个批次")

    for batch_idx in tqdm(range(0, len(texts), text_batch_size), desc=desc, total=total_batches):
        batch_start_time = time.time()
        batch_texts = texts[batch_idx:batch_idx + text_batch_size]
        
        # 缓存查询
        logger.debug(f"{desc} 开始缓存查询，批次 {batch_idx//text_batch_size + 1}/{total_batches}")
        cached_vectors, cached_count, valid_count = cache.get_batch(batch_texts, desc=f"{desc}")
        processing_stats['cache_hits'] += cached_count
        processing_stats['cache_misses'] += (valid_count - cached_count)
        total_uncached += (valid_count - cached_count)
        total_valid += valid_count
        uncached_indices = [i for i, vec in enumerate(cached_vectors) if vec is None]
        uncached_texts = [batch_texts[i] for i in uncached_indices]
        
        logger.debug(f"{desc} 批次 {batch_idx//text_batch_size + 1} 缓存命中: {cached_count}/{valid_count}")
        
        # 处理未缓存的文本
        batch_vectors = []
        batch_valid_texts = []
        if uncached_texts:
            logger.debug(f"{desc} 开始处理 {len(uncached_texts)} 个未缓存文本")
            for i in range(0, len(uncached_texts), current_batch_size):
                sub_batch_texts = uncached_texts[i:i+current_batch_size]
                sub_batch_texts = [t[:Config.MAX_LENGTH] for t in sub_batch_texts if isinstance(t, str) and len(t.strip()) > 0]
                if not sub_batch_texts:
                    invalid_count += len(uncached_texts[i:i+current_batch_size])
                    batch_vectors.extend([None] * len(uncached_texts[i:i+current_batch_size]))
                    batch_valid_texts.extend(uncached_texts[i:i+current_batch_size])
                    continue
                
                # 添加重试机制
                success = False
                for retry in range(max_retries):
                    try:
                        logger.debug(f"{desc} 处理子批次 {i//current_batch_size + 1}，大小: {len(sub_batch_texts)}")
                        inputs = tokenizer(sub_batch_texts, return_tensors='pt', truncation=True, padding=True,
                                          max_length=Config.MAX_LENGTH)
                        with torch.no_grad():
                            outputs = model(**inputs).last_hidden_state[:, 0, :].to(torch.float32)
                        
                        for text, vector in zip(sub_batch_texts, outputs):
                            cache.add(text, vector)
                            batch_vectors.append(vector)
                            batch_valid_texts.append(text)
                        
                        # 显式释放张量
                        del inputs, outputs
                        success = True
                        consecutive_successes += 1
                        consecutive_oom_errors = 0
                        
                        # 动态调整批次大小：成功时尝试增加
                        if (Config.ENABLE_DYNAMIC_BATCHING and 
                            consecutive_successes >= 3 and 
                            current_batch_size < Config.MAX_BATCH_SIZE):
                            new_batch_size = min(current_batch_size * 2, Config.MAX_BATCH_SIZE)
                            if new_batch_size != current_batch_size:
                                logger.info(f"{desc} 批次大小调整: {current_batch_size} -> {new_batch_size}")
                                current_batch_size = new_batch_size
                                processing_stats['batch_size_adjustments'] += 1
                        break
                        
                    except Exception as e:
                        processing_stats['retry_count'] += 1
                        consecutive_successes = 0
                        logger.error(f"{desc} 第 {batch_idx//text_batch_size + 1} 批 批量向量化失败 (重试 {retry + 1}/{max_retries}): {str(e)}, 批次文本样例: {sub_batch_texts[:3]}")
                        if retry == max_retries - 1:
                            batch_vectors.extend([None] * len(sub_batch_texts))
                            batch_valid_texts.extend(sub_batch_texts)
                            error_count += len(sub_batch_texts)
                        else:
                            time.sleep(1)  # 重试前等待
        
        # 合并批次结果
        text_to_global_index = {text: batch_idx + i for i, text in enumerate(batch_texts)}
        
        for i, vec in enumerate(cached_vectors):
            if vec is not None:
                global_idx = batch_idx + i
                vectors[global_idx] = vec
                valid_texts[global_idx] = batch_texts[i]
        
        for text, vector in zip(batch_valid_texts, batch_vectors):
            if text in text_to_global_index:
                global_idx = text_to_global_index[text]
                vectors[global_idx] = vector
                valid_texts[global_idx] = text
        
        # 定期内存清理和检查
        if Config.ENABLE_MEMORY_MONITORING and (batch_idx // text_batch_size) % Config.MEMORY_CLEANUP_INTERVAL == 0:
            # 检查内存安全性
            if not check_memory_safety():
                logger.error("系统内存使用率过高，执行紧急清理")
                if not emergency_memory_cleanup():
                    logger.error("紧急内存清理失败，建议重启程序")
                    break
            
            import gc
            gc.collect()
            processing_stats['memory_cleanups'] += 1
            monitor_memory_usage()
        
        processing_stats['processing_time'] += time.time() - batch_start_time
        
        # 每5个批次输出一次进度
        if (batch_idx // text_batch_size + 1) % 5 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_batch = elapsed_time / (batch_idx // text_batch_size + 1)
            remaining_batches = total_batches - (batch_idx // text_batch_size + 1)
            estimated_remaining_time = remaining_batches * avg_time_per_batch
            logger.info(f"{desc} 进度: {batch_idx // text_batch_size + 1}/{total_batches} 批次完成，"
                       f"预计剩余时间: {estimated_remaining_time/60:.1f}分钟")
    
    if invalid_count > 0:
        logger.debug(f"{desc} 无效文本数: {invalid_count}")
    if error_count > 0:
        logger.warning(f"{desc} 处理失败文本数: {error_count}")
    
    # 详细统计信息
    total_time = time.time() - start_time
    avg_batch_time = processing_stats['processing_time'] / total_batches if total_batches > 0 else 0
    cache_hit_rate = (processing_stats['cache_hits'] / total_valid * 100) if total_valid > 0 else 0
    
    logger.info(f"{desc} 完成: 总耗时 {total_time:.2f}秒")
    logger.info(f"{desc} 统计: 缓存命中率 {cache_hit_rate:.1f}%, 平均批次耗时 {avg_batch_time:.3f}秒")
    logger.info(f"{desc} 统计: 重试次数 {processing_stats['retry_count']}, 内存清理 {processing_stats['memory_cleanups']}次")
    logger.info(f"{desc} 统计: 批次大小调整 {processing_stats['batch_size_adjustments']}次")
    logger.info(f"{desc} 统计: 未命中 {total_uncached}/{total_valid} ({total_uncached/total_valid*100:.1f}%)")
    
    return vectors, valid_texts

# 相似度计算和处理
def process_similarity(df_input, input_file, tokenizer, model):
    try:
        global keyword_cache, search_cache
        start_time = time.time()
        
        # 数据预处理
        df_input['clean_keyword_name'] = df_input['keyword_name'].apply(clean_text)
        df_input['clean_search_word'] = df_input['search_word'].apply(clean_text)
        
        # 记录无效记录
        invalid_mask = df_input[['clean_keyword_name', 'clean_search_word']].apply(
            lambda x: len(x['clean_keyword_name']) == 0 or len(x['clean_search_word']) == 0, axis=1)
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            logger.info(f"检测到 {invalid_count} 条无效记录")
        
        # 获取唯一关键词和搜索词
        unique_keywords = df_input['clean_keyword_name'].dropna().unique().tolist()
        unique_searches = df_input['clean_search_word'].dropna().unique().tolist()
        
        # 向量化阶段
        logger.info(f"开始向量化: {len(unique_keywords)} 关键词, {len(unique_searches)} 搜索词")
        
        # CPU模式顺序处理向量
        keyword_vectors, valid_keywords = batch_embedding(unique_keywords, tokenizer, model, keyword_cache, desc="处理关键词向量")
        search_vectors, valid_searches = batch_embedding(unique_searches, tokenizer, model, search_cache, desc="处理搜索词向量")
        
        # 创建向量字典
        keyword_vectors_dict = {kw: vec for kw, vec in zip(valid_keywords, keyword_vectors) if vec is not None}
        search_vectors_dict = {sw: vec for sw, vec in zip(valid_searches, search_vectors) if vec is not None}
        
        logger.info(f"关键词向量字典大小: {len(keyword_vectors_dict)}, 搜索词向量字典大小: {len(search_vectors_dict)}")
        
        # 计算相似度
        logger.info("开始计算相似度")
        similarity_start = time.time()
        def calculate_similarity(row):
            k_text = row['clean_keyword_name']
            s_text = row['clean_search_word']
            k_vec = keyword_vectors_dict.get(k_text)
            s_vec = search_vectors_dict.get(s_text)
            if k_vec is None or s_vec is None:
                return None
            try:
                similarity = cosine_similarity(k_vec.unsqueeze(0), s_vec.unsqueeze(0)).item()
                return similarity
            except Exception as e:
                logger.warning(f"相似度计算失败: 关键词='{k_text}', 搜索词='{s_text}', 错误={str(e)}")
                return None
        
        # 使用tqdm包装apply操作
        from tqdm import tqdm
        tqdm.pandas(desc="计算相似度")
        df_input['similarity'] = df_input.progress_apply(calculate_similarity, axis=1)
        logger.info(f"相似度计算完成, 耗时: {time.time() - similarity_start:.2f}秒")
        
        # 保存结果
        final_df_history = df_input[['keyword_name', 'search_word', 'similarity']]
        timestamp = datetime.strftime(datetime.now(), "%Y%m%d_%H%M")
        history_output_file = Config.HISTORY_DIR / f'cpu_result_{timestamp}.xlsx'
        final_df_history.to_excel(history_output_file, index=False)
        logger.info(f"历史格式结果已保存至: {history_output_file}")
        
        # 保存完整结果
        final_df_full = df_input.drop(columns=['clean_keyword_name', 'clean_search_word'], errors='ignore')
        input_filename = input_file.stem
        full_output_file = Config.OUTPUT_DIR / f'{input_filename}_cpu_with_similarity_{timestamp}.xlsx'
        final_df_full.to_excel(full_output_file, index=False)
        logger.info(f"完整格式结果已保存至: {full_output_file}")
        
        logger.info(f"总处理耗时: {time.time() - start_time:.2f}秒")
        
    except Exception as e:
        logger.error(f"相似度计算失败: {str(e)}")
        raise

# 主处理流程
def main(input_file=None):
    try:
        start_time = time.time()
        
        # 设置CPU优化
        Config.setup_cpu_optimization()
        Config.init_dirs()
        
        # 根据系统内存选择性能配置
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 8:
            set_cpu_performance_profile('ultra_safe')
        elif memory_gb < 16:
            set_cpu_performance_profile('memory_saving')
        else:
            set_cpu_performance_profile('balanced')
        
        logger.info(f"CPU模式启动，系统内存: {memory_gb:.1f}GB")
        
        # 初始化缓存
        global keyword_cache, search_cache
        keyword_cache = VectorCache('cpu_keyword_vectors.db')
        search_cache = VectorCache('cpu_search_vectors.db')
        
        # 加载模型
        try:
            global tokenizer, model
            tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
            model = AutoModel.from_pretrained(Config.MODEL_PATH)
            logger.info(f"成功加载模型: {Config.MODEL_PATH}")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
        
        # 加载输入数据
        if input_file is not None:
            input_file = Path(input_file)
            df_input = pd.read_excel(input_file, engine='openpyxl')
            logger.info(f"加载输入文件: {input_file.name}, 记录数: {len(df_input)}, 字段数: {len(df_input.columns)}")
        else:
            input_files = list(Config.INPUT_DIR.glob('*.xlsx'))
            if not input_files:
                logger.warning(f"未找到输入文件于: {Config.INPUT_DIR}")
                return
            input_file = input_files[0]
            df_input = pd.read_excel(input_file, engine='openpyxl')
            logger.info(f"加载输入文件: {input_file.name}, 记录数: {len(df_input)}, 字段数: {len(df_input.columns)}")
        
        # 字段映射
        if '关键词/营销要点/知识问答' in df_input.columns and '搜索词' in df_input.columns:
            df_input = df_input.rename(columns={
                '关键词/营销要点/知识问答': 'keyword_name',
                '搜索词': 'search_word'
            })
        else:
            logger.error(f"输入文件缺少必要字段: '关键词/营销要点/知识问答' 和 '搜索词'")
            raise ValueError("输入文件缺少必要字段: '关键词/营销要点/知识问答' 和 '搜索词'")
        
        # 处理相似度
        process_similarity(df_input, input_file, tokenizer, model)
        logger.info(f"处理完成, 总耗时: {time.time() - start_time:.2f}秒")
        
    except Exception as e:
        logger.error(f"运行时错误: {str(e)}")
        raise
    finally:
        if 'keyword_cache' in globals():
            keyword_cache.close()
        if 'search_cache' in globals():
            search_cache.close()
        logger.info("程序运行结束")

if __name__ == '__main__':
    import sys
    tqdm.pandas()
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main() 