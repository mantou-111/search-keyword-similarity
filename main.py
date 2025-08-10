
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

# 配置类
class Config:
    MODEL_PATH = Path('D:/model/BAAI-bge-large-zh-v1.5')  # 模型路径
    INPUT_DIR = Path('./input')  # 输入目录
    OUTPUT_DIR = Path('./output')  # 输出目录
    HISTORY_DIR = OUTPUT_DIR / 'history'  # 历史文件目录
    CACHE_DIR = Path('./cache')  # 缓存目录
    MAX_LENGTH = 64  # 最大文本长度（降低以减少内存）
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 计算设备
    
    # 性能优化参数 - 大幅降低内存使用
    BATCH_SIZE = 32  # 向量化批处理大小（大幅降低）
    MEMORY_CACHE_SIZE = 50000  # 内存缓存大小（大幅降低）
    QUEUE_LIMIT = 10000  # 异步写入队列大小（降低）
    MAX_CACHE_VECTORS = 200000  # SQLite缓存最大向量数（降低）
    TEXT_BATCH_SIZE = 5000  # 文本分批大小（大幅降低）
    USE_PARQUET = False  # 是否使用Parquet格式保存（需安装 pyarrow）
    
    # 内存管理配置 - 更频繁清理
    MEMORY_CLEANUP_INTERVAL = 1  # 每处理1个批次清理一次内存
    GPU_MEMORY_THRESHOLD = 0.4  # GPU内存使用率阈值，超过此值强制清理（降低阈值）
    ENABLE_MEMORY_MONITORING = True  # 是否启用内存监控
    
    # 新增：动态批次调整 - 更保守
    ENABLE_DYNAMIC_BATCHING = True  # 启用动态批次大小调整
    MIN_BATCH_SIZE = 8  # 最小批次大小（降低）
    MAX_BATCH_SIZE = 64  # 最大批次大小（大幅降低）
    BATCH_ADJUSTMENT_FACTOR = 0.5  # 批次大小调整因子（更激进）
    
    # 并行处理配置 - 禁用并行处理
    ENABLE_PARALLEL_PROCESSING = False  # 禁用并行处理
    PARALLEL_WORKERS = 1  # 并行工作线程数
    PARALLEL_CHUNK_SIZE = 5000  # 并行处理块大小（降低）
    
    # 新增：内存保护机制
    FORCE_CPU_FALLBACK = False  # 是否强制使用CPU（内存不足时）
    MEMORY_CHECK_INTERVAL = 1  # 每处理多少个批次检查一次内存
    EMERGENCY_MEMORY_THRESHOLD = 0.8  # 紧急内存阈值，超过此值强制清理
    
    # 预设性能配置方案
    @classmethod
    def set_performance_profile(cls, profile='ultra_safe'):
        """设置性能配置方案"""
        if profile == 'ultra_safe':
            # 超安全模式 - 避免系统崩溃
            cls.BATCH_SIZE = 16
            cls.TEXT_BATCH_SIZE = 2000
            cls.MEMORY_CACHE_SIZE = 20000
            cls.MEMORY_CLEANUP_INTERVAL = 1
            cls.ENABLE_DYNAMIC_BATCHING = True
            cls.MIN_BATCH_SIZE = 4
            cls.MAX_BATCH_SIZE = 32
            cls.GPU_MEMORY_THRESHOLD = 0.3
            cls.EMERGENCY_MEMORY_THRESHOLD = 0.6
            logger.info("已设置为超安全模式")
            
        elif profile == 'memory_saving':
            # 内存节省模式 - 适合小内存GPU
            cls.BATCH_SIZE = 32
            cls.TEXT_BATCH_SIZE = 5000
            cls.MEMORY_CACHE_SIZE = 50000
            cls.MEMORY_CLEANUP_INTERVAL = 1
            cls.ENABLE_DYNAMIC_BATCHING = True
            cls.MIN_BATCH_SIZE = 8
            cls.MAX_BATCH_SIZE = 64
            cls.GPU_MEMORY_THRESHOLD = 0.4
            cls.EMERGENCY_MEMORY_THRESHOLD = 0.7
            logger.info("已设置为内存节省模式")
            
        elif profile == 'balanced':
            # 平衡模式 - 默认配置
            cls.BATCH_SIZE = 64
            cls.TEXT_BATCH_SIZE = 10000
            cls.MEMORY_CACHE_SIZE = 80000
            cls.MEMORY_CLEANUP_INTERVAL = 2
            cls.ENABLE_DYNAMIC_BATCHING = True
            cls.MIN_BATCH_SIZE = 16
            cls.MAX_BATCH_SIZE = 128
            cls.GPU_MEMORY_THRESHOLD = 0.5
            cls.EMERGENCY_MEMORY_THRESHOLD = 0.8
            logger.info("已设置为平衡模式")
            
        elif profile == 'speed_optimized':
            # 速度优化模式 - 适合大内存GPU
            cls.BATCH_SIZE = 128
            cls.TEXT_BATCH_SIZE = 15000
            cls.MEMORY_CACHE_SIZE = 120000
            cls.MEMORY_CLEANUP_INTERVAL = 3
            cls.ENABLE_DYNAMIC_BATCHING = True
            cls.MIN_BATCH_SIZE = 32
            cls.MAX_BATCH_SIZE = 256
            cls.ENABLE_PARALLEL_PROCESSING = False
            cls.GPU_MEMORY_THRESHOLD = 0.6
            cls.EMERGENCY_MEMORY_THRESHOLD = 0.85
            logger.info("已设置为速度优化模式")

    @classmethod
    def init_dirs(cls):
        # 初始化目录
        for d in [cls.INPUT_DIR, cls.OUTPUT_DIR, cls.HISTORY_DIR, cls.CACHE_DIR]:
            d.mkdir(parents=True, exist_ok=True)

# 初始化日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            Config.OUTPUT_DIR / 'simplified_similarity_analysis.log',
            maxBytes=10*1024*1024,
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 向量缓存类
class VectorCache:
    def __init__(self, cache_file):
        # 初始化缓存
        self.cache_file = Config.CACHE_DIR / cache_file
        self.memory_cache = {}
        self.conn = sqlite3.connect(str(self.cache_file), check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.write_failures = []
        self.lock = threading.Lock()  # 添加锁以同步数据库访问
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
                batch.append((text, vector.cpu().numpy().tobytes(), vector.size(0)))
                if len(batch) >= 2000 or time.time() - last_write > 20:  # 优化：2000条或20秒
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
                vector = torch.from_numpy(np.frombuffer(result[0], dtype=np.float32).copy()).to(Config.DEVICE)
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

        # SQLite缓存查询 - 优化批量查询
        uncached = [text for text in valid_texts if text not in self.memory_cache]
        if uncached:
            # 增加批量查询大小以提高效率
            batch_size = 1000  # 增加批量查询大小
            for i in range(0, len(uncached), batch_size):
                batch_uncached = uncached[i:i+batch_size]
                placeholders = ','.join('?' * len(batch_uncached))
                try:
                    with self.lock:
                        conn = sqlite3.connect(str(self.cache_file), check_same_thread=False)
                        cursor = conn.cursor()
                        # 使用PRAGMA优化查询性能
                        cursor.execute('PRAGMA temp_store = MEMORY')
                        cursor.execute('PRAGMA cache_size = 10000')
                        cursor.execute(f'SELECT text, vector, dimension FROM vectors WHERE text IN ({placeholders})', batch_uncached)
                        
                        # 批量处理结果
                        results = cursor.fetchall()
                        for text, vector, dim in results:
                            vector = torch.from_numpy(np.frombuffer(vector, dtype=np.float32).copy()).to(Config.DEVICE)
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
            if uncached:
                logger.debug(f"{desc} 未命中文本样例: {uncached[:5]}")
        
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
            self.memory_cache[text] = vector.to(Config.DEVICE)
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
                                  (text, vector.cpu().numpy().tobytes(), vector.size(0)))
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

# 内存监控工具函数
def monitor_memory_usage():
    """监控当前内存使用情况"""
    if Config.DEVICE == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        usage_ratio = allocated / total if total > 0 else 0
        logger.debug(f"GPU内存使用: {allocated:.2f}GB / {cached:.2f}GB / {total:.1f}GB (已分配/已缓存/总计), 使用率: {usage_ratio:.1%}")
        return allocated, cached, total, usage_ratio
    return None, None, None, 0

def emergency_memory_cleanup():
    """紧急内存清理"""
    if Config.DEVICE == 'cuda':
        logger.warning("执行紧急内存清理...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated, cached, total, usage_ratio = monitor_memory_usage()
        if usage_ratio > Config.EMERGENCY_MEMORY_THRESHOLD:
            logger.error(f"GPU内存使用率过高: {usage_ratio:.1%}, 建议重启程序或使用CPU模式")
            return False
        logger.info(f"紧急内存清理完成，当前使用率: {usage_ratio:.1%}")
        return True
    return True

def check_memory_safety():
    """检查内存安全性"""
    if Config.DEVICE == 'cuda':
        allocated, cached, total, usage_ratio = monitor_memory_usage()
        if usage_ratio > Config.EMERGENCY_MEMORY_THRESHOLD:
            logger.error(f"内存使用率过高: {usage_ratio:.1%}, 超过紧急阈值 {Config.EMERGENCY_MEMORY_THRESHOLD:.1%}")
            return False
        elif usage_ratio > Config.GPU_MEMORY_THRESHOLD:
            logger.warning(f"内存使用率较高: {usage_ratio:.1%}, 超过阈值 {Config.GPU_MEMORY_THRESHOLD:.1%}")
            return True
        return True
    return True

def get_memory_optimization_suggestions(gpu_memory_gb, data_size):
    """根据硬件配置和数据量提供优化建议"""
    suggestions = []
    
    if gpu_memory_gb < 6:
        suggestions.append("GPU内存较小，建议:")
        suggestions.append("- 将BATCH_SIZE设置为128或更小")
        suggestions.append("- 启用动态批次调整 (ENABLE_DYNAMIC_BATCHING = True)")
        suggestions.append("- 增加内存清理频率 (MEMORY_CLEANUP_INTERVAL = 3)")
    elif gpu_memory_gb < 12:
        suggestions.append("GPU内存中等，建议:")
        suggestions.append("- 将BATCH_SIZE设置为256-384")
        suggestions.append("- 启用动态批次调整")
    else:
        suggestions.append("GPU内存充足，建议:")
        suggestions.append("- 将BATCH_SIZE设置为512或更大")
        suggestions.append("- 可考虑启用并行处理")
    
    if data_size > 500000:
        suggestions.append("数据量很大，建议:")
        suggestions.append("- 将TEXT_BATCH_SIZE设置为15000-20000")
        suggestions.append("- 增加MEMORY_CACHE_SIZE到300000")
        suggestions.append("- 考虑分批处理数据")
    
    return suggestions

# 文本清理函数
def clean_text(text):
    # 清理文本，保留有效字符
    if pd.isna(text) or text is None or str(text).strip().lower() in ['nan', 'null', '']:
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。？！：；“”‘’、·…—\-—～《》]', '', text)
    return text[:250]

# 批量向量化
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
    
    # 添加超时保护
    last_progress_time = time.time()
    progress_timeout = 300  # 5分钟超时

    logger.info(f"{desc} 开始处理 {len(texts)} 个文本，分为 {total_batches} 个批次")

    for batch_idx in tqdm(range(0, len(texts), text_batch_size), desc=desc, total=total_batches):
        batch_start_time = time.time()
        batch_texts = texts[batch_idx:batch_idx + text_batch_size]
        
        # 检查超时
        if time.time() - last_progress_time > progress_timeout:
            logger.warning(f"{desc} 批次处理超时，可能卡住，尝试继续...")
            last_progress_time = time.time()
        
        # 日志记录文本长度
        if batch_texts:
            valid_batch = [t for t in batch_texts if isinstance(t, str) and t.strip()]
            if valid_batch:
                logger.debug(f"{desc} 批次 {batch_idx//text_batch_size + 1} 文本长度: min={min(len(t) for t in valid_batch)}, max={max(len(t) for t in valid_batch)}")
        
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
                                          max_length=Config.MAX_LENGTH).to(Config.DEVICE)
                        with torch.no_grad():
                            outputs = model(**inputs).last_hidden_state[:, 0, :].to(torch.float32)
                        
                        for text, vector in zip(sub_batch_texts, outputs):
                            cache.add(text, vector)
                            batch_vectors.append(vector)
                            batch_valid_texts.append(text)
                        
                        # 显式释放张量
                        del inputs, outputs
                        torch.cuda.empty_cache()
                        success = True
                        consecutive_successes += 1
                        consecutive_oom_errors = 0
                        last_progress_time = time.time()
                        
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
                    except torch.cuda.OutOfMemoryError:
                        processing_stats['retry_count'] += 1
                        consecutive_oom_errors += 1
                        consecutive_successes = 0
                        
                        # 动态调整批次大小：OOM时减少
                        if Config.ENABLE_DYNAMIC_BATCHING:
                            new_batch_size = max(int(current_batch_size * Config.BATCH_ADJUSTMENT_FACTOR), Config.MIN_BATCH_SIZE)
                            if new_batch_size != current_batch_size:
                                logger.info(f"{desc} GPU内存不足，批次大小调整: {current_batch_size} -> {new_batch_size}")
                                current_batch_size = new_batch_size
                                processing_stats['batch_size_adjustments'] += 1
                        
                        logger.warning(f"{desc} GPU内存不足，尝试减小批次大小 (重试 {retry + 1}/{max_retries})")
                        torch.cuda.empty_cache()
                        if retry == max_retries - 1:
                            # 最后一次重试失败，使用更小的批次
                            smaller_batch_size = max(current_batch_size // 2, Config.MIN_BATCH_SIZE)
                            logger.info(f"使用更小批次大小: {smaller_batch_size}")
                            for j in range(0, len(sub_batch_texts), smaller_batch_size):
                                mini_batch = sub_batch_texts[j:j+smaller_batch_size]
                                try:
                                    inputs = tokenizer(mini_batch, return_tensors='pt', truncation=True, padding=True,
                                                      max_length=Config.MAX_LENGTH).to(Config.DEVICE)
                                    with torch.no_grad():
                                        outputs = model(**inputs).last_hidden_state[:, 0, :].to(torch.float32)
                                    for text, vector in zip(mini_batch, outputs):
                                        cache.add(text, vector)
                                        batch_vectors.append(vector)
                                        batch_valid_texts.append(text)
                                    del inputs, outputs
                                    torch.cuda.empty_cache()
                                except Exception as e:
                                    logger.error(f"小批次处理失败: {str(e)}")
                                    batch_vectors.extend([None] * len(mini_batch))
                                    batch_valid_texts.extend(mini_batch)
                                    error_count += len(mini_batch)
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
        
        # 优化：使用字典映射提高合并效率
        text_to_global_index = {text: batch_idx + i for i, text in enumerate(batch_texts)}
        
        # 合并批次结果
        for i, vec in enumerate(cached_vectors):
            if vec is not None:
                global_idx = batch_idx + i
                vectors[global_idx] = vec
                valid_texts[global_idx] = batch_texts[i]
        
        # 合并新生成的向量
        for text, vector in zip(batch_valid_texts, batch_vectors):
            if text in text_to_global_index:
                global_idx = text_to_global_index[text]
                vectors[global_idx] = vector
                valid_texts[global_idx] = text
        
        # 定期内存清理和检查
        if Config.ENABLE_MEMORY_MONITORING and (batch_idx // text_batch_size) % Config.MEMORY_CLEANUP_INTERVAL == 0:
            if Config.DEVICE == 'cuda':
                # 检查内存安全性
                if not check_memory_safety():
                    logger.error("内存使用率过高，执行紧急清理")
                    if not emergency_memory_cleanup():
                        logger.error("紧急内存清理失败，建议重启程序")
                        break
                
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                processing_stats['memory_cleanups'] += 1
                # 监控内存使用情况
                monitor_memory_usage()
        
        processing_stats['processing_time'] += time.time() - batch_start_time
        
        # 每10个批次输出一次进度
        if (batch_idx // text_batch_size + 1) % 10 == 0:
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
            invalid_samples = df_input[invalid_mask][['keyword_name', 'search_word']].head()
            logger.debug(f"无效记录样例:\n{invalid_samples.to_string()}")
        
        # 获取唯一关键词和搜索词
        unique_keywords = df_input['clean_keyword_name'].dropna().unique().tolist()
        unique_searches = df_input['clean_search_word'].dropna().unique().tolist()
        
        # 检查文本清理后是否会导致重复
        if len(unique_keywords) < len(df_input['clean_keyword_name'].dropna()):
            logger.warning(f"关键词清理后存在重复: 原始 {len(df_input['clean_keyword_name'].dropna())} 条, 唯一 {len(unique_keywords)} 条")
        if len(unique_searches) < len(df_input['clean_search_word'].dropna()):
            logger.warning(f"搜索词清理后存在重复: 原始 {len(df_input['clean_search_word'].dropna())} 条, 唯一 {len(unique_searches)} 条")
        
        # 向量化阶段
        logger.info(f"开始向量化: {len(unique_keywords)} 关键词, {len(unique_searches)} 搜索词")
        
        if Config.ENABLE_PARALLEL_PROCESSING:
            logger.info(f"启用并行处理 (工作线程: {Config.PARALLEL_WORKERS})")
            # 使用线程池进行并行向量化
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=Config.PARALLEL_WORKERS) as executor:
                # 将关键词和搜索词分成块
                keyword_chunks = [unique_keywords[i:i + Config.PARALLEL_CHUNK_SIZE] for i in range(0, len(unique_keywords), Config.PARALLEL_CHUNK_SIZE)]
                search_chunks = [unique_searches[i:i + Config.PARALLEL_CHUNK_SIZE] for i in range(0, len(unique_searches), Config.PARALLEL_CHUNK_SIZE)]
                
                # 并行处理关键词向量
                keyword_results = list(executor.map(
                    lambda chunk: batch_embedding(chunk, tokenizer, model, keyword_cache, desc="处理关键词向量"),
                    keyword_chunks
                ))
                keyword_vectors = [vec for result in keyword_results for vec in result[0]]
                valid_keywords = [text for result in keyword_results for text in result[1]]
                
                # 并行处理搜索词向量
                search_results = list(executor.map(
                    lambda chunk: batch_embedding(chunk, tokenizer, model, search_cache, desc="处理搜索词向量"),
                    search_chunks
                ))
                search_vectors = [vec for result in search_results for vec in result[0]]
                valid_searches = [text for result in search_results for text in result[1]]
        else:
            # 顺序处理向量
            keyword_vectors, valid_keywords = batch_embedding(unique_keywords, tokenizer, model, keyword_cache, desc="处理关键词向量")
            search_vectors, valid_searches = batch_embedding(unique_searches, tokenizer, model, search_cache, desc="处理搜索词向量")
        
        # 创建向量字典，确保一一对应
        keyword_vectors_dict = {kw: vec for kw, vec in zip(valid_keywords, keyword_vectors) if vec is not None}
        search_vectors_dict = {sw: vec for sw, vec in zip(valid_searches, search_vectors) if vec is not None}
        
        # 检查向量字典大小
        logger.info(f"关键词向量字典大小: {len(keyword_vectors_dict)}, 搜索词向量字典大小: {len(search_vectors_dict)}")
        if len(keyword_vectors_dict) < len(unique_keywords) or len(search_vectors_dict) < len(unique_searches):
            logger.warning(f"向量缺失: 关键词 {len(unique_keywords) - len(keyword_vectors_dict)}/{len(unique_keywords)}, 搜索词 {len(unique_searches) - len(search_vectors_dict)}/{len(unique_searches)}")
        
        # 计算相似度
        logger.info("开始计算相似度")
        similarity_start = time.time()
        def calculate_similarity(row):
            k_text = row['clean_keyword_name']
            s_text = row['clean_search_word']
            k_vec = keyword_vectors_dict.get(k_text)
            s_vec = search_vectors_dict.get(s_text)
            if k_vec is None or s_vec is None:
                logger.debug(f"向量缺失: 关键词='{k_text}', 搜索词='{s_text}'")
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
        
        # 保存结果：历史格式文件
        final_df_history = df_input[['keyword_name', 'search_word', 'similarity']]
        timestamp = datetime.strftime(datetime.now(), "%Y%m%d_%H%M")
        history_output_file = Config.HISTORY_DIR / f'result_{timestamp}.{"parquet" if Config.USE_PARQUET else "xlsx"}'
        if Config.USE_PARQUET:
            final_df_history.to_parquet(history_output_file, index=False)
        else:
            final_df_history.to_excel(history_output_file, index=False)
        logger.info(f"历史格式结果已保存至: {history_output_file}")
        
        # 保存结果：新格式文件，保留原始列+similarity
        final_df_full = df_input.drop(columns=['clean_keyword_name', 'clean_search_word'], errors='ignore')
        input_filename = input_file.stem
        full_output_file = Config.OUTPUT_DIR / f'{input_filename}_with_similarity_{timestamp}.{"parquet" if Config.USE_PARQUET else "xlsx"}'
        if Config.USE_PARQUET:
            final_df_full.to_parquet(full_output_file, index=False)
        else:
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
        Config.init_dirs()
        logger.info(f"使用设备: {Config.DEVICE}")
        
        # 内存监控和配置
        if Config.DEVICE == 'cuda':
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU总内存: {gpu_memory:.1f}GB")
            
            # 检查GPU内存是否足够
            if gpu_memory < 4:
                logger.warning(f"GPU内存较小 ({gpu_memory:.1f}GB)，建议使用CPU模式或增加虚拟内存")
                Config.FORCE_CPU_FALLBACK = True
                Config.DEVICE = 'cpu'
                logger.info("已切换到CPU模式")
            else:
                # 自动选择性能配置
                if gpu_memory < 6:
                    Config.set_performance_profile('ultra_safe')
                elif gpu_memory < 12:
                    Config.set_performance_profile('memory_saving')
                elif gpu_memory < 24:
                    Config.set_performance_profile('balanced')
                else:
                    Config.set_performance_profile('speed_optimized')
                
                # 初始内存检查
                if not check_memory_safety():
                    logger.error("初始内存检查失败，建议重启程序")
                    return
        # 初始化缓存
        global keyword_cache, search_cache
        keyword_cache = VectorCache('keyword_vectors.db')
        search_cache = VectorCache('search_vectors.db')
        # 加载模型
        try:
            global tokenizer, model
            tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
            model = AutoModel.from_pretrained(Config.MODEL_PATH).to(Config.DEVICE)
            logger.info(f"成功加载模型: {Config.MODEL_PATH}")
            if Config.DEVICE == 'cuda':
                model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
                logger.info(f"模型内存占用: {model_memory:.2f}GB")
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
        # 获取优化建议（在加载数据后）
        if Config.DEVICE == 'cuda':
            suggestions = get_memory_optimization_suggestions(gpu_memory, len(df_input))
            if suggestions:
                logger.info("性能优化建议:")
                for suggestion in suggestions:
                    logger.info(suggestion)
        if len(df_input) > 100000:
            logger.info("检测到大数据量，性能优化建议:")
            logger.info("1. 已启用动态批次大小调整")
            logger.info("2. 已优化缓存查询性能")
            logger.info("3. 如需更快处理，可考虑启用并行处理 (设置 ENABLE_PARALLEL_PROCESSING = True)")
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