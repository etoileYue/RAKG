import time
import logging
from functools import wraps
from logger import get_logger
import traceback

logger = get_logger(name="AgentLog",
                    level=logging.INFO,
                    log_file="Agent.log")

def retry(max_retries=3, delay=1):
    """
    重试装饰器，报错信息包含函数名、参数和完整堆栈
    :param max_retries: 最大重试次数
    :param delay: 重试间隔（秒）
    """
    def decorator(func):
        @wraps(func)  # 保留原函数的元信息（如函数名）
        def wrapper(*args, **kwargs):
            # 遍历重试次数
            for attempt in range(max_retries):
                try:
                    # 执行原函数并返回结果
                    return func(*args, **kwargs)
                except Exception as e:
                    # 获取函数名（保留原函数名，不受装饰器影响）
                    func_name = func.__name__
                    # 格式化参数信息：位置参数 + 关键字参数
                    # 处理位置参数（args）：转换为字符串，避免打印复杂对象时过长
                    args_str = ", ".join([str(arg) for arg in args])
                    # 处理关键字参数（kwargs）：key=value 格式
                    kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                    # 拼接完整参数字符串
                    params_str = ""
                    if args_str:
                        params_str += args_str
                    if kwargs_str:
                        if params_str:
                            params_str += ", "
                        params_str += kwargs_str
                    
                    # 判断是否是最后一次重试
                    if attempt == max_retries - 1:
                        # 最后一次失败：打印完整信息（函数名+参数+异常类型+堆栈）
                        error_msg = (
                            f"\n=== 重试{max_retries}次后仍失败 ==="
                            f"\n函数名：{func_name}"
                            f"\n调用参数：({params_str})"
                            f"\n异常类型：{type(e).__name__}"
                            f"\n异常内容：{str(e)}"
                            f"\n完整堆栈：\n{traceback.format_exc()}"
                        )
                        logger.error(error_msg)
                        raise  # 抛出异常，让上层处理
                    else:
                        # 非最后一次失败：打印简化的重试信息
                        retry_msg = (
                            f"【重试提示】第{attempt + 1}次调用失败 - "
                            f"函数：{func_name}，参数：({params_str})，"
                            f"异常：{type(e).__name__} - {str(e)}，"
                            f"{delay}秒后进行第{attempt + 2}次重试..."
                        )
                        logger.warning(retry_msg)
                        time.sleep(delay)
        return wrapper
    return decorator

import json
def get_ner_result_from_file(file_path, sent_to_id):
    def rewrite(self, ner_result, entity_num):
        new_entities = {}
        # Process in original dictionary key order, extract numbers after entity and renumber
        for idx, (old_key, value) in enumerate(ner_result.items(), start=1):
            new_key = f"entity{entity_num + idx - 1}"
            new_entities[new_key] = value
        return new_entities
    
    def add_chunkid(self, ner_result, chunkid):
        new_ner_result = {}
        for entity_key, entity_value in ner_result.items():
            entity_value["chunkid"] = chunkid
            new_ner_result[entity_key] = entity_value
        return new_ner_result
    
    ner_result_for_all = {}
    entity_num = 1

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            json_obj = json.loads(line)
            text = json_obj.get('text')
            ner_result = json_obj.get('entities')
            
            if 'State' in ner_result:
                continue
            # Get the number of entities in ner_result
            ner_result_num = len(ner_result)
            # Rewrite ner_result, entity numbering starts from entity_num, first entity is entity{entity_num}, subsequent entities increment
            ner_result = rewrite(ner_result, entity_num)

            entity_num += ner_result_num
            ner_result_with_chunkid = add_chunkid(ner_result,sent_to_id[text])
            ner_result_for_all.update(ner_result_with_chunkid)
                

    return ner_result_for_all