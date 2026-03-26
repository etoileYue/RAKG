"""
日志模块 - 提供统一的日志记录功能
支持控制台输出、文件记录、日志轮转等功能
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class Logger:
    """日志管理类"""
    
    def __init__(
        self,
        name: str = "AppLogger",
        log_dir: str = "logs",
        log_file: str = "app.log",
        level: int = logging.INFO,
        console_output: bool = True,
        file_output: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        log_format: Optional[str] = None
    ):
        """
        初始化日志器
        
        Args:
            name: 日志器名称
            log_dir: 日志文件目录
            log_file: 日志文件名
            level: 日志级别
            console_output: 是否输出到控制台
            file_output: 是否输出到文件
            max_bytes: 单个日志文件最大字节数
            backup_count: 保留的日志文件数量
            log_format: 自定义日志格式
        """
        self.name = name
        self.log_dir = log_dir
        self.log_file = log_file
        self.level = level
        self.console_output = console_output
        self.file_output = file_output
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        # 默认日志格式
        if log_format is None:
            self.log_format = (
                '%(asctime)s - %(levelname)s - '
                '%(message)s'
            )
        else:
            self.log_format = log_format
        
        # 创建日志器
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """配置并返回日志器"""
        # 获取或创建日志器
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        
        # 避免重复添加处理器
        if logger.handlers:
            return logger
        
        # 创建格式化器
        formatter = logging.Formatter(
            self.log_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 添加控制台处理器
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 添加文件处理器
        if self.file_output:
            # 确保日志目录存在
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            
            # 创建轮转文件处理器
            log_path = os.path.join(self.log_dir, self.log_file)
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(self.level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def debug(self, message: str, *args, **kwargs):
        """记录 DEBUG 级别日志"""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """记录 INFO 级别日志"""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """记录 WARNING 级别日志"""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """记录 ERROR 级别日志"""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """记录 CRITICAL 级别日志"""
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """记录异常信息"""
        self.logger.exception(message, *args, **kwargs)
    
    def set_level(self, level: int):
        """动态设置日志级别"""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
    
    def get_logger(self) -> logging.Logger:
        """获取底层的 logger 对象"""
        return self.logger


class TimedRotatingLogger(Logger):
    """按时间轮转的日志器"""
    
    def __init__(
        self,
        name: str = "TimedLogger",
        log_dir: str = "logs",
        log_file: str = "app.log",
        level: int = logging.INFO,
        when: str = 'midnight',
        interval: int = 1,
        backup_count: int = 30,
        console_output: bool = True,
        log_format: Optional[str] = None
    ):
        """
        初始化按时间轮转的日志器
        
        Args:
            when: 轮转时间单位 ('S', 'M', 'H', 'D', 'midnight', 'W0'-'W6')
            interval: 轮转间隔
            backup_count: 保留的日志文件数量
        """
        self.when = when
        self.interval = interval
        self.timed_backup_count = backup_count
        
        # 调用父类初始化,但不启用文件输出(我们自己添加)
        super().__init__(
            name=name,
            log_dir=log_dir,
            log_file=log_file,
            level=level,
            console_output=console_output,
            file_output=False,
            log_format=log_format
        )
        
        # 添加时间轮转处理器
        self._add_timed_rotating_handler()
    
    def _add_timed_rotating_handler(self):
        """添加时间轮转文件处理器"""
        # 确保日志目录存在
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # 创建时间轮转文件处理器
        log_path = os.path.join(self.log_dir, self.log_file)
        handler = logging.handlers.TimedRotatingFileHandler(
            log_path,
            when=self.when,
            interval=self.interval,
            backupCount=self.timed_backup_count,
            encoding='utf-8'
        )
        
        # 设置日志文件名后缀
        handler.suffix = "%Y%m%d"
        
        formatter = logging.Formatter(
            self.log_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        handler.setLevel(self.level)
        
        self.logger.addHandler(handler)


# 便捷函数:创建默认日志器
def get_logger(
    name: str = "AppLogger",
    level: int = logging.INFO,
    log_dir: str = "logs",
    **kwargs
) -> Logger:
    """
    获取日志器实例的便捷函数
    
    Args:
        name: 日志器名称
        level: 日志级别
        log_dir: 日志目录
        **kwargs: 其他参数传递给 Logger
    
    Returns:
        Logger 实例
    """
    return Logger(name=name, level=level, log_dir=log_dir, **kwargs)


def get_timed_logger(
    name: str = "TimedLogger",
    level: int = logging.INFO,
    log_dir: str = "logs",
    when: str = 'midnight',
    **kwargs
) -> TimedRotatingLogger:
    """
    获取按时间轮转的日志器实例
    
    Args:
        name: 日志器名称
        level: 日志级别
        log_dir: 日志目录
        when: 轮转时间单位
        **kwargs: 其他参数传递给 TimedRotatingLogger
    
    Returns:
        TimedRotatingLogger 实例
    """
    return TimedRotatingLogger(name=name, level=level, log_dir=log_dir, when=when, **kwargs)


if __name__ == "__main__":
    # 示例1: 基本使用
    print("=" * 50)
    print("示例1: 基本日志记录")
    print("=" * 50)
    
    logger = get_logger(name="MyApp", level=logging.DEBUG)
    
    logger.debug("这是一条调试信息")
    logger.info("应用启动成功")
    logger.warning("这是一条警告信息")
    logger.error("发生了一个错误")
    logger.critical("严重错误!")
    
    # 示例2: 异常记录
    print("\n" + "=" * 50)
    print("示例2: 异常记录")
    print("=" * 50)
    
    try:
        result = 10 / 0
    except Exception as e:
        logger.exception("捕获到异常")
    
    # 示例3: 按时间轮转的日志器
    print("\n" + "=" * 50)
    print("示例3: 按时间轮转的日志器")
    print("=" * 50)
    
    timed_logger = get_timed_logger(
        name="TimedApp",
        when='midnight',  # 每天午夜轮转
        backup_count=7    # 保留7天的日志
    )
    
    timed_logger.info("这条日志会按时间轮转保存")
    
    # 示例4: 自定义配置
    print("\n" + "=" * 50)
    print("示例4: 自定义配置")
    print("=" * 50)
    
    custom_logger = Logger(
        name="CustomApp",
        log_dir="custom_logs",
        log_file="custom.log",
        level=logging.DEBUG,
        console_output=True,
        file_output=True,
        max_bytes=5 * 1024 * 1024,  # 5MB
        backup_count=3,
        log_format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    custom_logger.info("使用自定义格式的日志")
    
    # 示例5: 动态修改日志级别
    print("\n" + "=" * 50)
    print("示例5: 动态修改日志级别")
    print("=" * 50)
    
    logger.debug("这条 DEBUG 信息会显示")
    logger.set_level(logging.WARNING)
    logger.debug("这条 DEBUG 信息不会显示")
    logger.warning("但 WARNING 级别会显示")
    
    print("\n日志文件已保存到 'logs' 目录")
