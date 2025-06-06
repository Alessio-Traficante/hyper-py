import logging
import warnings
import sys
from astropy.wcs import FITSFixedWarning
from astropy.utils.exceptions import AstropyUserWarning

class StreamToLogger:
    def __init__(self, logger, level=logging.WARNING):
        self.logger = logger
        self.level = level
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())
    def flush(self):
        pass

class ProcessNameFilter(logging.Filter):
    def __init__(self, process_name):
        super().__init__()
        self.process_name = process_name

    def filter(self, record):
        record.process_name = self.process_name
        return True

def setup_logger(log_path="hyper.log", logger_name="HyperLogger", overwrite=True, process_name = ""):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    mode = "w" if overwrite else "a"
    file_handler = logging.FileHandler(log_path, mode=mode, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    # Console handler (only INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    logger.addFilter(lambda record: setattr(record, "process_name_suffix", f" ({process_name})" if process_name else "") or True)
    console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s%(process_name_suffix)s'))
    logger.addFilter(ProcessNameFilter(process_name))
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
        
    # File-only logger for warnings
    logger_file_only = logging.getLogger(f"{logger_name}FileOnly")
    logger_file_only.setLevel(logging.INFO)
    logger_file_only.propagate = False

    if logger_file_only.hasHandlers():
        logger_file_only.handlers.clear()
        
    logger_file_only.addHandler(file_handler)

    # Redirect warnings
    sys.stderr = StreamToLogger(logger_file_only, level=logging.WARNING)
    sys.__stderr__ = StreamToLogger(logger_file_only, level=logging.WARNING)

    def custom_showwarning(message, category, filename, lineno, file=None, line=None):
        logger_file_only.warning(f"{category.__name__}: {message} (from {filename}:{lineno})")

    warnings.showwarning = custom_showwarning
    warnings.simplefilter("always")
    warnings.filterwarnings("always", category=FITSFixedWarning)
    warnings.filterwarnings("always", category=AstropyUserWarning)
    warnings.filterwarnings("always", category=ResourceWarning)
    warnings.filterwarnings("ignore", category=FITSFixedWarning, module="astropy")
    warnings.filterwarnings("ignore", category=UserWarning, module="uncertainties")
    
    return logger, logger_file_only