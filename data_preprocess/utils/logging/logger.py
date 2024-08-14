import logging
from colorama import Fore, Style

class LOG:

    logger = logging.getLogger()

    console = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    LEVEL = logging.INFO
    INFO = logging.INFO
    WARN = logging.WARN
    WARNING = logging.WARNING
    FATAL = logging.FATAL
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    @staticmethod
    def addFileHandler(filename):
        handler = logging.FileHandler(filename)
        logging.getLogger().addHandler(handler)

    @staticmethod
    def setLevel(level):
        __class__.logger.setLevel(level)
        __class__.LEVEL = level

    @staticmethod
    def debug(*msg):
        logging.debug(Fore.MAGENTA+','.join([str(m) for m in msg])+Style.RESET_ALL)
    
    @staticmethod
    def info(*msg):
        logging.info(Fore.GREEN+','.join([str(m) for m in msg])+Style.RESET_ALL)
    @staticmethod

    def warning(*msg):
        logging.warning(Fore.YELLOW+','.join([str(m) for m in msg])+Style.RESET_ALL)
    
    @staticmethod
    def warn(*msg):
        logging.warn(Fore.YELLOW+','.join([str(m) for m in msg])+Style.RESET_ALL)

    @staticmethod
    def error(*msg):
        logging.error(Fore.RED+','.join([str(m) for m in msg])+Style.RESET_ALL)

    @staticmethod
    def critical(*msg):
        logging.critical(Fore.CYAN+','.join([str(m) for m in msg])+Style.RESET_ALL)

LOG.setLevel(LOG.LEVEL)