import logging
import inspect

class MakeLogger:
    '''creating logger instance'''

    def costum_log(self, loglevel = logging.DEBUG, filename="log.log", modification="w"):
        logger_name = inspect.stack()[1][3]
        logger=logging.getLogger(logger_name)
        logger.setLevel(loglevel)
        handler = logging.FileHandler(filename, mode=modification)
        formatter = logging.Formatter("%(asctime)s -%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger