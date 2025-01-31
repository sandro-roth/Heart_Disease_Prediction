import logging

class MakeLogger:
    '''creating logger instance'''

    def costum_log(self, loglevel = logging.DEBUG, filename="log.log", modification="w"):
        logger=logging.getLogger(filename)
        logger.setLevel(loglevel)
        handler = logging.FileHandler(filename, mode=modification)
        formatter = logging.Formatter("%(asctime)s -%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger