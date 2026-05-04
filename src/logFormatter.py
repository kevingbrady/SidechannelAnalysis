import logging

class logFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    green = "\033[1;32m"  #"\x1b[32;20m"
    blue = "\033[1;34m"  #"\x1b[34;20m"
    cyan =  "\033[1;36m"   #"\x1b[36;20m"
    magenta = "\x1b[35;20m"
    purple =  "\033[1;35m"  #"\033[0;35m"
    yellow = "\x1b[33;20m"
    gold = "\x1b[38;5;220m"
    orange = "\033[38;2;255;165;0m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    white = "\x1b[37;20m"
    black = "\x1b[30;20m"
    reset = "\x1b[0m"
    #fmt = '%(asctime)s - %(levelname)s - %(message)s'
    fmt = "%(message)s" # Default for no formatting
    colors = ['grey', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'gold', 'orange', 'red', 'bold_red', 'white']

    def __init__(self, log_colors_dict):
        super(logging.Formatter, self).__init__()

        self.FORMATS = {
            logging.DEBUG: getattr(self, log_colors_dict['DEBUG']) + self.fmt + self.reset,
            logging.INFO: getattr(self, log_colors_dict['INFO']) + self.fmt + self.reset,
            logging.WARNING: getattr(self, log_colors_dict['WARNING']) + self.fmt + self.reset,
            logging.ERROR: getattr(self, log_colors_dict['ERROR']) + self.fmt + self.reset,
            logging.CRITICAL: getattr(self, log_colors_dict['CRITICAL']) + self.fmt + self.reset
        }

    def format(self, record):

        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
