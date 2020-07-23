class Logger:
    OK = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[31m'
    END = '\033[0m'

    @staticmethod
    def info(info):
        try:
            print(Logger.OK + info + Logger.END)
        except UnicodeEncodeError:
            Logger.warn("[ERROR] Not found chinese font, you must install chinese font")

    @staticmethod
    def warn(info):
        try:
            print(Logger.WARNING + info + Logger.END)
        except UnicodeEncodeError:
            Logger.warn("[ERROR] Not found chinese font, you must install chinese font, if not")

    @staticmethod
    def error(info):
        try:
            print(Logger.FAIL + info + Logger.END)
        except UnicodeEncodeError:
            Logger.warn("[ERROR] Not found chinese font, you must install chinese font, if not")