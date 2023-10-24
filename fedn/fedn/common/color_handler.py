import logging

from termcolor import colored


class ColorizingStreamHandler(logging.StreamHandler):
    dark_theme = {
        'DEBUG': 'white',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red',
    }

    light_theme = {
        'DEBUG': 'black',
        'INFO': 'blue',
        'WARNING': 'magenta',
        'ERROR': 'red',
        'CRITICAL': 'red',
    }

    def __init__(self, theme='dark'):
        super().__init__()
        self.set_theme(theme)

    def set_theme(self, theme):
        if theme == 'dark':
            self.color_map = self.dark_theme
        elif theme == 'light':
            self.color_map = self.light_theme
        elif theme == 'default':
            self.color_map = {}  # No color applied
        else:
            self.color_map = {}  # No color applied

    def emit(self, record):
        try:
            # Separate the log level from the message
            level = '[{}]'.format(record.levelname)
            color = self.color_map.get(record.levelname, 'white')
            colored_level = colored(level, color)

            # Combine the colored log level with the rest of the message
            message = self.format(record).replace(level, colored_level)
            self.stream.write(message + "\n")
            self.flush()
        except Exception:
            self.handleError(record)
