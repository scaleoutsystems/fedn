from gunicorn.app.base import BaseApplication
import os
class GunicornApp(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items()
                  if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

def run_gunicorn(app, workers=4):
    workers=os.cpu_count()
    options = {
        "bind": "127.0.0.1:8000",  # Specify the bind address and port here
        "workers": workers,
    }
    GunicornApp(app, options).run()
if __name__ == "main":
    run_gunicorn()
