from gunicorn.app.base import BaseApplication

from fedn.network.controller.control import Control


class GunicornApp(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items() if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


def run_gunicorn(app, host, port, workers=4):
    bind_address = f"{host}:{port}"
    options = {
        "bind": bind_address,  # Specify the bind address and port here
        "workers": workers,
        # After forking, initialize the database connection
        # "post_fork": lambda: Control.instance().db.initialize_connection(),
    }
    GunicornApp(app, options).run()
