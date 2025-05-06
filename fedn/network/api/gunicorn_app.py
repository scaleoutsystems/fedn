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


def post_fork(server, worker):
    """Hook to be called after the worker has forked.

    This is where we can initialize the database connection for each worker.
    """
    # Initialize the database connection
    Control.instance().db.initialize_connection()


def run_gunicorn(app, host, port, workers=4):
    bind_address = f"{host}:{port}"
    options = {
        "bind": bind_address,  # Specify the bind address and port here
        "workers": workers,
        # After forking, initialize the database connection
        "post_fork": post_fork,
    }
    GunicornApp(app, options).run()
