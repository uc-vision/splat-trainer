from typing import Any, Optional, Union
import threading
import importlib.resources
import logging

from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf
import redis
import signal
import importlib.resources
import rq_dashboard
from flask import Flask, render_template
from rq import Worker

from splat_trainer.util.deploy import get_all_workers, deploy_workers, shutdown_all_workers


class ManageRQWorkers(Callback):
    def __init__(self, redis_url, flask_port, config_file):
        self.data = {}
        self.redis_url = redis_url
        self.flask_port = flask_port
        self.config_file = config_file


    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:

        result = deploy_workers(self.config_file)
        self.data.update(result)

        workers, conn = get_all_workers(self.redis_url)
        print(workers)

        flask_thread = threading.Thread(target=self.run_flask_app, daemon=True)
        flask_thread.start()

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTSTP, self.signal_handler)

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        shutdown_all_workers(self.redis_url)
        exit(0)
        

    def signal_handler(self, signum, frame):
        shutdown_all_workers(self.redis_url)

    
    def run_flask_app(self):

        flask_url = f"http://127.0.0.1:{self.flask_port}"

        print("\n" + " " + "-" * 61)
        print(f"|{' ' * 61}|")
        print(f"| Running on {flask_url:<46}   |")
        print(f"| RQ Dashboard running on: {(flask_url + '/rq'):<31}    |")
        print(f"| Running on all addresses (0.0.0.0){' ' * 25} |")
        print(f"| Press CTRL+C to quit the server{' ' * 28} |")
        print(f"|{' ' * 61}|")
        print(" " + "-" * 61 + "\n")

        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        with importlib.resources.path('splat_trainer', 'templates') as template_folder:
            app = Flask(__name__, template_folder=str(template_folder))

        app.config.from_object(rq_dashboard.default_settings)
        app.config['RQ_DASHBOARD_REDIS_URL'] = self.redis_url
        rq_dashboard.web.setup_rq_connection(app)
        app.register_blueprint(rq_dashboard.blueprint, url_prefix="/rq")
               
        @app.route('/')
        def index():
            return render_template('index.html', data=self.data)

        app.run(host="0.0.0.0", debug=True, port=self.flask_port, use_reloader=False)




