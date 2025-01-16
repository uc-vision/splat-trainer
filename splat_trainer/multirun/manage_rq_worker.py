import importlib.resources
import logging
import socket
import sys
import threading
import traceback
from typing import Any, Optional, Union
from types import SimpleNamespace

import redis
import rq_dashboard
import signal
from flask import Flask, render_template
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf
from rq import Worker

from splat_trainer.multirun.deploy import deploy_workers, shutdown_all_workers, flush_all


class ManageRQWorkers(Callback):
    def __init__(self, 
                config_file: str, 
                flask_port: int=8000, 
                redis_port: int=6379, 
                get_pass: bool=False,
                max_num_worker_on_each_machine: int=1):
        self.data = {}

        self.args = SimpleNamespace(
            config=config_file,
            getpass=get_pass,
            flask_port=flask_port,
            redis_url=f'redis://{socket.gethostname()}:{redis_port}',
            max_num_worker=max_num_worker_on_each_machine
        )

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTSTP, self.signal_handler)


    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        flush_all(self.args.redis_url)

        try:
            result = deploy_workers(self.args)
        
        except Exception as e:
            error_message = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            raise SystemExit(error_message)  
            self.shutdown_all_workers()

        self.data.update(result)

        flask_thread = threading.Thread(target=self.run_flask_app, daemon=True)
        flask_thread.start()


    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        self.shutdown_all_workers()
        pass
        

    def signal_handler(self, signum, frame):
        self.shutdown_all_workers()
        sys.exit(0)


    def shutdown_all_workers(self):
        try:
            print("Shutting down all workers and exiting.")
            shutdown_all_workers(self.args.redis_url)

        except Exception as e:
            print(f"Error while shutting down workers: {e}")

        # finally:
        #     sys.exit(0)

    
    def run_flask_app(self):
        flask_url = f"http://127.0.0.1:{self.args.flask_port}"

        print(" " + "-" * 61)
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
        app.config['RQ_DASHBOARD_REDIS_URL'] = self.args.redis_url
        rq_dashboard.web.setup_rq_connection(app)
        app.register_blueprint(rq_dashboard.blueprint, url_prefix="/rq")
               
        @app.route('/')
        def index():
            return render_template('index.html', data=self.data)

        app.run(host="0.0.0.0", debug=True, port=self.args.flask_port, use_reloader=False)



