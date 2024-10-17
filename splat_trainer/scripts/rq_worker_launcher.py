import signal
import importlib.resources

from flask import Flask, render_template

from splat_trainer.util.deploy import get_args, deploy_workers, terminate_all



with importlib.resources.path('splat_trainer', 'templates') as template_folder:
  app = Flask(__name__, template_folder=str(template_folder))

data = {}


@app.route('/')
def index():
  global data
  return render_template('index.html', data=data)


def signal_handler(signum, frame):
  global data
  result = terminate_all(data)
  data.update(result)

  exit(0)



def main():
  global data

  args = get_args()
  result = deploy_workers(args)
  data.update(result)

  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTSTP, signal_handler)

  app.run(host="0.0.0.0", debug=True, port=args.port, use_reloader=False)


if __name__ == '__main__':
    main()