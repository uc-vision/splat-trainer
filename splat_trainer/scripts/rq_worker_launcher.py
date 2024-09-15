import importlib.resources

from flask import Flask, render_template

from splat_trainer.util.deploy import get_args, deploy_workers



with importlib.resources.path('splat_trainer', 'templates') as template_folder:
    app = Flask(__name__, template_folder=str(template_folder))
    
data = {}
file_paths = {}

@app.route('/')
def index():
  global data
  return render_template('index.html', data=data)



def main():
    global data
    global file_paths

    args = get_args()
    deployment = deploy_workers(args)
    result = deployment()
    data.update(result)

    app.run(host="0.0.0.0", debug=True, port=8000, use_reloader=False)

if __name__ == '__main__':
    main()