from jinja2 import Environment, FileSystemLoader
from utils import create_directory

def write_code(config, project_path):
    generate_path = project_path + '/code'
    create_directory(generate_path)

    null_dict = {}

    environment = Environment(loader=FileSystemLoader("template/"))
    if config.model == 'kobert':
        template_py = environment.get_template("kobert_python.txt")
        template_ipynb = environment.get_template("kobert_ipynb.txt")
    elif config.model == 'kogpt2':
        template_py = environment.get_template("kogpt2_python.txt")
        template_ipynb = environment.get_template("kogpt2_ipynb.txt")

    filename_py = generate_path + f"/{config.model}.py"
    filename_ipynb = generate_path + f"/{config.model}.ipynb"

    content_py = template_py.render(null_dict, task=config.task, input=config.input, target=config.target, max_length=config.max_length, test_size=config.test,
                                    dropout=config.dropout, lr=config.lr, n_epochs=config.n_epochs, batch_size=config.batch_size)
    content_ipynb = template_ipynb.render(null_dict, task=config.task, input=config.input, target=config.target, max_length=config.max_length, test_size=config.test,
                                          dropout=config.dropout, lr=config.lr, n_epochs=config.n_epochs, batch_size=config.batch_size)

    with open(filename_py, mode="w", encoding="utf-8") as message:
        message.write(content_py)
        print(f"saving code to {filename_py}")

    with open(filename_ipynb, mode="w", encoding="utf-8") as message:
        message.write(content_ipynb)
        print(f"saving code to {filename_ipynb}")