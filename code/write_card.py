from jinja2 import Environment, FileSystemLoader

def write_card(config, for_card, project_path):
    null_dict = {}

    environment = Environment(loader=FileSystemLoader("template/"))
    template = environment.get_template("project_card.txt")

    if config.model == 'kobert':
        tokenizer = 'klue/bert-base'
        model = 'klue/bert-base'
    elif config.model == 'kogpt2':
        tokenizer = 'skt/kogpt2-base-v2'
        model = 'skt/kogpt2-base-v2'

    content = template.render(null_dict, task=config.task,
                              dataset=config.data_fn, data_size=for_card['data_size'], row=for_card['row'], column=for_card['column'],
                              split_type=config.split_type, train=1-config.test, valid=(1-config.test)*0.2, test=config.test,
                              input=config.input, target=config.target, num_labels=for_card['num_labels'], tokenizer=tokenizer,
                              model=model,
                              max_length=config.max_length, n_epochs=config.n_epochs, lr=config.lr, dropout=config.dropout, batch_size=config.batch_size,
                              auto_tuning=config.auto,
                              loss=for_card['result']['loss'], accuracy=for_card['result']['accuracy'],
                              val_loss=for_card['result']['val_loss'], val_accuracy=for_card['result']['val_accuracy'],
                              test_loss=for_card['result']['test_loss'], test_accuracy=for_card['result']['test_accuracy'])

    with open(project_path + '/project_card.json', mode="w", encoding="utf-8") as message:
        message.write(content)
        print("Project card save complete!")