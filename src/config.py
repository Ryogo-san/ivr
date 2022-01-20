class CFG:
    debug = False
    epochs = 300
    gpus = 1
    model_dir = "./logs/models/"
    output_dir = "./logs/"
    data_dir = "./data/dataset/"
    ckpt_path = "./logs/models/resnet34d_best_model.ckpt"
    #ckpt_path = "./logs/models/efficientnet_b0_best_model.ckpt"
    num_workers = 4
    image_size = 256
    seed = 1234
    model_name = "resnet34d"
    batch_size = 128
    loss = "CE"
    alpha=0.3 # balance of alpha*l1 + (1-alpha)*l2
    optimizer = "Adam"
    scheduler = "CosineAnnealingLR"
    T_max = 3
    learning_rate = 1e-4
    min_lr = 1e-6
    weight_decay = 1e-6
    patience = 10
    num_method_classes = 3
    num_letter_classes = 46

    classifier_models=["efficientnet_b0","efficientnet_b1", "densenet121d"]

    fc_models = ["resnet34", "resnet34d", "resnet50", "resnet50d"]

    pert_eps=8/255
    if debug:
        epochs = 1
