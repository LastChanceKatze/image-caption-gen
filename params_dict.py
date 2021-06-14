train_model_params = {'model':model,
     'epochs': 20,
     'batch_size': 64,
     'plot_hist': True,
     'train_captions': train_captions,
     'test_captions': test_captions,
     'filepath': drive_folder + "/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
     'tokenizer': tokenizer,
     'max_length': max_length,
     'vocab_size': vocab_size,
     'train_features': train_features,
     'test_features': test_features
     }