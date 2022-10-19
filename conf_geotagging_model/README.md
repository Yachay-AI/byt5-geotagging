# Confidence Geotagging model

## Training
To train you can follow instruction in `training.ipynb` or you can run it with `train.py` by the command in the terminal
```bash
python train.py --data_dir <data_path> --save_prefix <model_path> --arch char_lstm --split_uids
--batch_size 128 --loss l1 --optimizer adamw --scheduler constant --lr 5e-4 --num_epoch 10 
--conf_estim --confidence_validation_criterion
```

All arguments can be seen in [aux_files/args_parser.py](./aux_files/args_parser.py)
### Custom data
For loading custom date you need to implement Dataloader in [data_loading.py](./data_loading.py). This dataloader must return `list of texts, list of [longitude, latitude]`. Then add it to `get_dataset` method in [aux_files/args_parser.py](./aux_files/args_parser.py)
To load your data, you need to implement a Dataloader in [data_loading.py](./data_loading.py). This Dataloader must return "a list of texts, a list of [longitude, latitude]". Then add it to the `get_dataset` method in the file [aux_files/args_parser.py](./aux_files/args_parser.py) and then you can select it with `dataset_name` argument.

### Confidence
To use confidence estimation, set the `conf_estim` and `confidence_validation_criterion` arguments to true. You can set the array to `model_save_band` to show the top predictions  by `confidence_bands` (as a percentage from 0 to 100).
Use `model_save_band` to save the model by the best metric value for the selected band.
## Prediction
Example of using the trained model is in [prediction.ipynb](./prediction.ipynb)

