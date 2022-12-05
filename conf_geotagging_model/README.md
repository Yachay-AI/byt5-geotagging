# Yachay: Shared Knowledge

![Cover](https://user-images.githubusercontent.com/29067628/197814413-90cc6585-4580-48a8-88e4-ee2413198a09.png)


Yachay is an open-source platform for Machine Learning. It has collected decades worth of useful natural language data from traditional media (i.e. New York Times articles), social media (i.e. Twitter & Reddit), messenger channels, tech blogs, GitHub profiles and issues, the dark web, and legal proceedings, as well as the decisions and publications of government regulators and legislators all across the world.

Yachay has cleaned and annotated this data, prepared the infrastructure, and released a state-of-the-art Geolocation Detection tool. 

This repository is an open call for all Machine Learning developers and researchers to help us build and improve upon the existing geotagging models. 

## Geotagging Model How-Tos 

### Training 💪
To train your own model, follow the instructions in `training.ipynb` or run it with `train.py` in the terminal:
```bash
python train.py --data_dir <data_path> --save_prefix <model_path> --arch char_lstm --split_uids
--batch_size 128 --loss l1 --optimizer adamw --scheduler constant --lr 5e-4 --num_epoch 10 
--conf_estim --confidence_validation_criterion
```

All arguments can be seen in [aux_files/args_parser.py](./aux_files/args_parser.py)

#### Dependencies:
- pytorch == 1.7.1
- numpy == 1.19.2
- scikit-learn == 0.22.2
- tqdm == 4.62.3
- pandas == 1.0.3

### Custom Data 📚
To upload a custom dataset, you will need to implement a Dataloader in [data_loading.py](./data_loading.py). This Dataloader must return a `list of texts, a list of coordinates [longitude, latitude]`. Then, add the result to the `get_dataset` method in [aux_files/args_parser.py](./aux_files/args_parser.py), and you'll be able to select it with the `dataset_name` argument.

For relevant data sets, please check the Existing Challenges section

### Confidence 🔥
To use confidence estimation, set the `conf_estim` and `confidence_validation_criterion` arguments to True. You can set the array to `model_save_band` to show the top predictions by `confidence_bands` (as a percentage from 0 to 100).
Use `model_save_band` to save the model by the best metric value for the selected band.

### Prediction 🔮
An example of using the trained model is in [prediction.ipynb](./prediction.ipynb)

## Existing Challenges/Suggested Data Sets
### Challenge 1 - 100 regions 🌎

The first suggested methodology on training the model is to look into the dataset of 100 regions around the world, where each region is represented by 5,000 tweets. The dataset attached provides an annotated corpus of 500k texts, as well as the respective geocoordinates.

The data set is [here](https://drive.google.com/file/d/1J5ducw8O628wyXD7qdcvop2pyNbq26tO/view?usp=sharing)

![This is an image](https://media.tenor.com/lOPTx_JZJ3gAAAAC/the-office-steve-carell.gif)

### Challenge 2 - 12 months 🌎

Challenge 2 sets the goal to identify the correlation between the time/date of post, the content, and the location. Time zones difference, as well as seasonality of the events, should be analyzed and used to predict the location. For example: snow is more likely to appear in the Northern Hemisphere, especially if in December. Rock concerts are more likely to happen in the evening and in bigger cities, so the time of the post about a concert should be used to identify the time zone of the author and narrow down the list of potention locations.

The data set provided is a .json of >600.000 texts collected over the span of 12 months, covering 15 different time zones and 6 countries. (Cuba, Iran, Russia, North Korea, Syria, Venezuela). 

The data set is [here](https://drive.google.com/drive/folders/1P2QUGFBKaqdpZ4xAHmJMe2I57I94MJyO?usp=sharing)

## Resources
### Contact 📩

If you would like to contact us with any questions, concerns, or feedback, help@yachay.ai is our email.

You also can check out our site, [yachay.ai](https://www.yachay.ai/), or any of our socials below.

### Social Media 📱


<a href="https://discord.gg/msWFtcfmwe"><img src="https://cdn-icons-png.flaticon.com/512/3670/3670157.png" width=5% height=5%></img></a>     <a href="https://twitter.com/YachayAi"><img src="https://cdn-icons-png.flaticon.com/128/3670/3670151.png" width=5% height=5%></img></a>     <a href="https://www.reddit.com/user/yachay_ai"><img src="https://cdn-icons-png.flaticon.com/512/3670/3670226.png" width=5% height=5%></img></a>

