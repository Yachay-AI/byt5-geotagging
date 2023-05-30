[![Cover](https://user-images.githubusercontent.com/29067628/197814413-90cc6585-4580-48a8-88e4-ee2413198a09.png)](https://www.yachay.ai/) 

<p align="center">
<a href="https://twitter.com/YachayAi"><img src="https://img.shields.io/badge/Follow%20us-%40YachayAi-blue?style=plastic&logo=twitter"></img></a>
<a href="https://www.reddit.com/user/yachay_ai"><img src="https://img.shields.io/badge/Follow%20us-u%2Fyachay__ai-orange?style=plastic&logo=reddit"></img></a>
</p> 

<p align="center">
<a href="https://github.com/Yachay-AI/byt5-geotagging/stargazers"><img src="https://badgen.net/github/stars/Yachay-AI/byt5-geotagging"></img></a>
<a href="https://github.com/Yachay-AI/byt5-geotagging/forks"><img src="https://badgen.net/github/forks/Yachay-AI/byt5-geotagging"></img></a>
<a href="https://github.com/Yachay-AI/byt5-geotagging/contributors"><img src="https://badgen.net/github/contributors/Yachay-AI/byt5-geotagging"></img></a>
<a href="https://github.com/Yachay-AI/byt5-geotagging/issues"><img src="https://badgen.net/github/issues/Yachay-AI/byt5-geotagging"></img></a>
<a href="https://github.com/Yachay-AI/byt5-geotagging/blob/master/LICENSE.md"><img src="https://badgen.net/github/license/Yachay-AI/byt5-geotagging"></img></a>
</p> 

## Byt5 Geotagging Model
### Current Scores
Current byt5 based geotagging model has the following validation metrics:
- Median Average Error (Haversine Distance) of 19.22km
- Mean Average Error (Haversine Distance) of 434.16km 

### Training 
For ByT5 model training, use the following script:

```bash
python train_model.py --train_input_file <training_file> --test_input_file <test_file> --do_train true --do_test true --load_clustering .
```

The full list of parameters can be found in `train_model.py`.

#### Dependencies
```
transformers==4.29.1
tqdm==4.63.2
pandas==1.4.4
pytorch==1.7.1
```

### Custom Data 
Custom training and testing data sets should be formatted as CSVs with the following columns: `text`, `lat`, `lon`.

For relevant data sets suggested by the Yachay team, please check the Existing Challenges section.

### Confidence 
For confidence estimation, see `inference.py`. 
- Check the `Relevance` field, ranged from 0.0 to 1.0 
- Higher relevance values correspond to a higher prediction confidence

### Prediction 
An example of using the trained model is in `inference.py`.

## Existing Challenges and Suggested Data Sets
### Regions Challenge 

The first suggested methodology (Challenge 1) on training the model is to look into the dataset of top most populated regions around the world.

The provided dataset is **[here](https://drive.google.com/file/d/1thkE-hgT3sDtZqILZH17Hyayy0hkk_jh/view?usp=share_link)**, which:

- is an annotated corpus of 500k texts, as well as the respective geocoordinates
- covers 123 regions
- includes 5000 tweets per location

### Seasons Сhallenge

Challenge 2 sets the goal to identify the correlation between the time/date of post, the content, and the location. 

Time zone differences, as well as seasonality of the events, should be analyzed and used to predict the location. For example: snow is more likely to appear in the Northern Hemisphere, especially if in December. Rock concerts are more likely to happen in the evening and in bigger cities, so the time of the post about a concert should be used to identify the time zone of the author and narrow down the list of potential locations.


The provided dataset is **[here](https://drive.google.com/drive/folders/1P2QUGFBKaqdpZ4xAHmJMe2I57I94MJyO?usp=sharing)**, which:
- is a .json of >600.000 texts 
- collected over the span of 12 months
- covers 15 different time zones 
- focuses on 6 countries (Cuba, Iran, Russia, North Korea, Syria, Venezuela)

## Resources
### Contact 

If you would like to contact us with any questions, concerns, or feedback, help@yachay.ai is our email.

You also can check out our site, [yachay.ai](https://www.yachay.ai/), or any of our socials below.


<a href="https://discord.gg/msWFtcfmwe"><img src="https://cdn-icons-png.flaticon.com/512/3670/3670157.png" width=5% height=5%></img></a>     <a href="https://twitter.com/YachayAi"><img src="https://cdn-icons-png.flaticon.com/128/3670/3670151.png" width=5% height=5%></img></a>     <a href="https://www.reddit.com/user/yachay_ai"><img src="https://cdn-icons-png.flaticon.com/512/3670/3670226.png" width=5% height=5%></img></a>

## Stargazers 
[![Stargazers repo roster for @Yachay-AI/byt5-geotagging](https://reporoster.com/stars/Yachay-AI/byt5-geotagging)](https://github.com/Yachay-AI/byt5-geotagging/stargazers)
## Forkers
[![Forkers repo roster for @Yachay-AI/byt5-geotagging](https://reporoster.com/forks/Yachay-AI/byt5-geotagging)](https://github.com/Yachay-AI/byt5-geotagging/network/members)

