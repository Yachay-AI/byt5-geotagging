|Join our community | <a href="https://discord.gg/msWFtcfmwe"><img src="https://img.shields.io/badge/Discord-%235865F2.svg?style=for-the-badge&logo=discord&logoColor=white"></img></a>|
| :------------ | :-------------------------------------------------------------------------------------------------------- |


![Cover](./yachay.png) 


<p align="center">
<a href="https://github.blog/2023-07-13-release-radar-spring-23/#yachay-ai-1-0"><img src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103"></img></a>
<a href="https://huggingface.co/yachay"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-datasets-FCE205"></a>
<a href="https://python.org"><img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=python&logoColor=white"></img></a>
<a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-1.7.1-EE4C2C.svg?style=flat&logo=pytorch"></img></a>
<a href="https://github.com/Yachay-AI/byt5-geotagging/releases"><img alt="Version" src="https://img.shields.io/github/v/release/Yachay-AI/byt5-geotagging?include_prereleases"></a>
<a href="https://github.com/Yachay-AI/byt5-geotagging/blob/master/LICENSE.md"><img src="https://badgen.net/github/license/Yachay-AI/byt5-geotagging"></img></a>
</p> 



# Geotagging Model

This repository is designed to support developers in building and training their own geotagging models. The geotagging model architecture provided here allows for customization and training. Additionally, we publish datasets that are well-suited for training in different geolocation detection scenarios.


The current models reach 30km Median Error on Haversine Distance for top 10% most relevant texts. Challenges in the repository issues are open to improve the model's performance.

## Architecture and Training
<details>
<summary>Click to unfold geotagging model architecture diagram. </summary>

```mermaid
%%{init:{'theme':'neutral'}}%%
flowchart TD
subgraph "ByT5 classifier"
  a("Input text") --> b("Input_ids")
subgraph "byt5(T5EncoderModel)"
  b("Input_ids")  --> c("byt5.encoder.inp_input_ids")
subgraph "byt5.encoder(T5Stack)"
  c("byt5.encoder.inp_input_ids")  --> d("byt5.encoder.embed_tokens") 
subgraph "byt5.encoder.embed_tokens (Embedding)"
  d("byt5.encoder.embed_tokens")  --> f("embedding")
  e("byt5.encoder.embed_tokens.inp_weights") --> f("embedding") --> g("byt5.encoder.embed_tokens.out_0")
end
  g("byt5.encoder.embed_tokens.out_0") --> h("byt5.encoder.dropout(Dropout)") --> i("byt5.encoder.block.0(T5Block)") --> j("byt5.encoder.block.1(T5Block)") & k("byt5.encoder.block.2-9(T5Block)") & l("byt5.encoder.block.10(T5Block)")
  j("byt5.encoder.block.1(T5Block)") --> k("byt5.encoder.block.2(T5Block)<br><br> ...<br><br>byt5.encoder.block.10(T5Block) ") --> l("byt5.encoder.block.11(T5Block)") --> m("byt5.encoder.final_layer_norm(T5LayerNorm)")
  m("byt5.encoder.final_layer_norm(T5LayerNorm)")-->n("byt5.encoder.dropout(Dropout)")--> o("byt5.encoder.out_0")
end
o("byt5.encoder.out_0") --> p("byt5.out_0")
end
p("byt5.out_0")-->q("(Linear)")
end
q("(Linear)") -->r("logits")
```
</details>

|  Train your text-to-location model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yachay-AI/byt5-geotagging/blob/master/colab/byt5_training_demo.ipynb)|
| :------------ | :-------------------------------------------------------------------------------------------------------- |

#### Dependencies
Ensure that the following dependencies are installed in your environment to build and train your geotagging model:

```
transformers==4.29.1
tqdm==4.63.2
pandas==1.4.4
pytorch==1.7.1
```

To train your geotagging model using the ByT5-encoder based approach, execute the following script:

```bash
python train_model.py --train_input_file <training_file> --test_input_file <test_file> --do_train true --do_test true --load_clustering .
```

Refer to the `train_model.py` file for a comprehensive list of available parameters.

## Output Example

```json

{
   "text":"These kittens need homes and are located in the Omaha area! They have their shots and are spayed/neutered. They need to be gone by JAN 1st! Please Retweet to help spread the word!",
   "geotagging":{
      "lat":41.257160,
      "lon":-95.995102,
      "confidence":0.9950085878372192
   }
}
```

```geojson
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": 1,
      "properties": {
        "ID": 0
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [-96.296363, 41.112793],
            [-96.296363, 41.345177],
            [-95.786877, 41.345177],
            [-95.786877, 41.112793],
            [-96.296363, 41.112793]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "id": 2,
      "properties": {
        "ID": 0
      },
      "geometry": {
        "type": "Point",
        "coordinates": [-95.995102, 41.257160]
      }
    }
  ]
}

```

## Datasets
Our team has curated two comprehensive datasets for two distinct training approaches. These datasets are intended for use in training and validating the models. [Share your training results in the repository issues](https://github.com/Yachay-AI/byt5-geotagging/issues). 

|Regions dataset| [![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1thkE-hgT3sDtZqILZH17Hyayy0hkk_jh/view?usp=share_link) |
| :------------ | :-------------------------------------------------------------------------------------------------------- |

The goal of the Regions approach is to look into the dataset of top most populated regions around the world.

- is an annotated corpus of 500k texts, as well as the respective geocoordinates
- covers 123 regions
- includes 5000 tweets per location


|Seasons dataset| [![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1P2QUGFBKaqdpZ4xAHmJMe2I57I94MJyO?usp=sharing) |
| :------------ | :-------------------------------------------------------------------------------------------------------- |

The goal of the Seasons approach is to identify the correlation between the time/date of post, the content, and the location. Time zone differences, as well as seasonality of the events, should be analyzed and used to predict the location. For example: snow is more likely to appear in the Northern Hemisphere, especially if in December. Rock concerts are more likely to happen in the evening and in bigger cities, so the time of the post about a concert should be used to identify the time zone of the author and narrow down the list of potential locations. 

- is a .json of >600.000 texts 
- collected over the span of 12 months
- covers 15 different time zones 
- focuses on 6 countries (Cuba, Iran, Russia, North Korea, Syria, Venezuela)

**Your custom data.** The geotagging model supports training and testing on custom datasets. Prepare your data in CSV format with the following columns: `text`, `lat`, and `lon`. 


## Confidence and Prediction 
The geotagging model incorporates confidence estimation to assess the reliability of predicted coordinates. The Relevance field in the output indicates prediction confidence, ranging from `0.0` to `1.0.` Higher values indicate increased confidence.
 For detailed information on confidence estimation and how to utilize the model for geotagging predictions, please refer to the `inference.py` file. This file provides an example script demonstrating the model architecture and integration of confidence estimation.

## Welcome!
### Forkers
[![Forkers repo roster for @Yachay-AI/byt5-geotagging](https://bytecrank.com/nastyox/reporoster/php/forkersSVG.php?user=Yachay-AI&repo=byt5-geotagging)](https://github.com/Yachay-AI/byt5-geotagging/network/members)

Feel free to explore the code, adapt it to your specific requirements, and integrate it into your projects. If you have any questions or require assistance, please don't hesitate to reach out. We highly appreciate your feedback and are dedicated to continuously enhancing the geotagging models.
 
<p align="center"> 
<a href="https://www.reddit.com/user/yachay_ai/"><img src="https://img.shields.io/badge/Reddit-%23FF4500.svg?style=for-the-badge&logo=Reddit&logoColor=white"></img></a>  
<a href="https://twitter.com/YachayAi"><img src="https://img.shields.io/badge/Twitter-%231DA1F2.svg?style=for-the-badge&logo=Twitter&logoColor=white"></img></a>  
<a href="https://yachay.hashnode.dev/"><img src="https://img.shields.io/badge/Hashnode-2962FF?style=for-the-badge&logo=hashnode&logoColor=white"></img></a>  
<a href="https://dev.to/yachayai"><img src="https://img.shields.io/badge/dev.to-0A0A0A?style=for-the-badge&logo=dev.to&logoColor=white"></img></a>  
</p>


