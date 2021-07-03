# Measuring economic activity from space: a case study using flying airplanes and COVID-19

This work introduces a novel solution to measure economic activity through remote sensing for a wide range of spatial areas. We hypothesized that disturbances in human behavior caused by major life-changing events leave signatures in satellite imagery that allows devising relevant image-based indicators to estimate their impacts and support decision-makers. We present a case study for the COVID-19 coronavirus outbreak, which imposed severe mobility restrictions and caused worldwide disruptions, using flying airplane detection around the 30 busiest airports in Europe to quantify and analyze the lockdown's effects and post-lockdown recovery. Our solution won the [Rapid Action Coronavirus Earth observation (RACE) upscaling challenge](https://eo4society.esa.int/2020/04/24/from-the-covid-19-custom-script-contest-to-the-euro-data-cube-european-dashboard/), sponsored by the European Space Agency and the European Commission, and now integrates the [RACE dashboard](https://race.esa.int/?indicator=E13d). This platform combines satellite data and artificial intelligence to promote a progressive and safe reopening of essential activities. [[arXiv]](https://arxiv.org/abs/2104.10345)

## Awards

- [Best contribution (super-prize)](https://medium.com/sentinel-hub/race-upscaling-competition-results-8a339bb8c942) -- RACE upscaling challenge

## Authors

- Mauricio Pamplona Segundo ([USF](https://www.usf.edu/))
- Allan Pinto ([UNICAMP](https://www.unicamp.br/))
- Rodrigo Minetto ([UTFPR](http://www.utfpr.edu.br/))
- Ricardo Da Silva Torres ([NTNU](https://www.ntnu.edu/))
- Sudeep Sarkar ([USF](https://www.usf.edu/))

## Data availability

There are two ways of obtaining the data employed in this research. The first one is through a [Sentinel Hub account](https://www.sentinel-hub.com/) by using the script `download.py` . Make sure you update the following variables with your Sentinel Hub credentials:

```python
config.instance_id = '***REMOVED***'
config.sh_client_id = '***REMOVED***'
config.sh_client_secret = '***REMOVED***'
```

These credentials can be obtained in the [Sentinel Hub dashboard](https://apps.sentinel-hub.com/dashboard/).

The second option is to download the data from our institutional repository. The data consistis of one zip file for all cloud probability masks from all airports (cloud_masks.zip (11.8 GB)) and one zip file per airport with color images and valid pixel masks as follows:

| file (size) | file (size) | file (size) | file (size) | file (size) |
|---|---|---|---|---|
| AGP.zip (22.3 GB) | CDG.zip (14.5 GB) | HEL.zip (12.2 GB) | MAD.zip (33.4 GB) | PMI.zip (19.4 GB) |
| AMS.zip (10.2 GB) | CPH.zip (13.0 GB) | IST.zip (18.2 GB) | MAN.zip (7.4 GB) | STN.zip (14.7 GB) |
| ARN.zip (16.2 GB) | DUB.zip (9.0 GB) | LGW.zip (14.2 GB) | MUC.zip (18.3 GB) | TXL.zip (15.7 GB) |
| ATH.zip (17.5 GB) | DUS.zip (13.0 GB) | LHR.zip (11.1 GB) | MXP.zip (19.4 GB) | VIE.zip (20.0 GB) |
| BCN.zip (18.9 GB) | FCO.zip (20.6 GB) | LIS.zip (27.0 GB) | ORY.zip (14.7 GB) | WAW.zip (16.2 GB) |
| BRU.zip (14.2 GB) | FRA.zip (9.1 GB) | LTN.zip (9.8 GB) | OSL.zip (12.4 GB) | ZRH.zip (18.6 GB) |

These files will be made available upon request to the e-mail `mauriciop@usf.edu`. Requesters will be granted one month of access to download the aforementioned files. All these files contain modified Sentinel-2 data processed by Euro Data Cube. If you use any of these images for any kind of media (publications, reports, videos, etc), please add the message "Contain modified Sentinel-2 data processed by Euro Data Cube" close to the image.

## Usage instructions

To download images from Sentinel Hub, run the following command:

```
$ mkdir images
$ python3 download.py
```

This command will download all images from the chosen airports within the specified time interval and save them in the folder `images`. If you decide to download the images from our institutional repository, just unzip all airport files inside the folder `images` (cloud masks are not necessary for training/inference).

To detect airplanes, run:

```
$ python3 inference.py
```

This code will use our pre-trained model `models/flying41.pytorch`. The log of detections will be saved in the folder `log`. These log files are then converted into time series through the following command:

```
$ python3 create_timeseries.py
```

The time series are saved in the folder `timeseries`. Both `log` and `timeseries` folders have pre-generated results.

If you decide to train your own model, run:

```
$ python3 train.py
```

This will use the annotations available in this repository to train a new detection model. Make sure you rename the model name on `inference.py` to use your own model for inference.

### Time series analysis

To perform the airports' activity analysis by detecting the breaking points related to COVID-19 and the recovery rate of the monitored airports, please follow the instructions available in [covid19-airports-activity-analysis/README.md](https://github.com/allansp84/covid19-airports-activity-analysis.git)


## Citing

If you find the code in this repository useful in your research, please consider citing:
```
@misc{segundo2021measuring,
      title={Measuring economic activity from space: a case study using flying airplanes and COVID-19}, 
      author={Mauricio Pamplona Segundo and Allan Pinto and Rodrigo Minetto and Ricardo da Silva Torres and Sudeep Sarkar},
      year={2021},
      eprint={2104.10345},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
