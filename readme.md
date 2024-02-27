
# Dinstinguishing brain signals
In this project I will use an open dataset (from the LIMO task - Guillaume A. Rousselet. LIMO EEG dataset. 2016. doi:10.7488/ds/1556.).

In this dataset, participants were subjected to a task in which they would observe two different faces (face A/face B) across trials. In each trial, the images could be 'less coherent' - meaning more blurred. Then, participants had to report which face they had just seen. 

However, the question I'm trying to investigate is: can the EEG signal determine which face the participant was observing?

To this end, I built this script to read the EEG data via MNE and test different decoders to find which more accurately can distinguish from the EEG signal if the participant was subjected to seeing Face A or Face B.


## Run Locally

Clone the project

```bash
  git clone https://github.com/joaohnp/EEG_decoding
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  pip install -r requeriments.txt
```

Run EDA.py

```bash
  python EDA.py
```

After EDA has been ran, we now have accuracy scores for different models. With this we can investigate which classifier works best with our data. 

```bash
  python best_model.py
```
Now we can check in detail which decoder worked best in which electrode. 


## Tech Stack

**Data preparation and analysis:** MNE, numpy

**Machine Learning and Deep Learning:** TensorFlow, XGBoost, LDA, SVM


## Authors

- [@joaohnp](https://www.github.com/joaohnp)


## Support

For support, email joaohnp@gmail.com or join our Slack channel.

