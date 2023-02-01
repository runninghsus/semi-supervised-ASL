# Sign Language Translation App
[![GitHub stars](https://img.shields.io/github/stars/runninghsus/semi-supervised-ASL.svg?style=social&label=Star)](https://github.com/YttriLab/A-SOID)
[![GitHub forks](https://img.shields.io/github/forks/runninghsus/semi-supervised-ASL.svg?style=social&label=Fork)](https://github.com/YttriLab/A-SOID)

<p align="center">
  <img src="images/hands_logo.jpeg" />
</p>

## Introduction
This no code interface will serve to be a
proof-of-concept for sign language translation across 
English, Spanish, and Chinese.

## Installation
```commandline
conda env create environment.yaml
conda activate hands
```

## Usage
```commandline
cd semi-supervised-ASL
streamlit run hands.py
```
The streamlit app composes of 3 main modules.


<p align="center">
  <img src="images/app.png" />
</p>

* Upload video to obtain pose.
* Train a neural network that recognizes sign langugage.
* Translate into other languages.


## References

