## COMPETITION ON SCRIPT IDENTIFICATION IN THE WILD (SIW 2021)

This repository contains our source code in the RIVF2021 MC-OCR Competition.

### Introduction

Identifying the language of the script is a significant step that helps us to use the corresponding Optical Character Recognition (OCR) model for script recognition. In this work, we proposed our solution to identify the handwritten and printed script in the Script Identification in the Wild (SIW) competition in ICDAR2021. We developed a systematic process that first trains a ResNet model to classify the script as handwriting or print, then for each type of these we use our corresponding EfficientNet model to identify the language of the script. Our solution achieved compelling performance on the leaderboard during the competition.

Detailed information of SIW 2021 can be found [here](https://sites.google.com/view/ICDAR21-SIW2021/home).
