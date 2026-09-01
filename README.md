<div align="center">

<img src="assets/spacepoint_logo.png" width="200">

# SpacePoint — Tiered Edge AI Land Cover Classification

**Land cover classification deployed across three points on the power-versus-capability curve: ESP32 → STM32 → Jetson Nano, using drone imagery as a stand-in for a future CubeSat mission.**

![Python](https://img.shields.io/badge/Python-3.10+-a855f7?style=flat-square&labelColor=231134&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Model_Training-8b5cf6?style=flat-square&labelColor=231134&logo=tensorflow&logoColor=white)
![TFLite Micro](https://img.shields.io/badge/TFLite_Micro-ESP32_Inference-7c3aed?style=flat-square&labelColor=231134)
![Arduino](https://img.shields.io/badge/Arduino-ESP32_Firmware-653F84?style=flat-square&labelColor=231134&logo=arduino&logoColor=white)
![X--CUBE--AI](https://img.shields.io/badge/X--CUBE--AI-STM32_Inference-653F84?style=flat-square&labelColor=231134)
![OpenCV](https://img.shields.io/badge/OpenCV-Dataset_Labeling-6d28d9?style=flat-square&labelColor=231134&logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-In_Progress-6d28d9?style=flat-square&labelColor=231134)

━━━━━━━━━━━━━━━━━━━━ ✦ ━━━━━━━━━━━━━━━━━━━━
</div>

## 🛰️ At a Glance

- **Three-tier deployment** — the same land cover classification task, scaled and quantized to fit three different memory budgets, from a 20 KB microcontroller up to a Jetson Nano
- **Real annotated training data** — built from the TU Graz Aerial Semantic Segmentation Drone Dataset's human-labeled masks, not synthetic or pseudo-labels
- **Hardware-scoped, not hidden** — the STM32 tier's task was deliberately reduced from 7 to 4 classes to fit a 20 KB SRAM budget; the reasoning and trade-off are documented, not glossed over
- **Full quantization pipeline** — Keras → int8 TFLite → on-device C byte arrays, checked against the model's own true scale/zero-point rather than assumed constants
- **Class-weighted training** — corrects an ~18:1 imbalance between the rarest and most common classes in the STM32 tier's merged dataset
- **Bench-tested without camera hardware** — static image tests verified over serial before any live camera integration

<br/>

## 🛰️ Overview

Runs one land cover classification task — vegetation, water, bare soil, and built surfaces — across three tiers of embedded hardware, each trained and quantized for its own memory and compute budget rather than sharing a single model. The ESP32 and Jetson tiers run the full 7-class task the project's existing image analysis pipeline produces; the STM32 tier runs a 4-class reduction of the same task, scoped specifically to fit its 20 KB of SRAM.

<br/>

<div align="center">━━━━━━━━━━━━━━ ✦ ✧ ✦ ━━━━━━━━━━━━━━</div>

<br/>

## 🛰️ Deployment Tiers

| Tier | Board | Task | Framework |
|---|---|---|---|
| Ultra-constrained edge | ESP32-WROOM-32 | 7-class land cover | TensorFlow Lite Micro (MicroTFLite) |
| Constrained edge | STM32F103C8T6 | 4-class land cover | X-CUBE-AI |
| Near-edge (planned) | Jetson Nano | Multi-pass land cover tracking | ResNet-18, following [nyc-sentinel](https://github.com/nkasmanoff/nyc-sentinel) |

<br/>

## 🛰️ Engineering Highlights

- **Dataset split by source image, not by tile**, before cropping — tiles from the same photo are visually near-identical, so splitting after tiling would leak near-duplicates across train/val and inflate validation accuracy
- **Fixed an input double-quantization bug** on the ESP32 tier: the test-image generator was pre-shifting pixels into int8 range, then handing that to `ModelSetInput()`, which quantizes internally — the fix passes real normalized float values and lets the library quantize once
- **STM32 task explicitly re-scoped for its hardware**: reduced to 4 classes at 32×32 grayscale after the ESP32's exact architecture was confirmed to exceed the F103C8T6's entire 20 KB SRAM budget on its own
- **Every quantized test image is generated against the model's true scale/zero-point**, read directly from the converted `.tflite` file, rather than a hardcoded shift
- **Inverse-frequency class weighting** during STM32 training, since the merged built-surface class outnumbers water roughly 18 to 1 in the source imagery

<br/>

## 🛰️ Tech Stack

| Layer | Tools |
|---|---|
| Model training | TensorFlow / Keras |
| Quantization | TensorFlow Lite Converter (int8) |
| Dataset labeling | OpenCV, Pillow, NumPy, pandas |
| ESP32 firmware | Arduino, MicroTFLite, TensorFlow Lite Micro |
| ESP32 hardware | ESP32-WROOM-32, SSD1306 OLED (I2C) |
| STM32 firmware | STM32Cube (VS Code extension), X-CUBE-AI, HAL |
| STM32 hardware | STM32F103C8T6 ("Blue Pill"), ST-Link V2 |
| Jetson tier (planned) | PyTorch, ResNet-18, Docker |
| Source dataset | [TU Graz Aerial Semantic Segmentation Drone Dataset](http://dronedataset.icg.tugraz.at/) |

<br/>

## 🛰️ Project Structure
```
edge-ai-implementation/
├── common/
│ ├── data/ # shared TU Graz raw download
│ └── scripts/ # ESP32-tier dataset build, CV pipeline
├── esp32/
│ ├── scripts/ # quantization, header generation
│ ├── models/
│ └── firmware/esp32_land_cover/
├── stm32/
│ ├── data/stm32_dataset/ # separate 4-class grayscale dataset
│ ├── scripts/ # dataset build, training, quantization
│ ├── models/
│ └── cubeide_project/
```


<br/>

## 🛰️ Getting Started

```bash
git clone https://github.com/ktariqq/edge-ai-implementation.git
cd edge-ai-implementation

# Common: build the ESP32/Jetson-tier 7-class dataset
python common/scripts/build_tugraz_dataset.py
python common/scripts/train_tinyml_model.py

# ESP32: quantize and generate firmware headers
python esp32/scripts/convert_to_tflite.py
python esp32/scripts/generate_model_header.py
python esp32/scripts/generate_test_image_header.py

# STM32: separate reduced-class dataset and model
python stm32/scripts/build_stm32_dataset.py
python stm32/scripts/train_stm32_model.py
python stm32/scripts/convert_to_tflite_stm32.py
python stm32/scripts/generate_test_image_header.py
```

<br/>
<br/>

<div align="center">

━━━━━━━━━━━━━━ ✦ ✧ ✦ ━━━━━━━━━━━━━━

Built by **Kommal Tariq**

Copyright © 2026 SpacePoint. All rights reserved.

Training data: TU Graz Aerial Semantic Segmentation Drone Dataset.

</div>
