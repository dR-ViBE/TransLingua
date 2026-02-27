# 🤟 TransLingua: Real-Time Sign Language Translation Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-green)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-lightblue)
![License](https://img.shields.io/badge/License-MIT-purple)

**TransLingua** is a machine learning-powered computer vision system designed to bridge the communication gap by translating sign language gestures into text in real-time. It features a dual-classification engine capable of interpreting both static characters (alphabets) and dynamic sequences (words) with high confidence and processing speed.

---

## 📑 Table of Contents
1. [Summary](#-summary)
2. [Why this Project?](#-why-this-project)
3. [Design Pattern & Pipeline](#-design-pattern--pipeline)
4. [System Architecture](#-system-architecture)
5. [Key Features](#-key-features)
6. [Tech Stack](#-tech-stack)
7. [Installation & Setup](#-installation--setup)
8. [Project Architecture (Folder Structure)](#-project-architecture)

---

## 🎯 Summary
TransLingua leverages Google's MediaPipe for robust spatial landmark detection and Scikit-Learn for highly optimized inference. By capturing coordinate data directly from hand and pose landmarks, the system categorizes gestures via two distinct pipelines: one for static frames (individual letters) and another for sequential, multi-frame actions (words and phrases). 

## 💡 Why this Project?
For individuals who rely on sign language, daily interactions with non-signers can present significant friction. TransLingua aims to democratize communication accessibility. By using lightweight models and standard webcam inputs rather than expensive proprietary hardware, this project provides a scalable, open-source foundation for real-time gesture-to-text translation.

## ⚙️ Design Pattern & Pipeline
While not utilizing traditional LLM agents, TransLingua relies on a **Dual-Inference Pipeline Pattern** for computer vision:
* **Static Gesture Classifier:** Processes localized frame-by-frame spatial landmarks to output classifications for isolated characters.
* **Dynamic Sequence Classifier:** Processes a time-series window of landmarks to capture temporal dependencies, effectively translating continuous movements into complete words.
* **Performance Monitoring:** Built-in benchmarking dynamically tracks FPS, minimum/maximum inference times, and average prediction confidence for both static and dynamic pipelines.

## 🏗️ System Architecture


1.  **Input Layer:** Standard web camera captures video frames.
2.  **Feature Extraction Layer (MediaPipe):** Extracts specific X, Y, Z coordinates for hand and pose landmarks.
3.  **Preprocessing Layer (NumPy):** Normalizes landmarks and converts them into flattened `.npy` arrays.
4.  **Classification Layer (Scikit-Learn):** Routes array data to either the Static Model or Dynamic Model based on the detection mode.
5.  **Output Layer:** Renders the predicted text and confidence metrics back to the user interface.

## ✨ Key Features
* **Real-Time Processing:** Optimized for high FPS, executing inferences in milliseconds.
* **Comprehensive Data Handling:** Custom data collection logic supporting over 30,000 `.npy` files.
* **Static Translation:** Accurately classifies isolated manual alphabet characters.
* **Dynamic Translation:** Tracks temporal motion sequences to translate up to 10 distinct, continuous words.
* **System Profiling:** Real-time feedback on model performance, including average confidence percentages and inference speeds.

## 🛠️ Tech Stack
* **Programming Language:** Python
* **Computer Vision:** OpenCV, Google MediaPipe
* **Machine Learning:** Scikit-Learn
* **Data Processing:** NumPy, SciPy
* **Environment:** Jupyter Notebook

## 🚀 Installation & Setup

### Prerequisites
Ensure you have Python 3.8+ and `pip` installed on your machine.

### Step-by-Step Guide
1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/dR-ViBE/TransLingua.git](https://github.com/dR-ViBE/TransLingua.git)
   cd TransLingua
