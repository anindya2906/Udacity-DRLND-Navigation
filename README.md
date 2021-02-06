##### Udacity Deep Reinforcement Learning Nano-Degree
# Project 1: Navigation
***

### Introduction
In this project we will train a Deep Q-Network Agent to navigate in a big square world and collect certain kind of items (yellow banana) while avioding another kind of item (blue banana).
On collecting a yellow banana the agent gets a reward of **+1** while collecting a blue banana will get the agent a reward of **-1**. The goal of the agent is to **collect as many yellow banana as possible while avoiding blue banana.**
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
- **0 - move forward.**
- **1 - move backward.**
- **2 - turn left.**
- **3 - turn right.**

The task is episodic and in order to solve the environment **the agent must get an average score of +13 over 100 consecutive episodes.**

### Getting Started
To get started with the project, first we need to download the environment.
You can download the environment from the links given below based on your platform of choice.
- **Linux: [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)**
- **Mac OSX: [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)**
- **Windows (32-bit): [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)**
- **Windows (64-bit): [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)**

Once the file is downloaded, please extract the zip file into the root of the project.

### Setup the python environment
Please create a virtual environment with conda with the following command
```sh
conda create -n drlnd python=3.6
```
Once the environment is created run the follwoing command from the root of the project to install the required packages.
```sh
pip install -r requirements.txt
```
