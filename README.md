# Rovers Navigation DDC Project

Data-driven algorithms will be developed allowing unicycle robots to navigate in uncertain environments so that they: 
(i) reach a desired destination; 
(ii) avoid obstacles. 
In this context, we designed two algorithms that are crucial to enable autonomous navigation: 
(i) a forward data-driven control algorithm to compute the optimal policy; 
(ii) an inverse data-driven control algorithm to estimate, from observations, the robot navigation cost. Moreover, we validated their results by leveraging a state-of-the-art robotics platform: the Robotarium by Georgia Tech. 
The platform offers both a high fidelity simulator and a remotely accessible experimental hardware facility. In their mandatory WPs, students will use the simulator to validate their results. Additionally, by using the Robotarium APIs, students will also have the opportunity to deploy their algorithm on real robots and obtain a hardware validation of their algorithms.

-- This repository contains the Data Driven Control project work of group 2. 

## Contents

- `ddc_project_work_group_2.ipynb`: Main Jupyter notebook file project work.
- `tools` folder: Contains a tool to tune cost functions.
- `deployment` folder: Contains the source code for deployment of experiments the Robotarium platform.
- `img` folder: Contains some simulation images and plots referenced in the Jupyter notebook.

## Requirements
Requirements are listed in the `requirements.txt` file.
