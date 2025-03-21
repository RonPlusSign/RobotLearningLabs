# Robot Learning Laboratories

This repository contains a series of laboratory exercises developed for the Robot Learning course (01HFNOV) at Politecnico di Torino. Each exercise focuses on a different aspect of **Reinforcement Learning (RL)** applied to different environments ([CartPole](https://www.gymlibrary.dev/environments/classic_control/cart_pole/), [Hopper](https://www.gymlibrary.dev/environments/mujoco/hopper/), [Walker2D](https://www.gymlibrary.dev/environments/mujoco/hopper/)), ranging from classical control methods to deep RL approaches and finally to sim-to-real transfer challenges.

The repository is organized into four main subfolders, each corresponding to a separate experiment. Detailed reports (in PDF format) in each folder describe the objectives, methodology, results, and analysis of the experiments.

---

## Repository Structure

```
RobotLearningLabs/
├── 1 - Reinforcement Learning Fundamentals
│   ├── Lab1 Report s331998.pdf
│   ├── 2024_Robot_Learning_Ex1.pdf
│   ├── README.md
│   ├── source code files (agent.py, cartpole_lqr.py, cartpole_rl.py, etc.)
│   ├── models/
│   ├── plots/
│   └── requirements.txt
│
├── 2 - Q-learning
│   ├── Lab2 Report s331998.pdf
│   ├── 2024_Robot_Learning_Ex2.pdf
│   ├── README.md
│   ├── qlearning.py
│   ├── deep_qlearning.py
│   ├── models/
│   ├── plots/
│   └── additional files
│
├── 3 - Policy Gradient algorithms
│   ├── Lab3 Report s331998.pdf
│   ├── 2024_Robot_Learning_Ex3.pdf
│   ├── README.md
│   ├── agent.py
│   ├── cartpole.py
│   ├── cartpole_sb3.py
│   ├── cp_cont.py
│   ├── multiple_cartpoles.py
│   ├── utils.py
│   ├── checkpoints/, models/, monitor/, plots/
│   └── required_python_libs.txt
│
└── 4 - Sim2Real transfer
    ├── 2024_Robot_Learning_Project.pdf
    ├── RL_project_presentation.pdf
    ├── paper_RL_project_Delli_Modi_Necerini.pdf
    ├── project_Robot_Learning.ipynb
    ├── README.md
    ├── train.py
    ├── requirements.txt
    ├── env/, models/, plots/, videos/
    └── .gitignore
```

---

## Experiment Overviews

### 1. Reinforcement Learning Fundamentals

**Goal:** this exercise compares classical control with Reinforcement Learning by tackling the CartPole balancing problem. The experiment uses a Linear Quadratic Regulator (LQR) as a baseline and contrasts it with an RL agent trained via reward-based learning.

**Report Summary:** the report details the cost function setup for LQR, the training process of the RL agent, and a performance comparison between these methods. It discusses the advantages and drawbacks of both approaches (e.g., LQR’s predictability versus RL’s flexibility) and provides insights into parameter selection and stability issues.

**Key Files:**
- `Lab1 Report s331998.pdf`
- `cartpole_lqr.py` (Linear Quadratic Regulator)
- `cartpole_rl.py` (Reinforcement Learning agent)

### 2. Q-learning

**Goal:** this experiment implements two variants of Q-learning to control the CartPole system: Tabular Q-Learning (with discretized state/action spaces) and Deep Q-Learning (using a neural network to approximate the Q-function).

**Report Summary:** the report explains how state discretization, epsilon-greedy policies, and network architectures affect learning performance. It provides a comparative analysis of tabular and deep approaches in terms of convergence speed, stability, and sensitivity to hyperparameters.

**Key Files:**  
- `Lab2 Report s331998.pdf`  
- `qlearning.py` (Tabular approach)  
- `deep_qlearning.py` (Deep Q-Learning implementation)

### 3. Policy Gradient Algorithms

**Goal:** the third experiment explores Policy Gradient methods for solving the CartPole problem. In this lab, the emphasis is on using modern libraries (such as Stable-Baselines3 and gym) to implement and evaluate policy gradient algorithms.

**Report Summary:** the PDF report outlines the requirements (e.g., specific versions of gym and stable-baselines3), describes the training process using both custom and library-provided agents, and reviews performance metrics. The experiment highlights issues such as sample efficiency and the influence of network architecture on the learning process.

**Key Files:**  
- `Lab3 Report s331998.pdf`  
- Source files like `agent.py`, `cartpole.py`, and `cartpole_sb3.py`  
- Supplementary directories (`checkpoints`, `models`, `monitor`, and `plots`) for logging and visualization

### 4. Sim2Real Transfer

**Goal:** the final project deals with transferring policies learned in simulation to real-world (or higher-fidelity simulation) settings—a challenging and critical step in robotics applications. In order to test the reality gap without a phisical robot, we created a simulated environment with a different dynamics from the training one (sim2sim transfer).
The exercise uses the `mujoco-py` library and is meant to run on platforms like Google Colab.

**Report Summary:** the project report describes the overall pipeline from simulation training to real-world deployment. It covers details on the environment setup, the transfer challenges encountered (such as discrepancies between dynamics of different environments), and strategies to mitigate these issues, such as different **domain randomization techniques**. Presentation and additional academic references are included to provide context and validation of the results.

**Key Files:**  
- `paper_RL_project_Delli_Modi_Necerini.pdf` report of the project
- `RL_project_presentation.pdf` presentation slides
- `project_Robot_Learning.ipynb` Jupyter notebook for the project
- `train.py` script for training the policy

This project has been developed in collaboration with my colleagues [Giorgia Modi](https://github.com/GiorgiaModi) and [Ivan Necerini](https://github.com/IvanNece).

---

## How to Get Started

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/RonPlusSign/RobotLearningLabs.git
   cd RobotLearningLabs
   ```

2. **Explore an Experiment:**

   Each experiment folder contains its own README with specific instructions on dependency installation and running the code. For example, to start with the Policy Gradient algorithms:

   ```bash
   cd "3 - Policy Gradient algorithms"
   pip install -r required_python_libs.txt
   # Then run the desired script (e.g., cartpole_sb3.py)
   python cartpole_sb3.py
   ```

3. **Review the Reports:**

   Detailed analysis and results are available in the PDF reports within each folder. These reports provide context on the methodologies, experimental setups, and comparative evaluations.

4. **Sim2Real on Colab:**

   For the Sim2Real experiment, follow the instructions in the README regarding the use of the fallback runtime environment in Colab due to `mujoco-py` compatibility constraints.

---

This repository provides a comprehensive set of experiments that illustrate the progression from basic RL techniques to more advanced applications in robotics, including simulation-to-real-world transfer. Each experiment is self-contained with detailed documentation and report files, making it a valuable resource for learning and further research in Robot Learning.

*For further details, please refer to the respective PDF report files in each subfolder.*