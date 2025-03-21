# Experiment 4 – Sim2Real Transfer

This experiment demonstrates the transfer of policies learned in simulation to real-world robotic systems. It forms the final project for the Robot Learning exam at Politecnico di Torino and is fully detailed in the accompanying project report, [paper_RL_project_Delli_Modi_Necerini.pdf](paper_RL_project_Delli_Modi_Necerini.pdf).

To bridge the simulation–reality gap by training policies in a simulated environment (using [mujoco‑py](https://github.com/openai/mujoco-py)) and transferring them to real robots. The approach leverages techniques such as domain randomization to improve the robustness of the learned policies.
Since we cannot provide the real-world hardware for testing, the real-world deployment results are simulated using the same simulation environment but changing some of the dynamics to mimic real-world uncertainties.

---

## Repository Structure

- `project_Robot_Learning.ipynb`: Jupyter Notebook for interactive experimentation and analysis
- `paper_RL_project_Delli_Modi_Necerini.pdf`: The full project report with detailed descriptions of the methodology and results
- `train.py`: Main training script used to start the learning process
- `env/`: Contains environment definitions and configuration files for the simulation
- `models/`: Contains trained model files
- `plots/`: Holds reward plots generated during training
- `videos/`: Contains renders of the trained policies in simulation environment
- `requirements.txt`: Lists all Python dependencies. (Note: The project requires Python 3.10 or earlier because of compatibility with `mujoco-py`)

---

## Setup and Running the Experiment

### Prerequisites

- The code uses `mujoco-py`, which is compatible with Python 3.10 and earlier.  
- To run on Google Colab, use the fallback runtime environment that supports Python 3.10. (Go to: **Access Tools > Command Palette > Use fallback runtime environment**.)

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/RonPlusSign/RobotLearningLabs.git
   cd RobotLearningLabs/4\ -\ Sim2Real\ transfer
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Training Script

To start training, simply run:
```bash
python train.py
```

For interactive experiments, open and run the `project_Robot_Learning.ipynb` notebook.