# Calibration and Simulation of Robot Dynamics in Forced Motion

This repository contains the MATLAB code and Simulink models used in the research project detailed in the thesis: *The Case Study of Calibrating the Dynamics of a Robot in the Vicinity of its Forced Motion*. This project focuses on identifying and recalibrating the parameters of the dynamic model of a robotic system using grey-box modeling techniques and frequency response analysis. The research is demonstrated through simulations and real data collection using the Butterfly Robot developed by Robotikum AB.

## Project Overview

This project aims to develop and calibrate a dynamic model of a robotic system, particularly focusing on the first degree of freedom representing the rotation of the robotic hand. The project is divided into two main studies:

1. **Synthetic Case Study**: Simulated data is used to estimate the parameters of the robot's dynamics.
2. **Butterfly Robot Case Study**: Real experimental data is gathered from the Butterfly Robot, and the system parameters are recalibrated using the model.

### Key Components

- **Synthetic Data Generation**: MATLAB scripts used to generate synthetic data for testing the model.
- **Real Data Analysis**: Scripts to process and analyze experimental data gathered from the Butterfly Robot.
- **Grey-box Modeling**: Simulink models representing the dynamics of the robotic system.

## Files

- `butterfly_real_data.m`: MATLAB script to process real data from the Butterfly Robot experiments. The script reads experimental datasets, processes the data, and computes optimal system parameters based on frequency response analysis. (For details, see the experimental data processing and model recalibration section of the thesis)&#8203;:contentReference[oaicite:0]{index=0}.
  
- `synthetic_data.m`: MATLAB script to generate synthetic datasets for model testing. It simulates different system responses to various input frequencies and helps estimate dynamic parameters using grey-box modeling techniques&#8203;:contentReference[oaicite:1]{index=1}.

- `Simulink_1.slx` and `Simulink_2.slx`: Simulink models used to simulate the robot's dynamic behavior with friction.

## Prerequisites

To run this project, you will need the following:

- MATLAB R2022b or later
- Simulink (version R2022b or later)
- Optimization Toolbox (for parameter estimation)
- Control Systems Toolbox (for frequency response analysis)

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/ArtemShiryaev/Thesis_Butterfly_Robot.git
    cd Thesis_Butterfly_Robot
    ```

2. Make sure MATLAB and Simulink are installed and set up on your machine.

## Usage

### Running the MATLAB Scripts

1. **Synthetic Data Simulation**:
   
   To generate synthetic data and estimate system parameters, run the `synthetic_data.m` script:

    ```matlab
    run('synthetic_data.m')
    ```

2. **Real Data Analysis**:
   
   To analyze the real data collected from the Butterfly Robot, run the `butterfly_real_data.m` script:

    ```matlab
    run('butterfly_real_data.m')
    ```

   This script will process experimental data and compute the optimal frequency responses and system parameters based on the methodology described in the thesis.

### Simulink Models

1. **Open and Simulate the Model**:

   The Simulink model files can be opened and simulated directly in MATLAB. Open the `Simulink_1.slx` or `Simulink_2.slx` in Simulink and run the simulation.

    ```matlab
    open_system('Simulink_1.slx')
    ```

2. **Parameter Adjustment**:

   You can adjust the parameters within the Simulink model to simulate different scenarios of robot dynamics. The parameters such as friction coefficients, motor torques, and system inertia are all configurable within the model.

## Results

The project demonstrates successful recalibration of the robot dynamics using experimental and synthetic data. For detailed results and insights, refer to the thesis document. The recalibration enhances the accuracy of the dynamic model, particularly in the vicinity of forced motion scenarios, and improves the overall performance of the control algorithms applied to the Butterfly Robot.

## Contributions

Feel free to contribute to this project by submitting issues, pull requests, or feature suggestions. Collaboration on extending the dynamic models or improving the calibration algorithms is welcome.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

