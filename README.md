# Adaptive Point Transformer (AdaPT) Model Repository

Welcome to the AdaPT Repository. This repository contains a deep learning model for point cloud classification using transformers and adaptive token dropping.

## Installation

To get started with this project, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/alessandrobaiocchi/AdaPT_reloaded
   ```

2. Navigate to the project directory:

   ```bash
   cd AdaPT_reloaded
   ```

3. Create a Conda environment from the provided `Adapt_env.yaml` file:

   ```bash
   conda env create -f Adapt_env.yaml
   ```

4. Activate the Conda environment:

   ```bash
   conda activate Adapt_env
   ```

## Usage

Once you have set up the Conda environment, you can run the code using the following steps:

1. Ensure you have activated the Conda environment as mentioned in the Installation section.

2. Run the main script:

   ```bash
   python main.py
   ```

   Replace `main.py` with the name of the main script in your project.

## Hyperparameters

This project uses Hydra for hyperparameter management. Edit the `config.yaml` file to change the desired parameters.


## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or create a pull request. For major changes, please open an issue first to discuss the proposed changes.
