# Generalisation

**Task**: Evaluate and improve a model’s generalization across custom train/test splits.

Structure:
- `splits/`: Contains datasets or index files defining training and test splits.
- `Matt.py`: Script for training the model and saving checkpoints.
- `saved_model/`: Stores trained model artifacts for reuse.
- `*.ipynb`: Experimental notebooks exploring performance across splits.
- `requirements.txt`: Defines project dependencies.

**How to run**:
1. `pip install -r requirements.txt`
2. Generate splits (e.g., via notebook or script).
3. Run `python Matt.py` to train and save a model.
4. Open notebooks to visualize and assess generalization performance.

**Why it matters**:
Robust generalization is key in real-world ML—this project provides a reproducible framework to test and analyze model behavior across dataset variations.
