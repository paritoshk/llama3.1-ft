import os
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer

# TensorBoard setup
def setup_tensorboard(log_dir: str) -> SummaryWriter:
    """
    Sets up a TensorBoard SummaryWriter for logging training metrics.
    
    Args:
        log_dir: Directory where logs will be stored.
    
    Returns:
        SummaryWriter: Initialized writer for TensorBoard logging.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return SummaryWriter(log_dir)

# Evaluation visualization
def visualize_eval(trainer: Trainer, output_dir: str) -> None:
    """
    Visualizes evaluation results and checkpoints.
    
    Args:
        trainer: The Trainer instance used for training.
        output_dir: Directory where evaluation results are saved.
    """
    print("Evaluating the model on the validation set...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save evaluation metrics
    with open(f"{output_dir}/eval_results.txt", "w") as f:
        f.write(str(eval_results))
    print("Evaluation results saved!")
