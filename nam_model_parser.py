import argparse

def parse_args():
    
    parser = argparse.ArgumentParser(description="Neural Additive Model Training")

    parser.add_argument("--training_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--output_regularization",type=float, default=0.0, help="feature regularization")
    parser.add_argument("--l2_regularization",type=float, default=0.0, help="l2 weight decay")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")
    parser.add_argument("--feature_dropout", type=float, default=0.0, help="Prob. with which features are dropped")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument("--hidden_units", nargs='+', type=int, default=[], help="Hidden layer sizes.")
    parser.add_argument("--early_stopping_epochs", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--regression", action='store_true', help="True for regression, False for classification.")

    return parser.parse_args()
