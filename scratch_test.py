import torch
from model import Model

def test():
    print("Initializing model...")
    model = Model(n_features=2048, batch_size=2)
    model.eval()
    
    print("Creating dummy input...")
    # bs=4 (2 normal, 2 abnormal), ncrops=10, t=32, feature_dim=2048
    dummy_input = torch.rand(4, 10, 32, 2048)
    
    print("Running forward pass...")
    try:
        out = model(dummy_input)
        print("Forward pass successful!")
        print("Scores normal:", out[1].shape)
        print("Scores abnormal:", out[0].shape)
    except Exception as e:
        print("Error during forward pass:", e)

if __name__ == '__main__':
    test()
