# SeqVAE

## Setup

Create a Python environment and install the required dependencies:

```
pip install -r requirements.txt
```

## Training

To run the program, run `seq_train.py`. We currently have 3 explicit parameters: 
- --batch-size: how many batches to run (default: 256)
- --seq-length: how long each sequence should be (default: 50)
- --num-epochs: how many epochs to run (default: 10)

Example command:

```
python seq_train.py --batch-size 512 --seq-length 75 --num-epochs 15
```

