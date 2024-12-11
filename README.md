# SeqVAE

To run the program, run `seq_train.py`. We currently have 3 explicit parameters: 
- --batch-size: how many batches to run (default: 256)
- --seq-length: how long each sequence should be (default: 50)
- --num-epochs: how many epochs to run (default: 10)

Running such a command would result in a matrix `(n x t x m)` where `n` is batch size, `t` is embed size, and `m` is seq length

## Sample command
```
python seq_train.py --batch-size 512 --seq-length 75 --num-epochs 15
```

