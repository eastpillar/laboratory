## Download weights
- [Google Driver](https://drive.google.com/file/d/1xaTGwXPhIZBqeDME6X9yexJUEdKmgFQs/view?usp=sharing)

## Dataset
- TinyImageNet_200

## Experiment
- model : DenseNet_121


- setting
  - 
  * Dataset
      1. Image : TinyImageNet
      2. Size : 128 x 128
      3. Train : 207,005
      4. Test : 51,752
      5. Class : 200

  * Augmentation
      1. Random Crop
      2. Random Horizontal Flip

  * HyperParameter
      1. EPOCH : 150
      2. Batch size : 128
      3. Optimizer : SGD
      4. Learning Rate : 0.01
      5. Scheduling : Epoch(50%, 75%) x 0.1 
      6. Loss Function : Cross entropy

## Result

|  Model   |     Dataset      | comp_factor 0.5 acc (val) | Comp_factor 1 acc (val) |
|:--------:|:----------------:|:-------------------------:|:-----------------------:|
| DenseNet | TinyImageNet_200 |          71.48%           |         65.78%          |

<span align="center"><img src="DenseNEt_exp.png"/></span>