# Effect of Pretraining

A core concept behind our finetuning is the idea that a model which is pretrained on more data from an inexpensive level of theory should perform better when trained using data from a higher level.
We test that here by evaluating the performance of models pretrained using differing amounts of TTM data then fine-tuning of DFT.