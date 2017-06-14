# Cloze form QA



### Contents

1. [Installation](#Installation)
2. [Preparation](#preparation)
3. [Training](#Training)
4. [Evaluation](#evaluation)



### Installation

1. This is implemented in PyTorch with Python 2.7. The CUDA version is 8.0.  You can simply run this script.

   ```shell
   pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp27-none-linux_x86_64.whl 
   ```

### Preparation

1. Put the given dataset `train.txt`,  `dev.txt`, 	`test.txt`  in the `data` folder.
2. You can download Glove pretraining embedding [here](https://drive.google.com/drive/folders/0B7aCzQIaRTDUZS1EWlRKMmt3OXM?usp=sharing), and put it in the `data` folder.
3. Make sure some python packages are installed.

### Training

1. Simply run the following script.

   ```shell
   python train.py
   ```

2. If you want to fine-tune your model. You can run the following script.

   ```sh
   python train.py --resume_model=best_model_nlp.pth
   ```

   ​

### Evaluation

1. Simply run the following script. This will create a file named `result.txt`.

   ```shel
   python test.py
   ```

   ​