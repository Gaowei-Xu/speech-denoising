# speech-denoising

#### 1. Denoise the test audio data based on pre-trained model
    $ python3 infer.py

---

#### 2. Train the model from scratch 
##### *Step 1*: download trainset and unzip them into `noisy/` and `clean/` directories, respectively under `traindata/` folder.
    
    $ wget -c http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/noisy_trainset_wav.zip
    $ wget -c http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/clean_trainset_wav.zip
    
##### *Step 2*: start to train the model
    
    $ python3 train.py

##### *Step 3*: use your optimal trained model to infer the test data (modify the model parameters configured path in `config.py`)

##### *Step 4*: denoise the test audio data
    $ python3 infer.py

