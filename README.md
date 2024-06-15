

## Demo Setup

### 1. 安装套件


安裝過程用到的指令
建立虛擬環境指令：
conda create -n taide python=3.8

進入虛擬環境指令：
conda activate taide

安裝 Pytorch （官網：https://pytorch.org/）

安裝 需要的套件指令：
```
pip install transformers==4.37.0
pip install sentencepiece
pip install protobuf
pip install bitsandbytes
pip install accelerate 
pip install chardet
pip install gradio
pip install datasets
pip install peft
```


### 2. 準備模型權重

至 TAIDE 官方 Hugging Face(https://huggingface.co/taide/TAIDE-LX-7B-Chat) 申請下載權重
將下載好的模型權重和 tokenizer 放入到 `model_weight` 資料夾中


### 3. 執行demo

```python main.py```

