# Target-Aware Spatio-Temporal Reasoning via Answering Questions in Dynamic Audio-Visual Scenarios

**Yuanyuan Jiang (jyy55lesley@gmail.com), Jianqin Yin**  
**Beijing University of Posts and Telecommunications**

[\[paper\]](https://aclanthology.org/2023.findings-emnlp.630/)

## Preparation

1. Clone this repo
   ```
   git clone https://github.com/Bravo5542/TJSTG.git
   ```
2. Download data and extract feature
   
   MUSIC-AVQA: [https://gewu-lab.github.io/MUSIC-AVQA/](https://gewu-lab.github.io/MUSIC-AVQA/)

## Training  

```python
python net_tjstg/main.py --mode train
```

## Testing

```python
python net_tjstg/main.py --mode test
```

## Notice

We improve our target-aware process to obtain a more robust performance. The experimental results based on the updated code are as follows:

![image](/net_tjstg/figs/updated_results.png)

## Citation

```ruby
@inproceedings{jiang2023avqa,
  title={Target-Aware Spatio-Temporal Reasoning via Answering Questions in Dynamics Audio-Visual Scenarios},
  author={Jiang, Yuanyuan and Yin, Jianqin},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  year={2023},
  pages = "9399--9409"
}
```



