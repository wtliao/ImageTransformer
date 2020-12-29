# Image Captioning through Image Transformer

This repository includes the implementation for [Image Captioning through Image Transformer](https://arxiv.org/abs/2004.14231) (to appear in ACCV 2020).

This repo is not completely.

## Requirements

- Python 3.6
- Java 1.8.0
- PyTorch 1.0
- cider (already been added as a submodule)
- coco-caption (already been added as a submodule)
- tensorboardX

Or install full requirements by running:
```bash
pip install -r requirements.txt
```

## TODO
- [x] instruction to prepare dataset
- [ ] remove all unnecessary files
- [ ] add link to download our pre-trained model
- [ ] clean code including comments
- [ ] instruction for training
- [ ] instruction for evaluation

## Training ImageTransformer

### Prepare data

1. We used the preprocessed data from the work [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention). The adaptive ones are used in our work. Please refer to their repo for more information.
2. prepare the hierarchy information (denoted as flag in the code) by running `compute_nb_h.py`. Please modify the file path in Line 58 and the save path in Line 63.
3. You should also preprocess the dataset and get the cache for calculating cider score for [SCST](https://arxiv.org/abs/1612.00563):

```bash
$ python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train
```
### Start training

```bash
$ CUDA_VISIBLE_DEVICES=0 sh train_v3d1.sh
```

See `opts.py` for the options. (You can download the pretrained models from [here]()


### Evaluation

```bash
$ CUDA_VISIBLE_DEVICES=0 python eval.py --model log/log_aoanet_rl/model.pth --infos_path log/log_aoanet_rl/infos_aoanet.pkl  --dump_images 0 --dump_json 1 --num_images -1 --language_eval 1 --beam_size 2 --batch_size 100 --split test
```

### Performance
You will get the scores close to below after training under xe loss for 25 epochs:

## Reference

If you find this repo helpful, please consider citing:

```
@inproceedings{huang2019attention,
  title={Image Captioning through Image Transformer},
  author={He Sen and Liao, Wentong and Tavakoli, Hamed R. and Yang, Michael and Rosenhahn, Bodo and Pugeault, Nicolas},
  booktitle={Asia Conference on Computer Vision},
  year={2020}
}
```

## Acknowledgements

This repository is based on [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch),  and heavily borrow from [AoAnet](https://github.com/husthuaan/AoANet). You may refer to it for more details about the code.
