<div align="center">
<h1>MLLM-as-a-Judge:
Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark</h1>
<img src="figures/fig1.png">
<img src="figures/Radar.png">
<p align="center">

</p>
This repository is a official code of the research presented in the paper <a href="https://arxiv.org/abs/2401.05952" target='_blank'>[arXiv]</a>. The goal is to provide a transparent, open-source implementation for the community to explore and build upon.
</div>

# Benchmark: MLLM-as-a-Judge
This benchmark is structured into three main components: images, the main dataset, and sub-datasets. The arrangement is as follows:

- **Images**: All images utilized in our study are contained in this section. You can download all images in [google drive](https://drive.google.com/file/d/1z509Wr5f3vXxDbkiCj62mdclEkeMCPx4/view?usp=sharing).

- **MLLM-as-a-Judge**: This part of the dataset is developed in three steps, mirroring the structure outlined in our article. It includes MLLM outputs under three different settings: Scoring Evaluation, Pair Comparison, and Batch Ranking. Additionally, this section encompasses human annotation results and agreement data. In Scoring Evaluation, we also include responses data in a verbose setting for our ablation study.

- **MLLM-as-a-Judge-hard**: This subset focuses on challenging Judge scenarios where current MLLMs tend to underperform or are more susceptible to hallucinations.

- **MLLM-as-a-Judge-HQ**: Contrasting the above, this high-quality dataset includes items where MLLM Judges perform exceptionally well.

Our comprehensive dataset and benchmarks are crafted with the aim of contributing to the development of stronger and more reliable MLLM-as-a-Judge systems in the future.

## Contributing

Contributions to this project are welcome. Please consider the following ways to contribute:

- Reporting issues
- Improving documentation
- Proposing new features or improvements

## Acknowledgements

This project is based on the findings and methodologies presented in the paper [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) and [HallusionBench](https://arxiv.org/abs/2310.14566).

## Citation

```
@misc{gao2024llmasacoauthor,
      title={LLM-as-a-Coauthor: The Challenges of Detecting LLM-Human Mixcase}, 
      author={Chujie Gao and Dongping Chen and Qihui Zhang and Yue Huang and Yao Wan and Lichao Sun},
      year={2024},
      eprint={2401.05952},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```