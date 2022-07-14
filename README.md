# spanish-clinical-flair

Oficial repository for the paper: Clinical Flair: A Pre-Trained Language Model for Spanish Clinical Natural Language Processing

Models: https://drive.google.com/drive/folders/1M1b5FzZqEebTF7B2l58GQvciF4SXP5dT?usp=sharing

Tutorial: https://colab.research.google.com/drive/1AZVPEoEwy13Qv93sjo2yVIMyUHHUz7Lz?usp=sharing


If you use the Clinical Flair model, please cite:

```
@inproceedings{rojas-etal-2022-clinical,
    title = "Clinical Flair: A Pre-Trained Language Model for {S}panish Clinical Natural Language Processing",
    author = "Rojas, Mat{\'\i}as  and
      Dunstan, Jocelyn  and
      Villena, Fabi{\'a}n",
    booktitle = "Proceedings of the 4th Clinical Natural Language Processing Workshop",
    month = jul,
    year = "2022",
    address = "Seattle, WA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.clinicalnlp-1.9",
    pages = "87--92",
    abstract = "Word embeddings have been widely used in Natural Language Processing (NLP) tasks. Although these representations can capture the semantic information of words, they cannot learn the sequence-level semantics. This problem can be handled using contextual word embeddings derived from pre-trained language models, which have contributed to significant improvements in several NLP tasks. Further improvements are achieved when pre-training these models on domain-specific corpora. In this paper, we introduce Clinical Flair, a domain-specific language model trained on Spanish clinical narratives. To validate the quality of the contextual representations retrieved from our model, we tested them on four named entity recognition datasets belonging to the clinical and biomedical domains. Our experiments confirm that incorporating domain-specific embeddings into classical sequence labeling architectures improves model performance dramatically compared to general-domain embeddings, demonstrating the importance of having these resources available.",
}
```
