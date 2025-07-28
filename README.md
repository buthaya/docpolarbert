# DocPolarBERT

**Multimodal (Text + Layout) pre-training for Document AI**
![architecture.png](architecture.png)


## Introduction

DocPolarBERT is a multimodal pre-trained model that combines text and layout information for document understanding tasks. \
It is designed to enhance the performance of various Document AI applications by leveraging both textual content and spatial layout features.\
Notable differences with existing architectures include:
- **No absolute 2D positional encoding**: Instead, it uses relative positional encoding to capture the spatial relationships between text elements.
- **Self-attention with 2D Relative Positional Encoding in Polar Coordinates**: This allows the model to effectively process the layout of documents in a polar coordinate system, which is particularly useful for documents with complex layouts (invoices, financial documents, forms, etc.).
- **No vision feature**: The model does not rely on visual features extracted from images, focusing solely on text and layout information.

For more technical details, see the [DocPolarBERT paper on arXiv](https://arxiv.org/abs/2507.08606).

## Pre-trained Model

| Name           | Huggingface Link                                                                                     |
|----------------|------------------------------------------------------------------------------------------------------|
| DocPolarBERT   | [https://huggingface.co/buthaya/docpolarbert-base](https://huggingface.co/buthaya/docpolarbert-base) |


We pre-trained DocPolarBERT on a mix of 1.8M documents :
- Half are from the [Docile dataset](https://github.com/rossumai/docile),
- The other half are from the [OCR-IDL dataset](https://github.com/furkanbiten/idl_data).

*Why not use the IIT-CDIP dataset ?* \
&rarr; Because the IIT-CDIP documents  **do not come with layout annotations**.
Researchers usually run their own OCR on the images, and then pre-train their models.
This causes different versions of the same dataset to be used by different researchers, which makes it hard to compare results.
Instead, we use data that comes with publicly available layout annotations, so that we can ensure a fair comparison with other models.

## Fine-tuned Models

The model is then fine-tuned and evaluated on the following datasets. \
We provide the pre-processed datasets in the Hugging Face Datasets format, as well as the fine-tuned models, except for Docile which has to be downloaded through the official link.

| Dataset Name                  | 🤗 HF Dataset Link                                                                                   | Official Dataset Link | 🤗 HF Fine-tuned Model Link                                                                              |
|-------------------------------|------------------------------------------------------------------------------------------------------|-----------------------|----------------------------------------------------------------------------------------------------------|
| FUNSD                         | [https://huggingface.co/datasets/buthaya/funsd](https://huggingface.co/datasets/buthaya/funsd)       | [https://guillaumejaume.github.io/FUNSD/](https://guillaumejaume.github.io/FUNSD/) | [https://huggingface.co/buthaya/docpolarbert-funsd](https://huggingface.co/buthaya/docpolarbert-funsd)   |
| SROIE                         | [https://huggingface.co/datasets/buthaya/sroie](https://huggingface.co/datasets/buthaya/sroie)       | [https://rrc.cvc.uab.es/?ch=13](https://rrc.cvc.uab.es/?ch=13) | [https://huggingface.co/buthaya/docpolarbert-sroie](https://huggingface.co/buthaya/docpolarbert-sroie)   |
| CORD (v2)                     | [https://huggingface.co/datasets/buthaya/cord](https://huggingface.co/datasets/buthaya/cord)         | [https://github.com/clovaai/cord](https://github.com/clovaai/cord) | [https://huggingface.co/buthaya/docpolarbert-cord](https://huggingface.co/buthaya/docpolarbert-cord)     |
| Docile's (annotated-trainval) | N/A                                                                                                  | [https://github.com/rossumai/docile](https://github.com/rossumai/docile) | [https://huggingface.co/buthaya/docpolarbert-docile](https://huggingface.co/buthaya/docpolarbert-docile) |
| Payslips                      | [https://huggingface.co/datasets/buthaya/payslips](https://huggingface.co/datasets/buthaya/payslips) | [https://github.com/buthaya/payslips](https://github.com/buthaya/payslips) | [https://huggingface.co/buthaya/docpolarbert-payslips](https://huggingface.co/buthaya/docpolarbert-payslips) |
The Notebooks for fine-tuning and evaluating the model on these datasets are available in the [notebooks](notebooks) directory.

## Citation

```
@misc{uthayasooriyar2025docpolarbertpretrainedmodeldocument,
      title={DocPolarBERT: A Pre-trained Model for Document Understanding with Relative Polar Coordinate Encoding of Layout Structures}, 
      author={Benno Uthayasooriyar and Antoine Ly and Franck Vermet and Caio Corro},
      year={2025},
      eprint={2507.08606},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.08606}, 
}
```
## Contact
For help or issues using DocPolarBERT, please email [Benno Uthayasooriyar](https://github.com/buthaya) or submit a GitHub issue.

## License
This dataset is released under the MIT License. See [LICENSE](LICENSE) for details.

The license allows the commercial use, modification, distribution, and private use of the dataset, provided that the original copyright notice and this permission notice are included in all copies or substantial portions of the dataset.

The license does not provide any warranty, and the dataset is provided "as is".

The copyright notice and the permission notice shall be included in all copies or substantial portions of the dataset.