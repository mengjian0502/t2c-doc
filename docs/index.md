# Home

Torch2Chip is an End-to-end Deep Neural Network compression toolkit designed for prototype accelerator designer for algorithm-hardware co-design with high-degree of algorithm customization.

## News & Update

- **[04/15/2024]:** Initial version of Torch2Chip is published together with the camera-ready version of our **MLSys paper!**

## Why Torch2Chip?

The current "design-and-deploy" workflow faces under-explored challenges in the current hardware-algorithm co-design community due to some unavoidable flaws:

- **Deep Learning framework:** Although the state-of-the-art (SoTA) quantization algorithm can achieve ultra-low precision with negligible degradation of accuracy, the latest deep learning framework (e.g., PyTorch) can **only** support non-customizable 8-bit precision, data format (`torch.qint8`).

- **Algorithms:** Most of the current SoTA algorithm treats the quantized integer as an *intermediate result*, while the final output of the quantizer is the "discretized" floating-point values, ignoring the practical needs and adding additional workload to hardware designers for integer parameter extraction and layer fusion.

- **Industry standard Toolkit:** The compression toolkits designed by the industry (e.g., OpenVino) are constrained to their in-house product or a handful of algorithms. The limited degree of freedom in the current toolkit and the under-explored customization hinder the prototype ASIC-based accelerator.

![Figure1](./assets/Figure1.png)

From the perspectives of the hardware designers, the conflicts from the DL framework, SoTA algorithm, and current toolkits formulate the cumbersome and iterative designation workflow of chip prototyping, **which is what Torch2Chip aim to resolve.**

## What is Torch2Chip?

Torch2Chip is a toolkit that enables **customized** model compression (e.g., quantization) with **full-stack observability** for customized hardware designers. Starting from the user-customized compression algorithms, Torch2Chip perfectly meet the bottom-level needs for the customized AI hardware designers: 

- **[Model and Modules]:** Unlike the open-sourced quantization algorithms, Torch2Chip **does not require** the user to have your customized moule or even model file (e.g., `resnet.py`). 

  

- **[Customize]:** User just need to implement their own algorithm by following the proposed "dual-path" design. Torch2Chip will take care of the reamining step.

  

- **[Decompose & Extract]:** Torch2Chip **decompose** the entire model down to the basic operations (e.g., `matmul`) , where the inputs are compressed by the **user-customized** algorithm.

![Figure1](./assets/torch2chip_workflow.png)



## Authors

Members of [Seo Lab](https://seo.ece.cornell.edu/) @ Cornell University led by Professor Jae-sun Seo.

[Jian Meng](https://mengjian0502.github.io/), Yuan Liao, Anupreetham, Ahmed Hasssan, Shixing Yu, Han-sok Suh, Xiaofeng Hu, and Jae-sun Seo.

## Cite Us
**Publication:** *Torch2Chip: An End-to-end Customizable Deep Neural Network Compression and Deployment Toolkit for Prototype Hardware Accelerator Design* (Meng et al., MLSys, 2024).

### Acknowledgement

This work was supported in part by Samsung Electronics and the Center for the Co-Design of Cognitive Systems (CoCoSys) in JUMP 2.0, a Semiconductor Research Corporation (SRC) Program sponsored by the Defense Advanced Research Projects Agency (DARPA).
