const allProjects = [
  {
    title: "URWKV: Unified RWKV Model with Multi-state Perspective for Low-light Image Restoration",
    description: "This paper, based on RWKV, introduces a Unified Receptance Weighted Key Value (URWKV) model for low-light image restoration, addressing dynamically coupled degradations through a multi-state perspective. The model customizes the RWKV block to perceive complex degradations using intra- and inter-stage states, featuring Luminance-adaptive Normalization (LAN) for scene-aware luminance modulation and a State-aware Selective Fusion (SSF) module for dynamic feature integration. URWKV outperforms state-of-the-art models with fewer parameters and computational resources.",
    date: "2025-05-29",
    tags: "Image",
    img: "images/papers-images/img-20250529-1.png",
    link: "https://arxiv.org/abs/2505.23068"
  },
  {
    title: "RainRWKV: a deep RWKV model for video deraining",
    description: "This paper introduces RainRWKV, a deep RWKV model tailored for video deraining, which enhances low-frequency features using a wavelet transform shift mechanism and captures high-frequency details through a tubelet embedding mechanism. The model achieves state-of-the-art performance on video deraining tasks.",
    date: "2025-05-24",
    tags: "3D/4D",
    img: "images/papers-images/img-20250524-1.png",
    link: "https://link.springer.com/article/10.1007/s00371-025-03965-y"
  },
  {
    title: "DualComp: End-to-End Learning of a Unified Dual-Modality Lossless Compressor",
    description: "This paper proposes DualComp, the first unified and lightweight lossless compressor for both image and text data. Built on the RWKV-7 backbone, it introduces modality-unified tokenization, modality-switching contextual learning, and modality-routing mixture-of-experts to handle modality heterogeneity efficiently. DualComp achieves near real-time inference on desktop CPUs and matches or surpasses SOTA methods with fewer parameters.",
    date: "2025-05-22",
    tags: "General",
    img: "images/papers-images/img-20250522-1.png",
    link: "https://arxiv.org/abs/2505.16256"
  },
  {
    title: "ModRWKV: Transformer Multimodality in Linear Time",
    description: "This paper introduces ModRWKV, a multimodal framework based on the RWKV7 architecture, which achieves efficient multimodal information fusion through dynamically adaptable heterogeneous modality encoders. The framework leverages pretrained RWKV7 weights for initialization, demonstrating competitive performance and computational efficiency compared to traditional Transformer-based multimodal models.",
    date: "2025-05-20",
    tags: "General",
    img: "images/papers-images/img-20250520-1.png",
    link: "https://arxiv.org/abs/2505.14505"
  },
  {
    title: "Quantum-Enhanced Channel Mixing in RWKV Models for Time Series Forecasting",
    description: "This paper introduces QuantumRWKV, a hybrid quantum-classical extension of the RWKV model, replacing the feedforward network with a variational quantum circuit. Experiments on synthetic time-series tasks show quantum-enhanced performance in nonlinear or chaotic dynamics, while classical models excel in tasks with sharp discontinuities.",
    date: "2025-05-18",
    tags: "Sequence",
    img: "images/papers-images/img-20250518-1.png",
    link: "https://arxiv.org/abs/2505.13524"
  },
  {
    title: "Maximizing Asynchronicity in Event-based Neural Networks",
    description: "This paper introduces EVA, a novel asynchronous-to-synchronous (A2S) framework leveraging RWKV-6 architecture for event-based vision tasks. By adapting linear attention and self-supervised learning from NLP, EVA achieves highly expressive and generalizable event-by-event representations, outperforming prior A2S methods on recognition tasks and achieving 47.7 mAP on Gen1 detection tasks.",
    date: "2025-05-16",
    tags: "Image",
    img: "images/papers-images/img-20250516-1.png",
    link: "https://arxiv.org/abs/2505.11165"
  },
  {
    title: "Spatio-Temporal Weighted Graph Reason Learning for Multivariate Time-Series Anomaly Detection",
    description: "This paper introduces the Spatio-Temporal Weighted Graph Reasoning Learning (STWGRL) framework for multivariate time-series anomaly detection in IoT systems. It proposes a D-RWKV module for efficient temporal feature modeling and a TaGAA module for adaptive graph aggregation, achieving high accuracy with low latency and reliability.",
    date: "2025-05-12",
    tags: "Sequence",
    img: "images/papers-images/img-20250512-1.png",
    link: "https://ieeexplore.ieee.org/abstract/document/11002535"
  },
  {
    title: "Multi-View Learning with Context-Guided Receptance for Image Denoising",
    description: "This paper introduces CRWKV, a novel model for real-world image denoising that combines multi-view feature integration with efficient sequence modeling. The proposed approach features a Context-guided Token Shift (CTS) mechanism to capture spatial noise correlations and a Frequency Mix (FMix) module for frequency-domain noise isolation. By implementing a Bidirectional WKV (BiWKV) mechanism, the model achieves full pixel-sequence interaction with linear computational complexity. Experimental results demonstrate superior performance over state-of-the-art methods across multiple datasets while reducing inference time by up to 40%, effectively preserving fine details in complex noise scenarios.",
    date: "2025-05-05",
    tags: "Image",
    img: "images/papers-images/img-20250505-1.png",
    link: "https://arxiv.org/abs/2505.02705"
  },
  {
    title: "RADLADS: Rapid Attention Distillation to Linear Attention Decoders at Scale",
    description: "This paper introduces RADLADS, a method for efficiently converting softmax attention transformers into linear attention decoder models. The approach requires only 350-700M tokens (0.005% of original training data) and under $2,000 USD to convert large models up to 72B parameters while maintaining performance. The authors present new RWKV-variant architectures and demonstrate state-of-the-art results on language benchmarks through a three-step distillation process involving attention alignment, knowledge distillation, and context extension.",
    date: "2025-05-05",
    tags: "General",
    img: "images/papers-images/img-20250505-2.png",
    link: "https://arxiv.org/abs/2505.03005"
  },
  {
    title: "RWKVQuant: Quantizing the RWKV Family with Proxy Guided Hybrid of Scalar and Vector Quantization",
    description: "This paper addresses the challenges of quantizing RWKV models, a modern RNN architecture with Transformer-like performance, for efficient deployment on resource-constrained devices. The authors identify key limitations in existing post-training quantization methods when applied to RWKV, including non-linear operator interference and uniform weight distribution issues. They propose RWKVQuant, a framework combining coarse-to-fine proxy guidance for adaptive scalar/vector quantization selection and codebook optimization for RWKV's unique element-wise multiplication operations. Experimental results demonstrate 3-bit quantization with <1% accuracy loss and 2.14× speedup on RWKV-6-14B, outperforming standalone quantization approaches across language and vision tasks.",
    date: "2025-05-02",
    tags: "General",
    img: "images/papers-images/img-20250502-1.png",
    link: "https://arxiv.org/abs/2505.03803"
  },
  {
    title: "Multiple Span Bidirectional RWKV Network for Infrared Image Super‑Resolution",
    description: "This paper introduces MSB-RWKV, an efficient model for infrared image super-resolution that addresses the computational limitations of Transformers while maintaining global dependency modeling. The proposed method combines a Multiple Span Bidirectional WKV (MSB-WKV) attention mechanism with linear complexity for efficient 2D spatial correlation capture and a Wide Token Shift layer to enhance local context restoration. A prompt projection module further adapts to degradation diversity through learnable visual prompts. Experimental results demonstrate superior performance over state-of-the-art methods in both synthetic and real-world datasets, achieving enhanced detail reconstruction with reduced computational overhead.",
    date: "2025-04-30",
    tags: "Image",
    img: "images/papers-images/img-20250430-2.png",
    link: "https://link.springer.com/article/10.1007/s13042-025-02644-7"
  },
  {
    title: "RWKV-X: A Linear Complexity Hybrid Language Model",
    description: "RWKV-X introduces a hybrid architecture combining RWKV's efficiency for short-range modeling with a sparse attention mechanism for long-range context. It achieves linear-time training complexity and constant-time inference decoding while maintaining performance on both short and long-context tasks. The model demonstrates near-perfect accuracy on 64K-token passkey retrieval and outperforms prior RWKV variants in long-context benchmarks. RWKV-X enables stable decoding up to 1 million tokens, offering a scalable solution for general-purpose language modeling through optimized KV cache management and long-context continual pretraining strategies.",
    date: "2025-04-30",
    tags: "General",
    img: "images/papers-images/img-20250430-1.png",
    link: "https://arxiv.org/abs/2504.21463"
  },
  {
    title: "Zig-RiR: Zigzag RWKV-in-RWKV for Efficient Medical Image Segmentation",
    description: "This paper proposes Zig-RiR, a nested RWKV architecture for efficient medical image segmentation. Addressing the quadratic complexity limitations of transformer-based methods, Zig-RiR combines Outer and Inner RWKV blocks to capture global and local features while maintaining spatial continuity through zigzag scanning. The method treats image patches as 'visual sentences' and sub-patches as 'visual words', enabling linear computational complexity. Experiments on 2D and 3D medical datasets demonstrate 14.4× faster inference and 89.5% reduced GPU memory usage compared to state-of-the-art methods while achieving superior segmentation accuracy.",
    date: "2025-04-17",
    tags: "Image",
    img: "images/papers-images/img-20250417-1.png",
    link: "https://ieeexplore.ieee.org/document/10969076"
  },
  {
    title: "RGB-Event based Pedestrian Attribute Recognition: A Benchmark Dataset and An Asymmetric RWKV Fusion Framework",
    description: "This paper introduces EventPAR, the first large-scale RGB-Event pedestrian attribute recognition dataset containing 100K aligned samples with 50 attributes spanning appearance and emotional dimensions. To address RGB camera limitations in challenging conditions, the authors propose an RWKV-based framework featuring asymmetric fusion of spatial RGB features and temporal event data through similarity-based token filtering. The method achieves state-of-the-art performance on three datasets, demonstrating improved robustness through multi-modal fusion while maintaining computational efficiency via linear attention mechanisms.",
    date: "2025-04-14",
    tags: "Image",
    img: "images/papers-images/img-20250414-1.png",
    link: "https://arxiv.org/abs/2504.10018"
  },
  {
    title: "MolRWKV: Conditional Molecular Generation Model Using Local Enhancement and Graph Enhancement",
    description:"This paper introduces MolRWKV, a conditional molecular generation model built upon the RWKV architecture. It leverages RWKV's efficient sequence processing capabilities (combining RNN efficiency and Transformer parallelism) for handling SMILES strings. To enhance performance for chemical tasks, MolRWKV integrates CNN for local sequence features and GCN for graph-based scaffold information. Experiments show this RWKV-based model achieves comparable or improved results versus baselines in generating molecules under specific conditions, demonstrating the potential of the RWKV architecture when adapted and enhanced for the molecular domain.",
    date: "2025-04-10",
    tags: "Sequence",
    img: "images/papers-images/img-20250410-1.png",
    link: "https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.70100"
  }, 
  {
    title: "Kinematic Modeling of a 7-DOF Tendon-Like-Driven Robot Based on Optimization and Deep Learning",
    description: "This paper presents a 7-DOF tendon-driven redundant robot (TDR7) using a weighted inverse kinematics optimization algorithm (SWGPM-TDR7) and a deep learning fine-tuning model (RWKV-TDR7). The SWGPM-TDR7 integrates joint constraints, singularity avoidance, and energy minimization for efficient trajectory planning, while RWKV-TDR7 combines recurrent networks and self-attention mechanisms to reduce computational complexity in trajectory fitting. Experimental results demonstrate high accuracy in forward/inverse kinematics and trajectory tracking, offering solutions for medical and industrial robotic systems requiring flexible, stable motion control.",
    date: "2025-04-07",
    tags: "3D/4D",
    img: "images/papers-images/img-20250407-1.png",
    link: "https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.22544"
  },
  {
    title: "DREMnet: An Interpretable Denoising Framework for Semi-Airborne Transient Electromagnetic Signal",
    description:" The paper utilizes the RWKV architecture for data processing, introducing a context-WKV mechanism and implementing bidirectional signal modeling. By stacking embeddings, it preserves the powerful local perception capabilities of convolutional networks. Experimental results on test datasets demonstrate that the DREMnet method outperforms existing techniques, with processed field data more accurately reflecting theoretical signals and enhancing the ability to identify underground electrical structures.",
    date: "2025-03-28",
    tags: "Sequence",
    img: "images/papers-images/img-20250328-1.png",
    link: "https://arxiv.org/abs/2503.22223"
  }, 
  {
    title: "Geometry-Aware RWKV for Heterogeneous Light Field Spatial Super-Resolution",
    description: " The paper designs a texture transfer module with channel correlation and a spatial angle correction module based on RWKV. Additionally, it employs a geometry-aware RWKV to capture the intrinsic collective structure of squares. Experimental results demonstrate that the proposed method outperforms state-of-the-art approaches in both quantitative and qualitative comparisons, while achieving greater efficiency in terms of inference time and memory usage.",
    date: "2025-03-27",
    tags: "Image",
    img: "images/papers-images/img-20250327-1.png",
    link: "https://ieeexplore.ieee.org/abstract/document/10943155"
  }, 
  {
    title: "RSRWKV: A Linear-Complexity 2D Attention Mechanism for Efficient Remote Sensing Vision Task",
    description: "The paper proposes RSRWKV, which features a novel two-dimensional WKV scanning mechanism that connects sequence processing with two-dimensional spatial reasoning while maintaining linear complexity. It achieves multi-directional isotropic context aggregation. Experimental results demonstrate that RSRWKV outperforms convolutional neural networks and Transformer baselines on classification, detection, and segmentation tasks across various datasets, providing a scalable solution for high-resolution remote sensing analysis.",
    date: "2025-03-26",
    tags: "Image",
    img: "images/papers-images/img-20250326-1.png",
    link: "https://arxiv.org/abs/2503.20382"
  }, 
  {
    title: "RWKV-7 \"Goose\" with Expressive Dynamic State Evolution",
    description: "The paper proposes RWKV-7 \"Goose,\" a novel sequence modeling architecture that achieves state-of-the-art performance in multilingual tasks at the 3 billion parameter scale, matching top English models with significantly fewer training tokens. RWKV-7 requires only constant memory and computation per token during inference, enabling efficient state tracking and recognition of all regular languages. It surpasses Transformer capabilities under standard complexity conjectures and demonstrates strong performance on long-context tasks. The paper also releases a 3.1 trillion token multilingual corpus and pre-trained models ranging from 0.19B to 2.9B parameters, showcasing RWKV-7's scalability and efficiency.",
    date: "2025-03-19",
    tags: "General",
    img: "images/papers-images/img-20250319-1.png",
    link: "https://arxiv.org/abs/2503.14456"
  }, 
  {
    title: "CMGN: Text GNN and RWKV MLP-mixer combined with cross-feature fusion for fake news detection",
    description: "he paper proposes a novel cross-feature fusion network, CMGN, combining Text Graph Neural Networks (GNN) and RWKV MLP-mixer for fake news detection. The RWKV MLP-mixer processes news text by replacing self-attention with MLP layers to capture deep semantic features, while Text GNN models relationships among supplementary texts (e.g., titles, locations) as graph nodes. A cross-feature fusion mechanism integrates these features dynamically. Evaluated on LIAR, FA-KES, IFND, and CHEF datasets, CMGN outperforms existing methods, demonstrating enhanced accuracy. Focal loss addresses class imbalance, and ablation studies confirm RWKV's critical role in feature extraction. The model advances fake news detection by synergizing graph-based relational modeling and efficient text-sequence processing via RWKV.",
    date: "2025-03-12",
    tags: "Language",
    img: "images/papers-images/img-20250312-1.png",
    link: "https://www.sciencedirect.com/science/article/abs/pii/S0925231225004837"
  },  
  {
    title: "Linear attention based spatiotemporal multi graph GCN for traffic flow prediction",
    description: "The paper proposes LASTGCN, a deep learning model for traffic flow prediction, integrating a Multi-Factor Fusion Unit (MFF-unit) to dynamically merge meteorological data, a multi-graph convolutional network for spatial correlations, and the Receptance Weighted Key Value (RWKV) block. The RWKV mechanism replaces traditional Transformer attention with linear attention, reducing computational complexity while efficiently capturing long-term dependencies in traffic sequences. By combining RWKV's parallelizable training and RNN-like inference, the model achieves high efficiency for mid-term traffic management. Experiments on real-world datasets (PeMSD) demonstrate superior accuracy and robustness, especially for long-term predictions, outperforming state-of-the-art methods. External factors like weather integration further enhance performance.",
    date: "2025-03-10",
    tags: "Sequence",
    img: "images/papers-images/img-20250310-1.png",
    link: "https://www.nature.com/articles/s41598-025-93179-y"
  },
  {
    title: "BlackGoose Rimer: Harnessing RWKV-7 as a Simple yet Superior Replacement for Transformers in Large-Scale Time Series Modeling",
    description: "This paper introduces the integration of the RWKV-7 architecture into the Timer model for time series modeling. By leveraging its time mix and channel mix components, the proposed method achieves significant performance improvements of 1.13x to 43.3x with a 4.5x reduction in training time using only 1/23 of the original parameters.",
    date: "2025-03-08",
    tags: "Sequence",
    img: "images/papers-images/img-20250308-1.png",
    link: "https://arxiv.org/abs/2503.06121"
  },
  {
    title: "Toward Comprehensive Semantic Prompt for Region Contrastive Learning Underwater Image Enhancement",
    description: "The paper proposes SRCNet, an underwater image enhancement network integrating semantic guidance and region contrastive learning. The method introduces a semantic-aware RWKV block that leverages the global perception capability of RWKV architecture while incorporating semantic prompts to preserve regional color consistency and structural details. By combining RWKV's efficient attention mechanism with semantic-aware constraints, the network reduces interference from irrelevant pixels across different underwater regions. A novel region contrastive learning strategy further enhances degradation-sensitive feature learning through multi-perspective negative sample utilization. Experimental results demonstrate superior performance over state-of-the-art methods in restoring color accuracy and detail clarity for underwater images.",
    date: "2025-03-07",
    tags: "Image",
    img: "images/papers-images/img-20250307-5.png",
    link: "https://ieeexplore.ieee.org/abstract/document/10888780"
  },
  {
    title: "ID-RWKV: Image Deraining RWKV",
    description: "The paper proposes ID-RWKV, a novel image deraining framework leveraging the Receptance Weighted Key Value (RWKV) architecture to address Transformer's quadratic complexity limitations. By replacing self-attention with linear-complexity RWKV blocks, the model efficiently captures local-global dependencies through LG-WKV mechanisms in U-shaped networks. It introduces a multi-stage progressive deraining strategy with Fourier enhancement modules and deep-shallow feature fusion (DSFFM) to preserve background details. Experiments show ID-RWKV outperforms state-of-the-art Transformer-based methods on synthetic and real datasets while using fewer parameters (12.38M) and lower computation (60.2G FLOPs), demonstrating RWKV's potential in 2D vision tasks.",
    date: "2025-03-07",
    tags: "Image",
    img: "images/papers-images/img-20250307-3.png",
    link: "https://ieeexplore.ieee.org/abstract/document/10889384"
  },
  {
    title: "HFE-RWKV: High-Frequency Enhanced RWKV Model for Efficient Left Ventricle Segmentation in Pediatric Echocardiograms",
    description: "The paper proposes High-Frequency Enhanced RWKV (HFE-RWKV), a novel model integrating RWKV's efficient recurrent architecture with high-frequency feature enhancement for pediatric left ventricle segmentation in echocardiograms. By redesigning RWKV's spatial mixing module to explicitly amplify boundary-related high-frequency components and introducing space-frequency consistency loss, the model achieves superior shape-aware segmentation while maintaining computational efficiency. Compared to U-Mamba, HFE-RWKV improves Dice scores by 2% using only 67% parameters and 26% computational costs, demonstrating RWKV's adaptability for medical imaging tasks requiring both precision and resource efficiency.",
    date: "2025-03-07",
    tags: "Image",
    img: "images/papers-images/img-20250307-2.png",
    link: "https://ieeexplore.ieee.org/abstract/document/10888300"
  },
  {
    title: "RWKVMatch: Vision RWKV-based Multi-scale Feature Matching Network for Unsupervised Deformable Medical Image Registration",
    description: "The paper proposes RWKVMatch, a Vision-RWKV-based deformable medical image registration framework combining global attention and cross-fusion mechanisms. By extending RWKV to 3D Vision-RWKV blocks, it effectively captures spatial features in volumetric medical images while maintaining linear computational complexity. The cross-fusion blocks enable inter-image feature integration, complemented by elastic transformation augmentation to improve robustness. Evaluated on brain MRI datasets (LPBA40/IXI), RWKVMatch achieves state-of-the-art performance with 0.704 DSC on LPBA40 and reduced foldings (0.154% negative Jacobian), demonstrating RWKV's superiority in balancing registration accuracy and efficiency compared to CNN/Transformer/Mamba-based methods.",
    date: "2025-03-07",
    tags: "Image",
    img: "images/papers-images/img-20250307-4.png",
    link: "https://ieeexplore.ieee.org/abstract/document/10888484"
  },
  {
    title: "Flare-Aware RWKV for Flare Removal",
    description: "The paper proposes Flare-RWKV, a novel RWKV-based architecture for lens flare removal in images. By integrating a lightweight flare detection network with a restoration network built upon RWKV's efficient scanning mechanism (capturing global dependencies with linear complexity) and token shift mechanism (enhancing local context awareness), the method addresses flare-specific challenges. Key innovations include Flare-Aware Feature Selection (FAFS) that prioritizes background reconstruction using detected flare masks. Compared to UNet and transformer variants, Flare-RWKV demonstrates superior performance on synthetic and real-world datasets while maintaining parameter efficiency, establishing RWKV's effectiveness in flare removal tasks.",
    date: "2025-03-07",
    tags: "Image",
    img: "images/papers-images/img-20250307-1.png",
    link: "https://ieeexplore.ieee.org/abstract/document/10888487"
  },
  {
    title: "PathRWKV: Enabling Whole Slide Prediction with Recurrent-Transformer",
    description: "The paper proposes PathRWKV, a novel Recurrent-Transformer hybrid model for whole slide image (WSI) analysis in computational pathology. To address challenges in handling variable tile scales, model complexity, and training-inference trade-offs, PathRWKV integrates a dynamic recurrent structure for full-slide processing and adopts RWKV’s linear attention mechanism to reduce computational costs and mitigate overfitting. Multi-task learning jointly optimizes heterogeneous clinical indicators, improving training efficiency, while an asynchronous inference design enables sequential processing of all tiles during prediction. Evaluated across seven WSI datasets, PathRWKV achieves state-of-the-art performance in cancer subtyping, metastasis detection, and survival prediction, demonstrating superior generalization and scalability in pathology applications.",
    date: "2025-03-05",
    tags: "Image",
    img: "images/papers-images/img-20250305-1.png",
    link: "https://arxiv.org/abs/2503.03199"
  },
  {
    title: "Delta-WKV: A Novel Meta-in-Context Learner for MRI Super-Resolution",
    description: "The paper proposes Delta-WKV, a novel linear Transformer model for MRI super-resolution, integrating Meta-in-Context Learning (MiCL) and the Delta rule to dynamically adjust weights during inference for efficient local-global pattern recognition. Inspired by RWKV, Delta-WKV employs a quad-directional scanning mechanism and replaces traditional MLPs with a channel-mixing network, enhancing long-range dependency capture while preserving high-frequency details. Evaluated on IXI and fastMRI datasets, Delta-WKV achieves state-of-the-art PSNR/SSIM scores with 15% faster training and inference than SwinIR and MambaIR, demonstrating efficiency for clinical applications.",
    date: "2025-02-28",
    tags: "Image",
    img: "images/papers-images/img-20250228-1.png",
    link: "https://arxiv.org/abs/2502.20852"
  },
  {
    title: "TabulaTime: A Novel Multimodal Deep Learning Framework for Advancing Acute Coronary Syndrome Prediction through Environmental and Clinical Data Integration",
    description: "The paper proposes TabulaTime, a novel multimodal deep learning framework integrating clinical and environmental time-series data to improve Acute Coronary Syndrome (ACS) prediction. Key innovations include the PatchRWKV module, which combines recurrent neural networks (RNNs) and attention mechanisms for efficient time-series feature extraction with linear computational complexity. This module outperforms state-of-the-art models (e.g., Transformers, LSTMs) in capturing temporal dependencies. Experimental results show a 20.5% accuracy improvement over traditional methods, highlighting the significance of integrating air pollution data. The framework enhances interpretability through attention mechanisms, identifying critical predictors like systolic blood pressure and PM₁₀.",
    date: "2025-02-24",
    tags: "Sequence",
    img: "images/papers-images/img-20250224-1.png",
    link: "https://arxiv.org/abs/2502.17049v1"
  },
  {
    title: "Rwkv-vg: visual grounding with RWKV-driven encoder-decoder framework",
    description: "The paper proposes RWKV-VG, a novel visual grounding framework entirely built on the RWKV architecture. Unlike traditional CNN- or Transformer-based approaches, RWKV-VG leverages RWKV’s hybrid design, which combines RNN-like sequential processing and Transformer-like attention, to efficiently model intra-modal and cross-modal interactions. The framework employs RWKV-driven visual and linguistic encoders, a visual-linguistic decoder, and a learnable [REG] token for box regression. Evaluations on ReferItGame and RefCOCO benchmarks demonstrate state-of-the-art performance, surpassing Transformer-based methods like TransVG in accuracy and convergence speed. Ablation studies highlight the critical role of RWKV modules and the [REG] token placement. This work establishes RWKV as a competitive architecture for vision-language tasks, offering computational efficiency without sacrificing precision.",
    date: "2025-02-21",
    tags: "Image",
    img: "images/papers-images/img-20250221-1.png",
    link: "https://link.springer.com/article/10.1007/s00530-025-01720-w"
  },
  {
    title: "Substation equipment non-rigid defect detection via receptance weighted key value-based causality-aware networks",
    description: "The paper proposes a causal - aware equipment defect detection framework based on the RWKV architecture to address non - rigid defect detection and long - tailed distribution issues in substation equipment. The RWKV architecture, with its global receptive field, enhances defect feature extraction. It's integrated with other modules in the framework. Experiments show this framework outperforms baseline methods, validating its effectiveness.",
    date: "2025-02-13",
    tags: "Image",
    img: "images/papers-images/img-20250213-1.png",
    link: "https://link.springer.com/article/10.1007/s11760-025-03852-y"
  },
  {
    title: "Linear Attention Modeling for Learned Image Compression",
    description: "The paper proposes LALIC, a linear attention-based learned image compression framework utilizing Bi-RWKV blocks for efficient feature extraction. By integrating bidirectional RWKV (BiWKV) attention and Omni-Shift modules, LALIC captures global dependencies and local context in 2D latent representations with linear complexity. A novel RWKV-based Spatial-Channel Context Model (RWKV-SCCTX) further enhances entropy modeling by exploiting spatial and channel redundancies. Experiments demonstrate that LALIC outperforms VTM-9.1 by up to -17.32% in BD-rate across Kodak, Tecnick, and CLIC datasets, achieving competitive rate-distortion performance with lower computational overhead compared to transformer-based methods. This work highlights RWKV's effectiveness in balancing efficiency and compression quality for high-resolution images.",
    date: "2025-02-09",
    tags: "Image",
    img: "images/papers-images/img-20250209-2.png",
    link: "https://arxiv.org/abs/2502.05741"
  },
  {
    title: "Training Language Models for Social Deduction with Multi-Agent Reinforcement Learning",
    description: "The paper proposes training language models for social deduction games using multi-agent reinforcement learning (MARL), focusing on natural language communication without human demonstrations. By integrating 'istening' (predicting imposters from discussions) and 'speaking' (rewarding messages that shift others' beliefs), the framework employs the RWKV model—a recurrent architecture with linear attention—to efficiently handle long gameplay sequences and reduce computational overhead. Results show RWKV-based agents outperform standard RL methods, doubling win rates and exhibiting human-like strategies such as evidence-based accusations. The choice of RWKV addresses challenges in scalability and context length, critical for real-time multi-agent interactions.",
    date: "2025-02-09",
    tags: "Language",
    img: "images/papers-images/img-20250209-1.png",
    link: "https://arxiv.org/abs/2502.06060"
  },
  {
    title: "RWKV-UI: UI Understanding with Enhanced Perception and Reasoning",
    description: "The paper proposes the RWKV-UI, a visual language model based on the RWKV architecture, designed for high-resolution UI understanding. It addresses information loss and reasoning limitations in existing VLMs by integrating three visual encoders (SIGLIP, DINO, SAM) with a partition-encoding strategy to process 4096×4096 UI images while preserving details. Leveraging RWKV’s efficient RNN-based structure, the model combines layout detection and Chain-of-Thought (CoT) visual prompts to enhance spatial reasoning and multi-step interaction prediction. Experiments demonstrate superior performance on UI tasks, outperforming larger models in action grounding and element recognition. RWKV-UI highlights RWKV’s adaptability in multimodal scenarios through efficient feature fusion and reasoning mechanisms.",
    date: "2025-02-06",
    tags: "Image",
    img: "images/papers-images/img-20250206-1.png",
    link: "https://arxiv.org/abs/2502.03971"
  },
  {
    title: "Multi-Modal Dynamic Brain Graph Representation Learning for Brain Disorder Diagnosis Via Temporal Sequence Model",
    description: "The paper proposes the ET_MGNN model for brain disorder diagnosis. It integrates multimodal brain network information and uses RWKV for dynamic sequence modeling. By fusing structural and functional connectivity, the model can capture complex brain network features. Experiments on datasets like ABIDE II and ADNI show that ET_MGNN outperforms other methods, and RWKV plays a crucial role in improving performance.",
    date: "2025-02-05",
    tags: "Sequence",
    img: "images/papers-images/img-20250205-1.png",
    link: "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5114041"
  },
  {
    title: "Exploring Linear Attention Alternative for Single Image Super-Resolution",
    description: "The paper proposes the OmniRWKVSR model for single-image super-resolution, integrating the Receptance Weighted Key Value (RWKV) architecture with novel feature extraction techniques (VRSM and VRCM) to address computational complexity and reconstruction quality. By leveraging RWKV's linear computational efficiency and hybrid RNN-Transformer strengths, the model avoids quadratic attention costs while enhancing multi-scale feature capture. Experimental results demonstrate superior performance over MambaIR and SwinIR, achieving 0.26% PSNR and 0.16% SSIM improvements in 4× upscaling tasks, along with 15% faster training. The work highlights RWKV's effectiveness in balancing efficiency and image restoration quality, particularly for remote sensing applications.",
    date: "2025-02-01",
    tags: "Image",
    img: "images/papers-images/img-20250201-1.png",
    link: "https://arxiv.org/abs/2502.00404"
  },
  {
    title: "RWKV-Lite: Deeply Compressed RWKV for Resource-Constrained Devices",
    description: "The paper proposes the RWKV-Lite, a suite of compression techniques tailored for the RNN-based RWKV architecture to enable efficient deployment on resource-constrained devices. By combining low-rank approximations for projection matrices, sparsity-aware predictors for FFN layers, embedding caching, and hierarchical weight decomposition for classification heads, the approach reduces RWKV's memory footprint by 3.4–5× with minimal accuracy loss. Compared to similarly accurate transformer-based models, RWKV-Lite achieves 4× lower memory usage while retaining computational efficiency. The work highlights RWKV's adaptability to compression and its potential as a lightweight alternative to transformers for edge applications.",
    date: "2025-01-31",
    tags: "General",
    img: "images/papers-images/img-20250131-1.png",
    link: "https://arxiv.org/abs/2412.10856"
  },
  {
    title: "ARWKV: Pretrain is not what we need, an RNN-Attention-Based Language Model Born from Transformer",
    description: "The paper proposes ARWKV, an RNN-attention-based language model derived from the RWKV architecture, aiming to enhance expressiveness and state-tracking capabilities beyond transformers. By distilling knowledge from transformer-based models like Qwen2.5 into RNNs, ARWKV replaces self-attention with the RWKV-7 time-mixing module, enabling efficient training on limited resources (e.g., a 7B model on a single A100 GPU). The method involves three stages: attention alignment, knowledge distillation, and supervised fine-tuning. Evaluations show competitive performance on benchmarks, though architectural mismatches between teacher-student scales may degrade results. The work bridges transformer efficiency with RNN strengths, highlighting RWKV’s potential for hybrid architectures.",
    date: "2025-01-26",
    tags: "General",
    img: "images/papers-images/img-20250126-1.png",
    link: "https://arxiv.org/abs/2501.15570"
  },
  {
    title: "Rate-Aware Learned Speech Compression",
    description: "This paper proposes a learning-based speech compression scheme based on a channel-aware entropy model, which enhances rate-distortion performance by replacing traditional quantizers. It utilizes multi-scale convolutions and hybrid RWKV blocks to improve the representational capacity of both encoder and decoder. Experimental results demonstrate that the proposed method achieves significant improvements in bitrate savings and acoustic quality metrics compared to existing codecs. This research finding has important implications for addressing speech compression in real-time communication and provides new insights and directions for future research.",
    date: "2025-01-21",
    tags: "Audio",
    img: "images/papers-images/img-20250121-1.png",
    link: "https://arxiv.org/abs/2501.11999"
  },
  {
    title: "Learnable Sparsification of Die-to-Die Communication via Spike-Based Encoding",
    description: "The paper proposes SNAP, a hybrid neural network architecture that combines SNNs and ANNs. To evaluate SNAP, RWKV is integrated as a representative language model architecture. Experiments show that SNAP outperforms traditional SNNs and non-spiking models, achieving up to 5.3× energy efficiency improvements and 15.2× reductions in inference latency, highlighting its potential in large-scale AI systems.",
    date: "2025-01-15",
    tags: "General",
    img: "images/papers-images/img-20250115-1.png",
    link: "https://arxiv.org/abs/2501.08645"
  },
  {
    title: "RWKV-UNet: Improving UNet with Long-Range Cooperation for Effective Medical Image Segmentation",
    description: "The paper proposes RWKV-UNet, which integrates the RWKV structure into U-Net for medical image segmentation. The IR-RWKV module enhances the ability to capture long-range dependencies, and combined with the CCM module, it improves skip connections. Experiments show that it achieves SOTA performance on multiple datasets, and its variants balance performance and efficiency.",
    date: "2025-01-14",
    tags: "Image",
    img: "images/papers-images/img-20250114-1.png",
    link: "https://arxiv.org/abs/2501.08458"
  },
  {
    title: "ChemRB: a novel generative model based on bidirectional molecular ring constraints",
    description: "The paper proposes ChemRB, a novel generative model for molecular design in drug discovery, leveraging bidirectional molecular ring constraints to address limitations in existing unidirectional encoders. By integrating the RWKV mechanism, ChemRB combines the linear computational efficiency of RNNs with the contextual awareness of Transformers, effectively capturing long-range dependencies in SMILES sequences. The model introduces two pre-training tasks—ring-level feature prediction and global-span closure prediction—to enhance molecular validity, particularly for complex ring systems. Experimental results demonstrate ChemRB's superior performance in generating valid, unique, and novel molecules, outperforming state-of-the-art models on benchmark datasets. Additionally, its application to EGFR inhibitor redesign highlights practical utility, showcasing high binding affinity and structural fidelity.",
    date: "2025-01-10",
    tags: "Sequence",
    img: "images/papers-images/img-20250110-1.png",
    link: "https://jsnu.magtech.com.cn/CN/10.15983/j.cnki.jsnu.2025005"
  },
  {
    title: "Explore Activation Sparsity in Recurrent LLMs for Energy-Efficient Neuromorphic Computing",
    description: "The paper proposes a low - cost, training - free algorithm to sparsify Recurrent LLMs' activations for energy - efficient neuromorphic computing. It takes RWKV as an example to show the effectiveness of the method. By adding thresholding functions in RWKV, the average activation sparsity is increased. Hardware simulations show significant energy savings and latency improvements, and the method can also be extended to other models.",
    date: "2025-01-09",
    tags: "General",
    img: "images/papers-images/img-20250109-1.png",
    link: "https://arxiv.org/abs/2501.16337"
  },
  {
    title: "Reducing Cross-Sensor Domain Gaps in Tactile Sensing via Few-Sample-Driven Style-to-Content Unsupervised Domain Adaptation",
    description: "The paper proposes a few-sample-driven style-to-content unsupervised domain adaptation method FSSC to reduce cross-sensor domain gaps in tactile sensing. In the design of the bottleneck layer, modules are integrated based on the RWKV architecture to extract and fuse spatio-temporal information, improving the model performance, and experiments prove its effectiveness.",
    date: "2025-01-05",
    tags: "3D/4D",
    img: "images/papers-images/img-20250105-1.png",
    link: "https://www.mdpi.com/1424-8220/25/1/256"
  },
  {
    title: "Efficient Relational Context Perception for Knowledge Graph Completion",
    description: "The paper proposes a novel method for knowledge graph completion integrating the Triple Receptance Perception (TRP) architecture and Tucker decomposition module. Inspired by Rwkv, the TRP effectively models sequential information through time and channel mixing blocks to learn dynamic embeddings. Experiments show that this method outperforms existing models.",
    date: "2024-12-31",
    tags: "Language",
    img: "images/papers-images/img-20241231-1.png",
    link: "https://arxiv.org/abs/2501.00397"
  },
  {
    title: "Injecting Explainability and Lightweight Design into Weakly Supervised Video Anomaly Detection Systems",
    description: "This paper introduces TCVADS, a system for video anomaly detection. It operates in two stages. In the first stage, it employs an enhanced RWKV module for efficient time series analysis. Through knowledge distillation and cross-modal learning, it achieves better performance than existing methods.",
    date: "2024-12-28",
    tags: "3D/4D",
    img: "images/papers-images/img-20241228-1.png",
    link: "https://arxiv.org/abs/2412.20201"
  },
  {
    title: "StyleRWKV: High-Quality and High-Efficiency Style Transfer with RWKV-like Architecture",
    description: "StyleRWKV, a new style transfer method. It adopts an architecture inspired by RWKV to resolve the shortcomings of previous approaches, such as high computational complexity. By means of crucial elements like the Re-WKV attention mechanism, it accomplishes efficient and high-quality style transfer.",
    date: "2024-12-27",
    tags: "Image",
    img: "images/papers-images/img-20241227-1.png",
    link: "https://arxiv.org/abs/2412.19535"
  },
  {
    title: "L3TC: Leveraging RWKV for Learned Lossless Low-Complexity Text Compression",
    description: "L3TC, a novel text compression method. It selects RWKV for its fast decoding speed. With an outlier-aware tokenizer and high-rank reparameterization, L3TC achieves 48% bit saving vs gzip, 50× param reduction, and is the fastest learned compressor.",
    date: "2024-12-21",
    tags: "Language",
    img: "images/papers-images/img-20241221-1.png",
    link: "https://arxiv.org/abs/2412.16642"
  },
  {
    title: "A Survey of RWKV",
    description: "A collection of papers and resources related to a survey of RWKV.",
    date: "2024-12-19",
    tags: "General",
    img: "images/papers-images/img-20241219-1.png",
    link: "https://github.com/MLGroupJLU/RWKV-Survey"
    },
  {
    title: "PCF-RWKV: Product Carbon Footprint Estimation System Based on Large Language Model",
    description: "PCF-RWKV, a product carbon footprint assessment model built on the RWKV architecture, featuring stacked residual blocks and three task-specific LoRA adapters. Through Multi-Agents technology integration, the model automates LCI construction for production processes and matches them with emission factors to calculate carbon footprints, enhancing efficiency and security in enterprise carbon footprint assessment while overcoming traditional method limitations.",
    date: "2024-12-18",
    tags: "Language",
    img: "images/papers-images/img-20241218-1.png",
    link: "https://www.preprints.org/manuscript/202412.1705/v1"
  },
  {
    title: "Linear Attention Based Channel Estimation Scheme for V2X Communications",
    description: "An innovative channel estimation scheme for V2X communications. Considering the doubly selective fading and limited pilots in IEEE 802.11p standard, it introduces the RWKV network with linear attention combined with DPA. The RWKV-DPA estimator enhances performance and reduces complexity compared to existing ones.",
    date: "2024-12-13",
    tags: "Sequence",
    img: "images/papers-images/img-20241213-1.png",
    link: "https://ieeexplore.ieee.org/abstract/document/10779439"
  },
  {
    title: "Exploring Real&Synthetic Dataset and Linear Attention in Image Restoration",
    description: "RWKV-IR, a novel RWKV-based image restoration model that supports both global and local receptive fields. The model demonstrates superior performance on Urban100 x4 benchmark, achieving 0.08dB improvement over SwinIR and 0.03dB over MambaIR, showcasing RWKV-IR's advanced image restoration capabilities and fast convergence.",
    date: "2024-12-11",
    tags: "Image",
    img: "images/papers-images/img-20241211-1.png",
    link: "https://arxiv.org/abs/2412.03814"
  },
  {
    title: "Voice dialog system based on RWKV model",
    description: "This paper aims to develop an intelligent voice dialog system for the elderly. It uses the RWKV model fine-tuned by LoRA. Experimental results show it improves answer fluency and reasonableness. It has potential in elder care and future work will optimize the model.",
    date: "2024-11-28",
    tags: "Audio",
    img: "images/papers-images/img-20241128-1.png",
    link: "https://ieeexplore.ieee.org/abstract/document/10762107"
  },
  {
    title: "Visualrwkv-Hm: Enhancing Linear Visual-Language Models Via Hybrid Mixing",
    description: "This paper presents VisualRWKV-HM, a linear-complexity visual-language model. It integrates time and cross state mixing based on RWKV. Achieving SOTA on multiple benchmarks, it outperforms models like LLaVA-1.5 in efficiency at 24K context, showing strong scalability.",
    date: "2024-11-21",
    tags: "Image",
    img: "images/papers-images/img-20241121-1.png",
    link: "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5028149"
  },
  {
    title: "DFT: A Dual-branch Framework of Fluctuation and Trend for Stock Price Prediction",
    description: "A Dual-branch Framework of Fluctuation and Trend (DFT) for stock price prediction. The RWKV model is used in the DFT to model time correlations in both the fluctuation and trend branches. It combines the power of RNN and Transformer, maintaining the time sequence of input features and satisfying the causality of the input. This helps in effectively capturing short-term fluctuations and trend information from stocks while explicitly modeling temporal variations and causal correlations, leading to improved performance compared to existing methods",
    date: "2024-11-09",
    tags: "Sequence",
    img: "images/papers-images/img-20241109-1.png",
    link: "https://arxiv.org/abs/2411.06065"
  },
  {
    title: "Video RWKV: Video Action Recognition Based RWKV",
    description: "LSTM CrossRWKV (LCR) for video understanding. It uses a novel CrossRWKV gate to handle video challenges. LCR stores long-term memory and reduces redundant information. Experiments on datasets show its effectiveness, setting a new benchmark in video understanding with RWKV.",
    date: "2024-11-08",
    tags: "3D/4D",
    img: "images/papers-images/img-20241108-1.png",
    link: "https://arxiv.org/abs/2411.05636"
  },
  {
    title: "From Explicit Rules to Implicit Reasoning in an Interpretable Violence Monitoring System",
    description: "RuleVM for weakly supervised violence monitoring. It uses a dual-branch structure with different designs for images and text. The implicit branch uses visual features for coarse-grained classification, and the explicit branch uses language-image alignment with YOLO-World and data mining. RWKV is used in the lightweight time-series module.",
    date: "2024-10-29",
    tags: "3D/4D",
    img: "images/papers-images/img-20241029-1.png",
    link: "https://arxiv.org/abs/2410.21991"
  },
  {
    title: "Modern Sequence Models in Context of Multi-Agent Reinforcement Learning",
    description: "MAM and MARWKV architectures inspired by MAT. Experiments show they perform comparably to MAT. MARWKV offers better inference computational efficiency, especially with more agents. RWKV is used in MARWKV for sequence modeling.",
    date: "2024-10-28",
    tags: "Sequence",
    img: "images/papers-images/img-20241028-1.png",
    link: "https://epub.jku.at/obvulihs/content/titleinfo/10580112"
  },
  {
    title: "AutoGMM-RWKV: A Detecting Scheme Based on Attention Mechanisms Against Selective Forwarding Attacks in Wireless Sensor Networks",
    description: "This paper presents AutoGMM - RWKV to detect selective forwarding attacks in WSNs. It focuses on node SFRs time series. By integrating autoencoder, GMM, and K - means with RWKV, it improves detection accuracy. Simulation shows low FDR and MDR, offering a robust solution.",
    date: "2024-10-23",
    tags: "Sequence",
    img: "images/papers-images/img-20241023-1.png",
    link: "https://ieeexplore.ieee.org/abstract/document/10729884"
  },
  {
    title: "MATCC: A Novel Approach for Robust Stock Price Prediction Incorporating Market Trends and Cross-time Correlations",
    description: "MATCC, extracts market trends, decomposes stock data, and mines cross-time correlation. Experiments show MATCC outperforms previous works. It uses RWKV to model inter-temporal correlations",
    date: "2024-10-21",
    tags: "Sequence",
    img: "images/papers-images/img-20241021-1.png",
    link: "https://dl.acm.org/doi/abs/10.1145/3627673.3679715"
  },
  {
    title: "VisualRWKV-HD and UHD: Advancing High-Resolution Processing for Visual Language Models",
    description: "VisualRWKV-HD and VisualRWKV-UHD for high-resolution visual inputs in visual language models. It details techniques like lossless downsampling and image segmentation. Experiments on benchmarks show their effectiveness, with RWKV models achieving better performance in handling high-resolution tasks.",
    date: "2024-10-15",
    tags: "Image",
    img: "images/papers-images/img-20241015-1.png",
    link: "https://arxiv.org/abs/2410.11665"
  },
  {
    title: "AttnInput: Revolutionizing Pinyin Input with Context-Aware RWKV Language Models",
    description: "AttnInput, a novel approach leveraging RWKV for Pinyin IME. It integrates Pinyin into RWKV s internal state, addressing semantic discontinuity. Using a pre-training strategy, it reduces costs. Experimental results show it achieves state-of-the-art performance on abbreviated Pinyin input.",
    date: "2024-10-13",
    tags: "Language",
    img: "images/papers-images/img-20241013-1.png",
    link: "https://openreview.net/forum?id=9OxTqscUwi"
  },
  {
    title: "OccRWKV: Rethinking Efficient 3D Semantic Occupancy Prediction with Linear Complexity",
    description: "OccRWKV, an efficient 3D semantic occupancy network inspired by RWKV. It separates predictions into branches with Sem-RWKV and GeoRWKV blocks. By projecting features to BEV space and using BEV-RWKV block, it achieves real-time inference. It outperforms state-of-the-art methods on SemanticKITTI dataset",
    date: "2024-09-26",
    tags: "3D/4D",
    img: "images/papers-images/img-20240926-1.jpg",
    link: "https://www.arxiv.org/abs/2409.19987"
  },
  {
    title: "DiSHA: Dimension-Sharding Adaptation of Large Language Models with Fast Convergence and Fast Computation",
    description: "The paper proposes DiSHA, a dimension-sharding adaptation framework for efficient fine-tuning of large language models (LLMs), addressing LoRA's slow convergence by partitioning pre-trained weights into shards updated via a shared trainable matrix. DiSHA introduces Block Affine Efficient Computation (Bone) for high efficiency and Block Affine Transformation (Bat) to resolve collinear updates. Evaluations demonstrate DiSHA's superiority over LoRA variants in NLU and NLG tasks. Notably, Bone achieves higher performance on RWKV-7B and RWKV6-3B models with equal or fewer parameters, showcasing faster convergence and better generalization. The framework reduces memory and computational costs, enabling resource-efficient adaptation, particularly benefiting architectures like RWKV through optimized parameter sharing and nonlinear updates.",
    date: "2024-09-19",
    tags: "General",
    img: "images/papers-images/img-20240919-1.png",
    link: "https://arxiv.org/abs/2409.15371"
  },
  {
    title: "Multi-scale RWKV with 2-dimensional temporal convolutional network for short-term photovoltaic power forecasting",
    description: "MSRWKV-2DTCN for short-term PV power forecasting. It uses FFT to identify periodicity, combines RWKV with a multi-scale 2D TCN, and conducts experiments on real datasets. The model shows high accuracy and strong generalization capabilities.",
    date: "2024-09-06",
    tags: "Sequence",
    img: "images/papers-images/img-20240906-1.png",
    link: "https://www.sciencedirect.com/science/article/abs/pii/S0360544224028433"
  },
  {
    title: "Experimentation in Content Moderation using RWKV",
    description: "Investigates RWKV's efficacy in content moderation. It creates a novel dataset for distillation, generates responses using LLMs, and fine-tunes RWKV. The study shows RWKV can improve content moderation accuracy and efficiency, and paves the way for more efficient models.",
    date: "2024-09-05",
    tags: "Language",
    img: "images/papers-images/img-20240905-1.png",
    link: "https://arxiv.org/abs/2409.03939"
  },
  {
    title: "Temporal and Interactive Modeling for Efficient Human-Human Motion Generation",
    description: "TTIM for efficient human-human motion generation. It proposes Causal Interactive Injection, Role-Evolving Mixing, and Localized Pattern Amplification. Experiments on InterHuman show TIM's superiority, achieving state-of-the-art results with only 32% of InterGen's trainable parameters, using RWKV",
    date: "2024-08-30",
    tags: "3D/4D",
    img: "images/papers-images/img-20240830-1.png",
    link: "https://arxiv.org/abs/2408.17135"
  },
  {
    title: "OnlySportsLM: Optimizing Sports-Domain Language Models with SOTA Performance under Billion Parameter",
    description: "A small sports-domain language model. It creates the OnlySports collection (dataset, benchmark, LM). Using 600 billion tokens data, it optimizes RWKV-v6 for sports tasks, training a 196M param model. OnlySportsLM outperforms prior models and rivals larger ones in the sports domain.",
    date: "2024-08-30",
    tags: "Language",
    img: "images/papers-images/img-20240830-2.png",
    link: "https://arxiv.org/abs/2409.00286"
  },
  {
    title: "Revenge of the Fallen? Recurrent Models Match Transformers at Predicting Human Language Comprehension MetricsRevenge of the Fallen? Recurrent Models Match Transformers at Predicting Human Language Comprehension Metrics",
    description: "The paper proposes that while transformers have been dominant in natural language processing, the newly developed RWKV and Mamba recurrent models are now challenging this status. It shows that these recurrent models can perform as well as or even better than transformers in predicting human language comprehension metrics, thus opening up new discussions on the suitability of different architectures for this task.",
    date: "2024-08-26",
    tags: "General",
    img: "images/papers-images/img-20240826-1.png",
    link: "https://arxiv.org/abs/2404.19178"
  },
  {
    title: "Why Perturbing Symbolic Music is Necessary: Fitting the Distribution of Never-used Notes through a Joint Probabilistic Diffusion Model",
    description: "Music-Diff architecture, which uses a joint probabilistic diffusion model. It improves note distribution fitting and sample diversity compared to language models like RWKV-music, enhancing rhythmic and structural coherence in generated music.",
    date: "2024-08-04",
    tags: "Audio",
    img: "images/papers-images/img-20240804-1.png",
    link: "https://arxiv.org/abs/2408.01950"
  },
  {
    title: "Optimizing Robotic Manipulation with Decision-RWKV: A Recurrent Sequence Modeling Approach for Lifelong Learning",
    description: "Explores RWKV's integration with decision transformer and experience replay in robotic manipulation. It proposes the Decision-RWKV model, tests it on D4RL and D'Claw platforms, and shows its effectiveness in single-task and lifelong learning, with code open-sourced.",
    date: "2024-07-23",
    tags: "Sequence",
    img: "images/papers-images/img-20240723-1.png",
    link: "https://arxiv.org/abs/2407.16306"
  },
  {
    title: "BSBP-RWKV: Background Suppression with Boundary Preservation for Efficient Medical Image Segmentation",
    description: "BSBP-RWKV for accurate and efficient medical image segmentation. It combines the advantages of PMD and RWKV, devises DWT-PMD RWKV Block and Multi-Step Runge-Kutta convolutional Block, and proposes a novel loss function. Experiments show its superior accuracy and efficiency.",
    date: "2024-07-21",
    tags: "Image",
    img: "images/papers-images/img-20240721-1.png",
    link: "https://openreview.net/pdf?id=ULD5RCk0oo"
  },
  {
    title: "GoldFinch: High Performance RWKV/Transformer Hybrid with Linear Pre-Fill and Extreme KV-Cache Compression",
    description: "GoldFinch, a hybrid Linear Attention/Transformer model. It uses a new technique to generate a highly compressed KV-Cache. GoldFinch stacks GOLD transformer on an enhanced RWKV-6 (Finch) architecture. It shows improved performance with reduced cache size compared to Finch and Llama.",
    date: "2024-07-16",
    tags: "General",
    img: "images/papers-images/img-20240716-1.png",
    link: "https://arxiv.org/abs/2407.12077"
  },
  {
    title: "Restore-RWKV: Efficient and Effective Medical Image Restoration with RWKV",
    description: "Restore-RWKV, the first RWKV-based model for medical image restoration. It modifies RWKV's attention and token shift layers to handle 2D images, capturing global and local dependencies. Experiments show its superiority in various tasks, serving as an efficient and effective backbone.",
    date: "2024-07-14",
    tags: "Image",
    img: "images/papers-images/img-20240714-1.png",
    link: "https://arxiv.org/abs/2407.11087"
  },
  {
    title: "Enhancing Transformer RNNs with Multiple Temporal Perspectives",
    description: "This paper introduces the concept of multiple temporal perspectives to enhance RNNs. Applied to RWKV, it enriches context understanding with minimal parameter increase. Empirical results vali date its effectiveness, showing improved performance on benchmarks while maintaining linear inference complexity.",
    date: "2024-07-11",
    tags: "General",
    img: "images/papers-images/img-20240711-1.png",
    link: "https://arxiv.org/abs/2402.02625"
  },
  {
    title: "Mamba or RWKV: Exploring High-Quality and High-Efficiency Segment Anything Model",
    description: "RWKV-SAM, an efficient segment-anything model with a mixed backbone of convolution and RWKV operation. This model achieves high accuracy and efficiency, outperforming others in benchmarks. It also trains on a combined high-quality dataset for better segmentation.",
    date: "2024-06-27",
    tags: "Image",
    img: "images/papers-images/img-20240627-1.png",
    link: "https://arxiv.org/abs/2406.19369"
  },
  {
    title: "VisualRWKV: Exploring Recurrent Neural Networks for Visual Language Models",
    description: "VisualRWKV, the first application of the linear RNN model RWKV in multimodal learning. It proposes novel mechanisms like data-dependent recurrence. Experiments show it performs competitively compared to Transformer models, with efficient computation and memory usage.",
    date: "2024-06-19",
    tags: "Image",
    img: "images/papers-images/img-20240619-1.png",
    link: "https://arxiv.org/abs/2406.13362"
  },
  {
    title: "RWKV-CLIP: A Robust Vision-Language Representation Learner",
    description: "RWKV-CLIP, the first RWKV-driven vision-language model. Experiments show RWKV-CLIP's robustness and effectiveness, achieving state-of-the-art performance in multiple downstream tasks.",
    date: "2024-06-11",
    tags: "Image",
    img: "images/papers-images/img-20240611-1.png",
    link: "https://arxiv.org/abs/2406.06973"
  },
  {
    title: "PointRWKV: Efficient RWKV-Like Model for Hierarchical Point Cloud Learning",
    description: "PointRWKV, a new model with linear complexity adapted from RWKV in NLP for 3D point cloud learning. It uses modified multi-headed matrix-valued states and a dynamic attention recurrence mechanism to explore global processing capabilities and a parallel branch to encode local geometric features, outperforming other models and saving FLOPs.",
    date: "2024-05-24",
    tags: "3D/4D",
    img: "images/papers-images/img-20240524-1.png",
    link: "https://arxiv.org/abs/2405.15214"
  },
  {
    title: "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence",
    description: "Eagle (RWKV-5) and Finch (RWKV-6), improving RWKV-4. Their architectural enhancements include multiheaded matrix-valued states and dynamic recurrence. New multilingual corpus and tokenizer are introduced. Trained models show competitive performance, and all are publicly released.",
    date: "2024-04-08",
    tags: "General",
    img: "images/papers-images/img-20240408-1.png",
    link: "https://arxiv.org/abs/2404.05892"
  },
  {
    title: "Diffusion-RWKV: Scaling RWKV-Like Architectures for Diffusion Models",
    description: "Diffusion-RWKV, an architecture adapting RWKV for diffusion models in image generation. It handles long-range hidden states linearly, showing comparable performance to Transformers but with lower complexity, thus being a promising alternative in this field.",
    date: "2024-04-06",
    tags: "Image",
    img: "images/papers-images/img-20240406-1.png",
    link: "https://arxiv.org/abs/2404.04478"
  },
  {
    title: "Onboard deep lossless and near-lossless predictive coding of hyperspectral images with line-based attention",
    description: "Deep learning in spacecraft hyperspectral image compression was challenging. This paper designs LineRWKV, a predictive neural network. It uses a novel hybrid operation, combines Transformers & RNNs. LineRWKV outperforms CCSDS-123.0-B-2 in compression and shows good throughput on a 7W system.",
    date: "2024-03-26",
    tags: "Image",
    img: "images/papers-images/img-20240326-1.png",
    link: "https://arxiv.org/abs/2403.17677"
  },
  {
    title: "Vision-RWKV: Efficient and Scalable Visual Perception with RWKV-Like Architectures",
    description: "Vision-RWKV, an adaptation of the RWKV model for vision tasks. It offers efficient handling of sparse inputs and strong global processing, with reduced spatial aggregation complexity. VRWKV outperforms ViT in image classification and shows advantages in dense prediction tasks, being a promising alternative for visual perception.",
    date: "2024-03-07",
    tags: "Image",
    img: "images/papers-images/img-20240307-1.png",
    link: "https://arxiv.org/abs/2403.02308"
  },
  {
    title: "TLS-RWKV: Real-Time Online Action Detection with Temporal Label Smoothing",
  description: "TLS-RWKV for online action detection. It utilizes the RWKV model with temporal label smoothing. Experiments on THUMOS'14 and TVSeries datasets show state-of-the-art performance and high efficiency, making it suitable for real-time applications and resource-constrained devices.",
    date: "2024-02-19",
    tags: "3D/4D",
    img: "images/papers-images/img-20240219-1.png",
    link: "https://link.springer.com/article/10.1007/s11063-024-11540-0"
  },
  {
    title: "SDiT: Spiking Diffusion Model with Transformer",
    description: "Spiking Diffusion Transformer (SDiT), a novel SNN diffusion model. It uses RWKV for efficient self-attention. SDiT aims to provide a baseline for SNN generative models and shows competitiveness on multiple datasets, generating high-quality images with lower cost and shorter sampling time.",
    date: "2024-02-18",
    tags: "Image",
    img: "images/papers-images/img-20240218-1.png",
    link: "https://arxiv.org/abs/2402.11588"
  },
  {
    title: "RWKV-TS: Beyond Traditional Recurrent Neural Network for Time Series Tasks",
    description: "Traditional RNNs have declined in time series tasks. This paper presents RWKV-TS, an efficient RNN-based model. It has O(L) complexity, captures long-term info well, and is computationally efficient. RWKV-TS shows competitive performance with reduced latency and memory use in various tasks.",
    date: "2024-01-17",
    tags: "Sequence",
    img: "images/papers-images/img-20240117-1.png",
    link: "https://arxiv.org/abs/2401.09093"
  },
  {
    title: "Advancing VAD Systems Based on Multi-Task Learning with Improved Model Structures",
    description: "Semantic VAD systems based on multi-task learning with improved models (RWKV for real-time, SAN-M for offline) to address issues in traditional binary VAD. Evaluations show significant improvements in CER, DCF, and NRR metrics compared to DFSMN-based systems.",
    date: "2023-12-19",
    tags: "Audio",
    img: "images/papers-images/img-20231219-1.png",
    link: "https://arxiv.org/abs/2312.14860"
  },
  {
    title: "RWKV-based Encoder-Decoder Model for Code Completion",
    description: "An RWKV-based encoder-decoder model for code completion. It aims to address challenges in this area. The model shows good performance and has potential for improving code generation efficiency, but more research is needed for wider application and optimization.",
    date: "2023-11-17",
    tags: "Language",
    img: "images/papers-images/img-20231117-1.png",
    link: "https://ieeexplore.ieee.org/abstract/document/10442108"
  },
  {
    title: "RWKV: A Linear Attention Mechanism for Temperature and Humidity Compensation for Gas Sensors",
    description: "A novel methodology for a PANI-CeO2 ammonia gas sensor to address temperature and humidity compensation. It uses the RWKV network with a Linear attention mechanism. The process has three stages. The method shows high predictive accuracy, with low mean absolute and relative errors.",
    date: "2023-10-25",
    tags: "Sequence",
    img: "images/papers-images/img-20231025-1.png",
    link: "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4612708"
  },
  {
    title: "Exploring RWKV for Memory Efficient and Low Latency Streaming ASR",
    description: "Applying RWKV, a linear attention transformer variant, to streaming ASR. It combines transformer performance and RNN inference efficiency. Experiments show RWKV-Transducer and RWKV-Boundary-Aware-Transducer achieve good accuracy with minimal latency and memory cost.",
    date: "2023-09-26",
    tags: "Audio",
    img: "images/papers-images/img-20230926-1.png",
    link: "https://arxiv.org/abs/2309.14758"
  },
  {
    title: "RWKV: Reinventing RNNs for the Transformer Era",
    description: "RWKV, a novel model architecture. It combines the efficient parallelizable training of transformers with the efficient inference of RNNs. RWKV uses a linear attention mechanism, scales to 14 billion parameters, and performs comparably to similar-sized transformers, advancing sequence processing tasks.",
    date: "2023-05-22",
    tags: "General",
    img: "images/papers-images/img-20230522-1.png",
    link: "https://arxiv.org/abs/2305.13048"
  },
];

let currentProjects = allProjects;
let currentPage = 1;
const itemsPerPage = 4;

function filterProjects(category, element) {
  document.querySelectorAll(".filter-btn").forEach((btn) => {
    btn.classList.remove("bg-blue-500", "text-white", "hover:bg-blue-600");
    btn.classList.add("bg-gray-100", "text-gray-700", "hover:bg-gray-200");
  });

  element.classList.add("bg-blue-500", "text-white");
  element.classList.remove("bg-gray-100", "text-gray-700", "hover:bg-gray-200");
  element.classList.add("hover:bg-blue-600");

  currentProjects =
    category === "ALL"
      ? allProjects
      : allProjects.filter((project) => project.tags.includes(category));
  currentPage = 1;
  displayProjects();
  renderPagination();
  updateTotalCount();
}

function displayProjects() {
  const projectGrid = document.getElementById("projectGrid");
  projectGrid.innerHTML = "";

  const start = (currentPage - 1) * itemsPerPage;
  const end = start + itemsPerPage;
  const projectsToDisplay = currentProjects.slice(start, end);

  projectsToDisplay.forEach((project) => {
    const projectLink = document.createElement("a");
    projectLink.href = project.link;
    projectLink.target = "_blank";
    projectLink.className =
      "block transform transition-transform duration-200 hover:scale-105 h-[420px]";

    const projectCard = document.createElement("div");
    projectCard.className =
      "bg-white rounded-lg overflow-hidden shadow border border-gray-200 h-full flex flex-col";

    projectCard.innerHTML = `
      <div class="relative w-full h-48 bg-gray-100 flex items-center justify-center overflow-hidden flex-shrink-0">
        <img src="${project.img}" alt="Project Image" class="w-full h-full object-cover transition-transform duration-200 ease-in-out hover:scale-105">
      </div>
      <div class="p-4 flex flex-col flex-grow">
        <h3 class="text-lg font-semibold text-gray-800 mb-2 project-card-title line-clamp-2" title="${project.title}">
          ${project.title}
        </h3>
        <p class="text-gray-600 text-sm leading-snug mb-4 project-card-description line-clamp-4 flex-grow">${project.description}</p>
        <div class="flex items-center justify-between mt-auto">
          <div class="text-gray-500 text-xs">#${project.tags}</div>
          <div class="text-gray-500 text-xs">${project.date}</div>
        </div>
      </div>
    `;

    projectLink.appendChild(projectCard);
    projectGrid.appendChild(projectLink);
  });
}

function renderPagination() {
  const pagination = document.getElementById("pagination");
  pagination.innerHTML = "";
  const totalPages = Math.ceil(currentProjects.length / itemsPerPage);

  if (totalPages <= 1) {
    pagination.style.display = "none";
    return;
  } else {
    pagination.style.display = "flex";
  }

  const prevButton = document.createElement("button");
  prevButton.className = `px-2 py-1 text-gray-600 ${currentPage === 1 ? "opacity-50 cursor-not-allowed" : ""
    }`;
  prevButton.innerHTML = "&lt;";
  prevButton.onclick = () => {
    if (currentPage > 1) goToPage(currentPage - 1);
  };
  pagination.appendChild(prevButton);

  for (let i = 1; i <= totalPages; i++) {
    const pageButton = document.createElement("button");
    pageButton.className = `px-3 py-1 rounded border ${i === currentPage ? "bg-blue-500 text-white" : "bg-gray-200 text-gray-700"
      }`;
    pageButton.innerText = i;
    pageButton.onclick = () => goToPage(i);
    pagination.appendChild(pageButton);
  }

  const nextButton = document.createElement("button");
  nextButton.className = `px-2 py-1 text-gray-600 ${currentPage === totalPages ? "opacity-50 cursor-not-allowed" : ""
    }`;
  nextButton.innerHTML = "&gt;";
  nextButton.onclick = () => {
    if (currentPage < totalPages) goToPage(currentPage + 1);
  };
  pagination.appendChild(nextButton);
}

function goToPage(page) {
  currentPage = page;
  displayProjects();
  renderPagination();
}

function updateTotalCount() {
  const totalCount = document.getElementById("totalCount");
  totalCount.innerText = ``; //Total ${currentProjects.length} papers.`;

  if (currentProjects.length <= 4) {
    totalCount.style.display = "none";
  }
}

filterProjects("ALL", document.querySelector(".filter-btn"));
