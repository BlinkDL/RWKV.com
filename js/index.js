const allProjects = [
  {
    title: "LADY: Linear Attention for Autonomous Driving Efficiency without Transformers",
    description: "This paper, based on RWKV-7 modules, proposes LADY, the first fully linear attention-based end-to-end autonomous driving model. It introduces a lightweight linear cross-attention mechanism to enable efficient cross-modal fusion while maintaining linear complexity. LADY fuses multi-frame camera and LiDAR features with constant computational and memory overhead, enabling long-range temporal context integration. Experiments on NAVSIM and Bench2Drive benchmarks show state-of-the-art performance with significantly reduced computational cost, validated on edge devices.",
    date: "2025-12-17",
    tags: "3D/4D",
    img: "images/papers-images/img-20251217-1.png",
    link: "https://arxiv.org/abs/2512.15038"
  },
  {
    title: "SemanticBBV: A Semantic Signature for Cross-Program Knowledge Reuse in Microarchitecture Simulation",
    description: "This paper, based on RWKV, introduces SemanticBBV, a two-stage framework for generating semantic, performance-aware program signatures to enable cross-program knowledge reuse in microarchitecture simulation. It first uses a lightweight RWKV-based encoder to create Basic Block Embeddings (BBEs) capturing assembly semantics, then aggregates these with an order-invariant Set Transformer co-trained with triplet loss and CPI regression. This approach achieves 86.3% accuracy in cross-program performance estimation using just 14 universal points, yielding a 7143× simulation speedup.",
    date: "2025-12-11",
    tags: "Sequence",
    img: "images/papers-images/img-20251211-1.png",
    link: "https://arxiv.org/abs/2512.10231"
  },
  {
    title: "Fourier-RWKV: A Multi-State Perception Network for Efficient Image Dehazing",
    description: "This paper, based on RWKV, proposes Fourier-RWKV, a novel dehazing framework that integrates spatial, frequency-domain, and semantic perceptual states. It introduces DQ-Shift for adaptive spatial perception, extends WKV attention to the Fourier domain for global modeling, and employs SBM for encoder-decoder feature alignment. The model achieves state-of-the-art performance with linear complexity, effectively handling non-uniform haze while reducing computational overhead.",
    date: "2025-12-09",
    tags: "Image",
    img: "images/papers-images/img-20251209-2.png",
    link: "https://arxiv.org/abs/2512.08161"
  },
  {
    title: "FRWKV:Frequency-Domain Linear Attention for Long-Term Time Series Forecasting",
    description: "This paper, based on RWKV's linear attention mechanism, proposes FRWKV, a frequency-domain linear-attention framework for long-term time series forecasting. By integrating RWKV's O(T) linear attention with frequency-domain analysis, FRWKV overcomes the quadratic complexity of traditional Transformers while effectively exploiting spectral information. The model processes real and imaginary frequency components separately through RWKV-style state recursion, achieving scalable long-sequence modeling. Extensive experiments on eight datasets demonstrate state-of-the-art performance, with ablation studies confirming the synergy between linear attention and frequency-domain processing.",
    date: "2025-12-09",
    tags: "Sequence",
    img: "images/papers-images/img-20251209-1.png",
    link: "https://arxiv.org/abs/2512.07539"
  },
  {
    title: "EG-Net: Edge-Global aware network for accurate skin lesion segmentation",
    description: "This paper proposes EG-Net, a novel segmentation network for melanoma that integrates RWKV-based global modeling. The Edge-Global Aware Network combines an edge feature extraction module using HSV color space and gradient information with an RWKV-based global context module to enhance boundary precision and overall segmentation accuracy. Experiments on ISIC2016, HAM10000, and PH2 datasets demonstrate state-of-the-art performance, achieving Dice scores above 90% in cross-dataset validation, showcasing strong robustness and clinical potential.",
    date: "2025-12-01",
    tags: "Image",
    img: "images/papers-images/img-20251201-1.png",
    link: "https://www.sciencedirect.com/science/article/abs/pii/S1746809425018142"
  },
  {
    title: "AFF-UNet-RWKV: A Lightweight Model for High-Quality Deblurring in Medical Imaging",
    description: "This paper, based on RWKV-lite spatial mixer, proposes AFF-UNet-RWKV, a lightweight deep learning model for medical image deblurring. By integrating Attention Feature Fusion (AFF) with RWKV-lite, the model effectively fuses encoder-decoder features and captures long-range spatial dependencies. Experiments on the PathMNIST dataset show superior performance, achieving 32.03 dB PSNR and 0.898 SSIM, outperforming traditional methods and DeblurGAN in restoring image details and structures.",
    date: "2025-11-20",
    tags: "Image",
    img: "images/papers-images/img-20251120-2.png",
    link: "https://madison-proceedings.com/index.php/aetr/article/view/4338"
  },
  {
    title: "Evolution Strategies at the Hyperscale",
    description: "Applying its novel method to fine-tune RWKV-7 models, this paper introduces Evolution Guided General Optimization via Low-rank Learning (EGGROLL), an evolution strategies algorithm for backprop-free optimization of billion-parameter networks. EGGROLL overcomes the computational and memory bottlenecks of traditional ES by employing low-rank parameter perturbations, analogous to LoRA. This dramatically improves training throughput, enabling stable pre-training of novel recurrent architectures and outperforming gradient-based methods in reasoning tasks. The method's effectiveness is demonstrated across reinforcement learning, LLM fine-tuning, and pre-training from scratch with purely integer datatypes.",
    date: "2025-11-20",
    tags: "General",
    img: "images/papers-images/img-20251120-1.png",
    link: "https://www.arxiv.org/abs/2511.16652"
  },
  {
    title: "基于动态邻接融合与通道混合的图神经网络社团检测方法",
    description: "Inspired by the RWKV-style ChannelMix architecture, this paper proposes the Temporal-Channel Graph Attention Network (TC-GAT) for dynamic community detection. The model addresses limitations in existing dynamic graph methods by introducing two novel components: a Dynamic Adjacency Fusion (DAF) module to capture diverse temporal behaviors via node-adaptive weights, and a lightweight Graph Channel Mixer (GCM) to enhance node representations by modeling feature channel interactions. Experiments on real-world dynamic graph datasets demonstrate that TC-GAT significantly outperforms mainstream models in both accuracy and efficiency, effectively balancing performance and computational cost.",
    date: "2025-11-18",
    tags: "Sequence",
    img: "images/papers-images/img-20251118-1.png",
    link: "https://www.arocmag.cn/abs/2025.07.0271"
  },
  {
    title: "RawRWKV : An efﬁcient raw image enhancement framework via RWKV architecture",
    description: "This paper proposes RawRWKV, a novel framework based on the RWKV architecture for low-light raw image enhancement. Addressing the performance-efficiency trade-off in existing CNN and Vision Transformer methods, RawRWKV integrates RWKV blocks into a multi-scale U-Net structure. This design leverages linear attention to model global dependencies with low computational cost. Experiments on SID and MCR raw datasets show that the model achieves state-of-the-art performance in PSNR and SSIM metrics while maintaining low parameters and FLOPs, demonstrating RWKV's potential for efficient and effective image restoration.",
    date: "2025-11-17",
    tags: "Image",
    img: "images/papers-images/img-20251117-1.png",
    link: "https://link.springer.com/article/10.1007/s11760-025-04940-9"
  },
  {
    title: "ASALP: An Automatic Scaling Architecture for Edge Node Resources Based on Load Prediction",
    description: "This paper proposes ASALP, an automatic scaling architecture that enhances Kubernetes for edge computing using an improved RWKV-EFE model for proactive load prediction. To address the shortcomings of the default reactive autoscaler, ASALP integrates the RWKV-based predictor to forecast traffic and adjust resources in advance. This system, built on Kubernetes and KubeEdge, enables autonomous edge node scaling and dynamic load balancing. Experimental results demonstrate that this approach significantly improves request success rates and system stability compared to traditional methods, effectively mitigating issues caused by unstable network links in edge environments.",
    date: "2025-11-16",
    tags: "Sequence",
    img: "images/papers-images/img-20251116-1.png",
    link: "https://link.springer.com/chapter/10.1007/978-3-032-10466-3_32"
  },
  {
    title: "Otter: Mitigating Background Distractions of Wide-Angle Few-Shot Action Recognition with Enhanced RWKV",
    description: "This paper, based on an enhanced Receptance Weighted Key Value (RWKV) architecture, introduces Otter to address background distractions in wide-angle few-shot action recognition (FSAR). The proposed model features two key components: the Compound Segmentation Module (CSM), which highlights subjects by segmenting frames and learning patch weights, and the Temporal Reconstruction Module (TRM), which reconstructs degraded temporal relations using bidirectional scanning. Otter combines these modules to simultaneously emphasize subjects and improve temporal modeling, achieving state-of-the-art results on challenging wide-angle video benchmarks.",
    date: "2025-11-11",
    tags: "3D/4D",
    img: "images/papers-images/img-20251111-1.png",
    link: "https://arxiv.org/abs/2511.06741"
  },
  {
    title: "MRT: Learning Compact Representations with Mixed RWKV-Transformer for Extreme Image Compression",
    description: "This paper proposes a Mixed RWKV-Transformer (MRT) architecture for extreme image compression, addressing the spatial redundancy in conventional 2-D latent representations. The model encodes images into a more compact 1-D format by synergistically using RWKV modules to capture global dependencies across image windows and Transformer blocks for local details within them. A dedicated RWKV Compression Model (RCM) further enhances efficiency by compressing these 1-D features. Experiments show MRT achieves state-of-the-art rate-distortion performance, significantly outperforming existing methods at extremely low bitrates by reducing redundancy more effectively.",
    date: "2025-11-10",
    tags: "Image",
    img: "images/papers-images/img-20251110-1.png",
    link: "https://arxiv.org/abs/2511.06717"
  },
  {
    title: "RWKVSR: Receptance Weighted Key-Value Network for Hyperspectral Image Super-Resolution",
    description: "This paper, based on RWKV architecture, proposes RWKVSR for hyperspectral image super-resolution, addressing computational inefficiencies and spectral-spatial fusion challenges. The method integrates a linear-complexity RWKV module for global dependency modeling, a Spectral-Spatial Residual Module with anisotropic 3D convolutions for multi-scale feature extraction, and a Hyperspectral Frequency Loss for spectral consistency. Experiments on CAVE and Harvard datasets demonstrate state-of-the-art performance in balancing accuracy and efficiency.",
    date: "2025-10-30",
    tags: "Image",
    img: "images/papers-images/img-20251030-2.png",
    link: "https://ieeexplore.ieee.org/document/11222729"
  },
  {
    title: "SleepRWKVNet: A multimodal sleep staging network integrating bidirectional interactive RWKV and physiological prior-driven sequence-aware loss",
    description: "This paper proposes SleepRWKVNet, a novel multimodal sleep staging network built upon a bidirectional interactive RWKV module (Bi-IFM). It addresses inconsistent modality contributions and inefficient long-sequence modeling by effectively fusing features from EEG, EOG, and EMG signals. The network also introduces a physiological prior-driven sequence-aware loss (PS-Loss), which incorporates subject-specific Markov transition probabilities to mitigate class imbalance and improve modeling of sleep stage transitions. Experiments on three public datasets demonstrate superior performance over existing methods, offering a robust solution for accurate automated sleep staging.",
    date: "2025-10-30",
    tags: "Sequence",
    img: "images/papers-images/img-20251030-1.png",
    link: "https://www.sciencedirect.com/science/article/abs/pii/S1746809425012248"
  },
  {
    title: "WKV-sharing embraced random shuffle RWKV high-order modeling for pan-sharpening",
    description: "This paper, based on the RWKV architecture, proposes a novel model for image pan-sharpening. It introduces a Random Shuffle scanning strategy to eliminate biases from fixed-sequence scanning in the spatial mixer. The model also incorporates a WKV-sharing mechanism to transfer activations across layers, reducing latency and enabling high-order interactions in the channel mixer. A random weight manifold loss is used to regularize the optimization space. This approach, named RS-RWKV, is shown to outperform state-of-the-art methods on pan-sharpening benchmarks, demonstrating superior multi-modal synergy and efficiency.",
    date: "2025-10-29",
    tags: "Image",
    img: "images/papers-images/img-20251029-1.png",
    link: "https://openreview.net/forum?id=gqfQfqDQhx"
  },
  {
    title: "Freq-RWKV: Granularity-Aware Spatial-Frequency Synergy via Dual-Domain Recurrent Scanning for Pan-sharpening",
    description: "This paper introduces Freq-RWKV, a novel spatial-frequency adaptive RWKV framework designed for pan-sharpening. To address the challenge of reconstructing high-frequency details, it proposes a dual-domain scanning mechanism guided by wavelet analysis within a U-shaped, coarse-to-fine fusion network. The architecture uses specialized modules to coordinate granularity-aware scanning across spatial and frequency domains, enabling the model to effectively improve the spatial resolution of multispectral images by integrating textural information from corresponding panchromatic images. Experimental results on multiple satellite datasets validate the method's superior performance.",
    date: "2025-10-27",
    tags: "Image",
    img: "images/papers-images/img-20251027-1.png",
    link: "https://dl.acm.org/doi/abs/10.1145/3746027.3755521"
  },
  {
    title: "Learning Structural Priors via Laplacian RWKV Diffusion with Light-Effect Dataset for Nighttime Visibility Enhancement",
    description: "This paper, based on the RWKV architecture, addresses joint nighttime visibility enhancement by tackling both low-light conditions and light-effect suppression. It introduces a new paired dataset, NightLight, for supervised training. The proposed method is a two-stage diffusion model that utilizes a novel Dual-Loop Laplacian RWKV (Lap-RWKV) to extract structural priors from the image. These priors guide the diffusion process to accurately remove light artifacts and enhance dark regions, outperforming state-of-the-art methods in joint light-effect suppression and low-light image enhancement tasks.",
    date: "2025-10-27",
    tags: "Image",
    img: "images/papers-images/img-20251027-2.png",
    link: "https://dl.acm.org/doi/abs/10.1145/3746027.3755510"
  },
  {
    title: "RWKV-PCSSC: Exploring RWKV Model for Point Cloud Semantic Scene Completion",
    description: "This paper proposes RWKV-PCSSC, a lightweight network for point cloud semantic scene completion inspired by the Receptance Weighted Key Value (RWKV) mechanism. To address the high complexity of existing methods, it introduces an RWKV Seed Generator (RWKV-SG) to produce a coarse scene representation and RWKV Point Deconvolution (RWKV-PD) modules for progressive refinement. This novel architecture significantly reduces model parameters and memory usage while achieving state-of-the-art performance on various indoor and outdoor datasets, demonstrating an efficient approach to generating complete semantic scenes from partial point cloud inputs.",
    date: "2025-10-27",
    tags: "3D/4D",
    img: "images/papers-images/img-20251027-3.png",
    link: "https://dl.acm.org/doi/abs/10.1145/3746027.3754908"
  },
  {
    title: "RWKV3D: An RWKV-Based Model with Multiple Training Strategies for Point Cloud Analysis",
    description: "This paper introduces RWKV3D, a novel computational framework based on the RWKV architecture, specifically tailored for point cloud analysis. To adapt RWKV for unordered 3D data, the model replaces the standard MLP layer with a Local Feature Mixer (LFM) for enhanced fine-grained feature extraction and introduces a Bidirectional Multi-head Shift (BMS) mechanism to expand the receptive field. The framework is adaptable to multiple training strategies, and experimental results show that RWKV3D outperforms Transformer and Mamba-based methods on benchmarks like ModelNet40 and ScanObjectNN while maintaining lower computational costs.",
    date: "2025-10-27",
    tags: "3D/4D",
    img: "images/papers-images/img-20251027-4.png",
    link: "https://dl.acm.org/doi/abs/10.1145/3746027.3755658"
  },
  {
    title: "RS³-RWKV: Leveraging RWKV for Efficient Remote Sensing Semantic Segmentation",
    description: "This paper introduces RS3-RWKV, a novel framework based on the RWKV architecture for efficient semantic segmentation of high-resolution remote sensing images. To address challenges like multi-scale targets and complex spatial dependencies, the authors propose a proximity-sensitive WKV attention mechanism (PS-WKV) with a spiral scan and a scale-adaptive shift mechanism (SA-Shift). These innovations enhance the model's ability to capture global context and adapt to varying object sizes. Experiments on the LoveDA and ISPRS Potsdam datasets show that RS3-RWKV achieves a superior balance of accuracy and computational efficiency compared to CNN, Transformer, and Mamba models.",
    date: "2025-10-22",
    tags: "Image",
    img: "images/papers-images/img-20251022-1.png",
    link: "https://ieeexplore.ieee.org/abstract/document/11214221"
  },
  {
    title: "FS-RWKV: Leveraging Frequency Spatial-Aware RWKV for 3T-to-7T MRI Translation",
    description: "This paper introduces FS-RWKV, an RWKV-based framework for synthesizing high-quality 7T MRI images from more accessible 3T scans. The model features two novel components: a Frequency Spatial Omnidirectional-Shift (FSO-Shift) module that uses wavelet decomposition to enhance global context while preserving high-frequency details, and a Structural Fidelity Enhancement Block (SFEB) for adaptive feature fusion. Comprehensive experiments show that FS-RWKV outperforms existing CNN, Transformer, GAN, and RWKV-based methods in 3T-to-7T translation, achieving superior anatomical fidelity and perceptual quality on medical imaging datasets.",
    date: "2025-10-10",
    tags: "Image",
    img: "images/papers-images/img-20251010-1.png",
    link: "https://arxiv.org/abs/2510.08951"
  },
  {
    title: "Bridging Transformers and RWKV: Towards Efficient Multimodal Video Understanding",
    description: "This paper proposes a hybrid RWKV-Transformer architecture to address the prohibitive computational cost of processing long videos in Multimodal Large Language Models (MLLMs). It replaces some Transformer layers with efficient RWKV modules, initializing their weights from pre-trained attention projections and using a progressive distillation strategy. To combat RWKV's history decay, cross-attention with global scene tokens is incorporated. This hybrid model significantly boosts throughput—by 20% when replacing 25% of layers—while matching or exceeding the performance of the original Transformer on multiple video understanding benchmarks.",
    date: "2025-10-08",
    tags: "3D/4D",
    img: "images/papers-images/img-20251008-1.png",
    link: "https://openreview.net/forum?id=kmNqnwA4aV"
  },
  {
    title: "DREAMSTATE: Diffusing States and Parameters for Recurrent Large Language Models",
    description: "This paper, based on the RWKV architecture, investigates the model's internal state as an editable knowledge representation. It introduces the DREAMSTATE framework, which uses a conditional Diffusion Transformer to model and generate RWKV states for controlled inference. Addressing the limitations of static recurrence, the authors propose a novel hybrid architecture where a diffusion model dynamically synthesizes the core WKV parameters based on global context. Experiments validate that the RWKV state is a structured manifold and confirm the training stability of this dynamic, context-aware design, opening new avenues for controllable generative models.",
    date: "2025-10-08",
    tags: "Language",
    img: "images/papers-images/img-20251008-2.png",
    link: "https://openreview.net/forum?id=HHsD970kdE"
  },
  {
    title: "GDSR: Global-Detail Integration through Dual-Branch Network with Wavelet Losses for Remote Sensing Image Super-Resolution",
    description: "Introducing the Receptance Weighted Key Value (RWKV) model to remote sensing image super-resolution, this paper proposes GDSR, a dual-branch network designed to capture both global and local features simultaneously. One branch leverages RWKV to model long-range dependencies with linear complexity, while a parallel convolutional branch extracts fine details. A Global-Detail Reconstruction Module fuses these features, and a novel Dual-Group Multi-Scale Wavelet Loss enhances reconstruction fidelity. The proposed method outperforms state-of-the-art models in performance and computational efficiency on several remote sensing benchmarks.",
    date: "2025-10-06",
    tags: "Image",
    img: "images/papers-images/img-20251006-1.png",
    link: "https://ieeexplore.ieee.org/document/11192566"
  },
  {
    title: "VRWKV-Editor: Reducing quadratic complexity in transformer-based video editing",
    description: "This paper introduces VRWKV-Editor, a novel video editing framework based on the RWKV architecture designed to overcome the quadratic computational complexity of traditional attention mechanisms. By integrating a linear spatio-temporal aggregation module from VRWKV into a video diffusion model, the proposed method significantly reduces computational and memory costs. Experiments demonstrate that VRWKV-Editor achieves up to a 3.7x speedup and 60% lower memory usage compared to state-of-the-art methods. This efficiency is gained without sacrificing performance, maintaining competitive frame consistency and text alignment, especially for long, high-resolution videos.",
    date: "2025-09-30",
    tags: "3D/4D",
    img: "images/papers-images/img-20250930-1.png",
    link: "https://arxiv.org/abs/2509.25998v2"
  },
  {
    title: "C3-OWD: A Curriculum Cross-modal Contrastive Learning Framework for Open-World Detection",
    description: "Leveraging RWKV for efficient multimodal fusion, this paper proposes C3-OWD, a framework to address the trade-off between robustness in adverse conditions and generalization to unseen categories in object detection. It uses a two-stage curriculum learning approach: Stage 1 pre-trains on visible-infrared (RGBT) data for robustness, while Stage 2 performs vision-language alignment for open-vocabulary capabilities. An Exponential Moving Average (EMA) mechanism is introduced to prevent catastrophic forgetting between stages. The model demonstrates competitive performance on both robustness (FLIR) and open-world detection (OV-COCO, OV-LVIS) benchmarks.",
    date: "2025-09-27",
    tags: "Image",
    img: "images/papers-images/img-20250927-1.png",
    link: "https://arxiv.org/abs/2509.23316"
  },
  {
    title: "DPC-QA Net: A No-Reference Dual-Stream Perceptual and Cellular Quality Assessment Network for Histopathology Images",
    description: "This paper introduces DPC-QA Net, a dual-stream network for no-reference quality assessment of histopathology images, which utilizes an Aggr-RWKV module to aggregate cellular-level embeddings. The model combines a global perceptual stream using wavelet features with a cellular quality stream assessing nuclear and membrane fidelity. Fusing these streams via cross-attention, the network accurately detects staining, membrane, and nuclear issues, achieving high performance on pathology and general image quality datasets. The model’s quality scores strongly correlate with the success of downstream cell recognition tasks, enabling practical pre-screening for computational pathology.",
    date: "2025-09-19",
    tags: "Image",
    img: "images/papers-images/img-20250919-1.png",
    link: "https://arxiv.org/abs/2509.15802"
  },
  {
    title: "Mastering Air Combat through Model-Based Reinforcement Learning",
    description: "Based on an RWKV-style linear attention module in its world model, this paper introduces a model-based reinforcement learning agent for Within-Visual-Range air combat. The agent enhances the Dreamer framework with safety-aware objectives, contrastive predictive coding for long-range dependencies, and Dyna-style actor-critic updates. Trained through a population-based self-play pipeline with curriculum initialization, the agent achieves superior zero-shot performance, higher sample efficiency than model-free baselines, and rapid adaptation against novel opponents in a high-fidelity simulation, demonstrating a viable approach for deployable autonomous combat systems.",
    date: "2025-09-17",
    tags: "Sequence",
    img: "images/papers-images/img-20250917-1.png",
    link: "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5499593"
  },
  {
    title: "RWKV-VIO: An Efficient and Low-Drift Visual–Inertial Odometry Using an End-to-End Deep Network",
    description: "This paper introduces RWKV-VIO, a novel Visual–Inertial Odometry (VIO) framework based on the RWKV architecture. It addresses challenges in existing deep learning VIO methods, such as temporal modeling and computational efficiency, by leveraging RWKV's lightweight structure and linear computational complexity. The framework also integrates a new Res-Encoder and a parallel encoding strategy for IMU data to enhance feature extraction. Experimental results demonstrate that RWKV-VIO achieves competitive localization accuracy while significantly reducing model size and inference time compared to state-of-the-art approaches.",
    date: "2025-09-15",
    tags: "3D/4D",
    img: "images/papers-images/img-20250915-1.png",
    link: "https://www.mdpi.com/1424-8220/25/18/5737"
  },
  {
    title: "Enhanced Traffic Sign Recognition via RWKV with Deformable Attention",
    description: "This paper proposes a novel vision encoder, RWKV with Deformable Attention (RWKV-DA), to improve traffic sign recognition. The architecture integrates the linear computational efficiency of RWKV with the adaptive focus of deformable attention, enabling it to handle diverse input features and image deformations effectively. Tested on the German Traffic Sign Recognition Benchmark (GTSRB), the model achieves state-of-the-art accuracy and shows superior computational efficiency, outperforming Vision Transformers by up to 12 times on high-resolution images, making it suitable for real-time autonomous systems.",
    date: "2025-09-15",
    tags: "Image",
    img: "images/papers-images/img-20250915-2.png",
    link: "https://dl.acm.org/doi/10.1145/3757749.3757779"
  },
  {
    title: "Multi-modal dynamic brain graph representation learning for brain disorder diagnosis via temporal sequence model",
    description: "Inspired by the RWKV large language model architecture, this paper proposes an efficient temporal multi-modal graph neural network (ET_MGNN) for diagnosing brain disorders. The model integrates dynamic functional connectivity (DFC) and structural connectivity (SC) into a unified brain network representation. By leveraging an RWKV block to capture complex short- and long-term temporal dependencies in dynamic brain graph sequences, ET_MGNN demonstrates significantly improved classification accuracy for conditions like autism and Alzheimer's disease across three datasets, outperforming several strong baselines.",
    date: "2025-09-13",
    tags: "Image",
    img: "images/papers-images/img-20250913-2.png",
    link: "https://www.sciencedirect.com/science/article/abs/pii/S0925231225021812"
  },
  {
    title: "A Traditional Approach to Symbolic Piano Continuation",
    description: "This paper, based on the RWKV-7 architecture, proposes a simple yet effective method for symbolic piano music continuation. Contesting the trend of large foundation models, the authors train a small, 20-million-parameter RWKV model on the Aria-MIDI dataset using a standard next-token prediction objective. Results from the MIREX 2025 challenge show their specialized model performed on par with a much larger Transformer baseline. The work demonstrates that smaller, task-specific models remain highly competitive for constrained generative music tasks when built on strong fundamentals like quality data and an efficient architecture.",
    date: "2025-09-13",
    tags: "Audio",
    img: "images/papers-images/img-20250913-1.png",
    link: "https://arxiv.org/abs/2509.12267"
  },
  {
    title: "ITC-RWKV: Interactive Tissue–Cell Modeling with Recurrent Key-Value Aggregation for Histopathological Subtyping",
    description: "This paper introduces ITC-RWKV, an approach that adapts the Receptance Weighted Key-Value (RWKV) architecture for efficient cell-level feature aggregation in histopathological image analysis. ITC-RWKV proposes a novel dual-stream framework to address limitations of existing models in fine-grained tasks like cancer subtype classification. It combines a macroscale tissue feature pathway with a dedicated cell pathway featuring an Aggr-RWKV module for linear-complexity aggregation of cellular representations. A bidirectional tissue-cell interaction module further refines cross-scale information. The method outperforms state-of-the-art models on four benchmarks, demonstrating the critical role of its RWKV-based aggregation and tissue-cell interaction for accurate computational pathology.",
    date: "2025-09-12",
    tags: "Image",
    img: "images/papers-images/img-20250912-1.png",
    link: "https://research.manchester.ac.uk/en/publications/itc-rwkv-interactive-tissuecell-modeling-with-recurrent-key-value"
  },
    {
    title: "EfficientIML: Efficient High-Resolution Image Manipulation Localization",
    description: "This paper proposes EfficientIML, an efficient framework for high-resolution image manipulation localization, addressing challenges posed by emerging diffusion-based forgeries and computational constraints. It introduces SIF, the first ultra-high-resolution semantic inpainting forgery dataset (1200+ images > 1024x1024). The core of EfficientIML is a lightweight, three-stage EfficientRWKV backbone, a hybrid state-space and attention network derived from Vision-RWKV, which captures global context and local details with linear computational complexity. Coupled with a multi-scale supervision strategy, EfficientIML significantly outperforms state-of-the-art baselines in localization performance, FLOPs, and inference speed.",
    date: "2025-09-10",
    tags: "Image",
    img: "images/papers-images/img-20250910-1.png",
    link: "https://arxiv.org/abs/2509.08583"
  },
  {
    title: "Robotic control optimization based on receptance-weighted reinforcement learning",
    description: "This paper proposes applying an improved Recurrent Weighted Kernel Value (RWKV) neural network architecture to reinforcement learning (RL) for robotic control. The authors optimize RWKV's channel mixing module and use the modified architecture to replace the self-attention mechanism in the Decision Transformer model. By treating robotic control as a sequence modeling task, this approach leverages RWKV's efficiency. Experiments on D4RL datasets show the proposed RL RWKV model achieves higher accuracy and faster performance compared to the Decision Transformer, demonstrating its potential for low-latency, real-world robotic applications.",
    date: "2025-09-08",
    tags: "General",
    img: "images/papers-images/img-20250908-1.png",
    link: "https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13801/138011N/Robotic-control-optimization-based-on-receptance-weighted-reinforcement-learning/10.1117/12.3076952.short"
  },
  {
    title: "Spectral Channel Mixing Transformer with Spectral-Center Attention for Hyperspectral Image Classification",
    description: "This paper proposes TC-Former, a novel framework for hyperspectral image (HSI) classification that integrates RWKV linear attention with Transformer architecture to address high computational complexity in long sequence tasks. The TC-Former combines TimeMixFormer and HyperMixFormer modules. TimeMixFormer optimizes processing efficiency for long sequences using time decay weights, while HyperMixFormer enhances cross-channel interaction with a gated WKV mechanism. This innovative integration significantly improves HSI classification accuracy and reduces computational complexity, outperforming state-of-the-art algorithms on public datasets.",
    date: "2025-09-05",
    tags: "Image",
    img: "images/papers-images/img-20250905-1.png",
    link: "https://www.mdpi.com/2072-4292/17/17/3100"
  },
  {
    title: "AudioRWKV: Efficient and Stable Bidirectional RWKV for Audio Pattern Recognition",
    description: "This paper, based on RWKV7, proposes AudioRWKV (A-RWKV), an efficient and stable architecture for audio modeling. It addresses the O(L2) complexity of Transformers and the instability of Mamba models for long audio sequences. A-RWKV inherits RWKV7's recurrent formulation, replacing its 1D token-shift with a 2D depthwise separable convolution to capture local spectro-temporal patterns. Furthermore, it introduces a bidirectional WKV (Bi-WKV) kernel for global context modeling with linear complexity. Experiments show A-RWKV achieves performance parity with larger models while offering superior stability and speedup for long-form audio.",
    date: "2025-09-02",
    tags: "Audio",
    img: "images/papers-images/img-20250902-1.png",
    link: "https://arxiv.org/abs/2509.02167"
  },
  {
    title: "Hybrid CNN-RWKV with high-frequency enhancement for real-world chinese-english scene text image super-resolution",
    description: "This paper proposes a Hybrid CNN-RWKV with High-Frequency Enhancement (HCR-HFE) model for real-world Scene Text Image Super-Resolution (STISR). Leveraging the RWKV architecture's capability for long-distance modeling with linear computational complexity, HCR-HFE addresses the limitations of existing methods on complex Chinese characters. The model integrates a recurrent bidirectional WKV attention to establish 2D image dependencies, a high-frequency enhancement module, a multi-scale large kernel convolutional block, and multi-frequency channel attention. Extensive experiments on the Real-CE dataset demonstrate HCR-HFE's superior performance in text legibility, image fidelity, and perceptual quality, also showing broad applicability to general SR tasks.",
    date: "2025-08-30",
    tags: "Image",
    img: "images/papers-images/img-20250830-1.png",
    link: "https://link.springer.com/article/10.1007/s10489-025-06785-8"
  },
  {
    title: "Finch-LIC: Learned Image Compression with Gated Multihead Linear Attention",
    description: "This paper introduces FinchLIC, a novel linear attention-based learned image compression architecture that builds upon RWKV-like mechanisms. It proposes Multihead Bi-RWKV blocks to enhance feature extraction by expanding the scale of internal states, analogous to multi-head attention. Furthermore, a K-Manhattan distance token shift (KMshift) method is introduced to effectively model neighboring context and expand the receptive field. FinchLIC achieves competitive rate-distortion performance while maintaining linear computational complexity and lower GPU memory usage, demonstrating efficiency and effectiveness for high-resolution image compression.",
    date: "2025-08-28",
    tags: "Image",
    img: "images/papers-images/img-20250828-3.png",
    link: "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5413219"
  },
  {
    title: "DSM-Seg: A CNN-RWKV Hybrid Framework for Forward-Looking Sonar Image Segmentation in Deep-Sea Mining",
    description: "This paper introduces DSM-Seg, a novel CNN-RWKV hybrid framework for semantic segmentation of forward-looking sonar (FLS) images in deep-sea mining. Addressing challenges like noise and blurred boundaries, DSM-Seg integrates a Physical Prior-Based Semantic Guidance Module (PSGM) leveraging sonar-specific priors for boundary enhancement. Additionally, an RWKV-Based Global Fusion with Semantic Constraints (RGFSC) module is introduced to manage long-range dependencies and fuse local/global semantic information. Experiments on deep-sea terrain and marine debris datasets demonstrate significant improvements in segmentation accuracy and real-time performance, crucial for deep-sea mining vehicles.",
    date: "2025-08-28",
    tags: "Image",
    img: "images/papers-images/img-20250828-2.png",
    link: "https://www.mdpi.com/2072-4292/17/17/2997"
  },
  {
    title: "PointDGRWKV: Generalizing RWKV-like Architecture to Unseen Domains for Point Cloud Classification",
    description: "This paper introduces PointDGRWKV, the first RWKV-based framework specifically designed for Domain Generalization in Point Cloud Classification (DG PCC). It addresses challenges of applying RWKV to unstructured point clouds, such as spatial distortions from fixed token shifts and attention drift from exponential weighting. The proposed PointDGRWKV incorporates Adaptive Geometric Token Shift (AGT-Shift) for improved local geometric modeling and Cross-Domain Key Feature Distribution Alignment (CD-KDA) to enhance cross-domain robustness. Experiments demonstrate state-of-the-art performance on DG PCC benchmarks while maintaining RWKV's linear efficiency.",
    date: "2025-08-28",
    tags: "3D/4D",
    img: "images/papers-images/img-20250828-1.png",
    link: "https://arxiv.org/abs/2508.20835"
  },
  {
    title: "VAFTrack: asynchronous feature fusion via visual receptive weighted key-value perceptual for visual tracking",
    description: "This paper, based on the Receptance Weighted Key-Value (RWKV) model, introduces VAFTrack, an Asynchronous Fusion Tracking Model for Visual Object Tracking. It addresses challenges of insufficient feature fusion and redundancy by proposing a Visual Receptive Weighted Key-Value Perceptual Fusion Module (VPFM). VPFM integrates Contextual Spatial and Channel Perception Modules, employing a bidirectional attention mechanism and Quad-Directional Token Shift, alongside Spatial and Channel Optimization Enhancement Modules. VAFTrack achieves state-of-the-art results, enhancing tracking accuracy and robustness on datasets like TrackingNet, LaSOT, and GOT-10k by improving target awareness and adaptability to scene variations.",
    date: "2025-08-21",
    tags: "Image",
    img: "images/papers-images/img-20250821-1.png",
    link: "https://link.springer.com/article/10.1007/s00530-025-01913-3"
  },
  {
    title: "REB-former: RWKV-enhanced E-branchformer for Speech Recognition",
    description: "This paper introduces REB-former, an RWKV-enhanced E-Branchformer, to address the quadratic complexity of Transformer-based ASR models. It interleaves E-Branchformer and RWKV layers, proposing the GroupBiRWKV module to enable efficient bidirectional contextual feature capture and overcome RWKV's inherent unidirectional limitation. The model also incorporates an RWKVDecoder to further enhance temporal modeling. Experimental results demonstrate that REB-former achieves state-of-the-art performance on the LibriSpeech 100h dataset, with improved computational efficiency and a significant reduction in word error rate.",
    date: "2025-08-17",
    tags: "Audio",
    img: "images/papers-images/img-20250817-1.png",
    link: "https://www.isca-archive.org/interspeech_2025/song25b_interspeech.html"
  },
  {
    title: "A Multimodal Bone Stick Matching Approach Based on Large-Scale Pre-Trained Models and Dynamic Cross-Modal Feature Fusion",
    description: "This paper, based on RWKV-derived models (Vision-RWKV for images and RWKV for inscriptions), proposes a multimodal method for matching fragmented bone sticks. It integrates image, inscription, and archeological metadata using pre-trained models (Vision-RWKV, RWKV, BERT) and a dynamic cross-modal feature fusion mechanism. The approach achieves 94.73% matching accuracy at Rank-15 by effectively handling fractures, corrosion, and missing sections, outperforming traditional methods.",
    date: "2025-08-05",
    tags: "Image",
    img: "images/papers-images/img-20250805-1.png",
    link: "https://www.mdpi.com/2076-3417/15/15/8681"
  },
  {
    title: "Monthly Service Prediction for 4G/5G Systems: A Short Time Series Based Neural Network Solution",
    description: "This paper, leveraging RWKV in its encoder, proposes a framework for monthly service prediction in 4G/5G networks. It introduces deep temporal clustering representation (DTCR) using RWKV-based encoding to cluster short time series data, followed by a decreasing time-difference network (DTD-Net) that crops input features block-wise to prevent overfitting. The solution achieves low prediction errors on real-world mobile network data, addressing challenges of limited data length and chaotic internal logic.",
    date: "2025-07-30",
    tags: "Sequence",
    img: "images/papers-images/img-20250730-1.png",
    link: "https://ieeexplore.ieee.org/abstract/document/11104274"
  },
  {
    title: "RWKV-Receptance Recurrent Key Value in the field of Speaker Diarization",
    description: "This paper, based on RWKV, introduces a novel approach to speaker diarization by integrating the RWKV architecture with End-to-End Neural Diarization (EEND). RWKV combines recurrent and transformer-like components to enable linear-time processing and memory efficiency. By replacing attention modules in EEND with RWKV-based time-mixing and channel-mixing blocks, the proposed RWKV-EEND framework efficiently handles long audio contexts. Evaluations show reduced diarization error rates and faster inference, making it suitable for real-time and low-resource audio systems.",
    date: "2025-07-29",
    tags: "Audio",
    img: "images/papers-images/img-20250729-1.png",
    link: "https://openreview.net/forum?id=5WG3x1hgdN"
  },
  {
    title: "SpikeRWKV:Energy-efficient Large Language Model with Spiking Neural Network",
    description: "This paper introduces SpikeRWKV, an energy-efficient language model based on the RWKV architecture that integrates Spiking Neural Networks (SNNs). To address the performance and energy trade-offs in traditional SNNs, the authors propose a novel Multi-head Spike Encoding scheme. This method enables parallel processing and hierarchical decomposition of spikes to enhance computational efficiency and fidelity. Experiments show SpikeRWKV significantly reduces energy consumption compared to the non-spiking RWKV model while achieving superior performance on natural language understanding tasks, including lower perplexity and improved bits-per-character scores.",
    date: "2025-07-27",
    tags: "Language",
    img: "images/papers-images/img-20250727-1.png",
    link: "http://poster-openaccess.com/files/ICIC2025/4202.pdf"
  },
  {
    title: "LowKeyEMG: Electromyographic typing with a reduced keyset",
    description: "This paper leverages the RWKV recurrent transformer language model to develop LowKeyEMG, a real-time interface enabling efficient text entry via surface electromyography (sEMG) with only 7 gesture keys. The system reduces the alphabet to 4 keys plus controls, using RWKV for beam search to predict words from sparse inputs. Experiments show participants achieved 23.3 words per minute with 99.2% top-3 accuracy, demonstrating reliable typing for motor-impaired users and constrained-input scenarios.",
    date: "2025-07-26",
    tags: "Language",
    img: "images/papers-images/img-20250726-1.png",
    link: "https://arxiv.org/abs/2507.19736"
  },
  {
    title: "Smooth Reading: Bridging the Gap of Recurrent LLM to Self-Attention LLM on Long-Context Tasks",
    description: "This paper, based on RWKV and other recurrent LLMs, proposes Smooth Reading—a chunk-wise inference method inspired by human reading strategies to address recurrent models' limitations in long-context tasks. By iteratively processing context in chunks and summarizing information, it reduces memory demands while preserving linear computational complexity. Experiments show RWKV and sliding-window LLMs with Smooth Reading match or exceed self-attention LLMs' performance on benchmarks like LongBench and NIAH, achieving 3× faster training and 2× faster inference at 64k context lengths.",
    date: "2025-07-25",
    tags: "General",
    img: "images/papers-images/img-20250725-1.png",
    link: "https://arxiv.org/abs/2507.19353"
  },
  {
    title: "DRWKV: Focusing on Object Edges for Low-Light Image Enhancement",
    description: "This paper, based on RWKV, proposes the DRWKV model for low-light image enhancement, prioritizing object edge preservation. It integrates Global Edge Retinex theory to decouple illumination and edge structures, introduces Evolving WKV Attention for spatial continuity modeling, and employs a Bilateral Spectrum Aligner with MS²-Loss for color alignment. Extensive benchmarks demonstrate superior PSNR, SSIM, and NIQE performance with low computational complexity, while enhancing downstream object tracking tasks.",
    date: "2025-07-24",
    tags: "Image",
    img: "images/papers-images/img-20250724-1.png",
    link: "https://arxiv.org/abs/2507.18594"
  },
  {
    title: "MSFF-RWKV : Single-Structure Multi-stage Feature Fusion Lightweight Super-Resolution Network",
    description: "This paper introduces MSFF-RWKV, a lightweight super-resolution network based on the RWKV architecture. It proposes a multi-stage feature fusion strategy using a single RWKV block that recursively integrates outputs with previous features to reduce parameters and computational costs. The model incorporates a Local Pixel Perception layer for adaptive pixel-level interactions and an ME-Shift module for multi-scale feature extraction. Experiments demonstrate state-of-the-art performance with a 0.14 dB PSNR gain and 26.6% parameter reduction.",
    date: "2025-07-15",
    tags: "Image",
    img: "images/papers-images/img-20250715-3.png",
    link: "https://link.springer.com/chapter/10.1007/978-981-96-9949-0_35"
  },
  {
    title: "An Efficient Image Fusion Network Exploiting Unifying Language and Mask Guidance",
    description: "This paper, based on RWKV, proposes an efficient image fusion framework leveraging language descriptions and semantic masks as guidance. It adapts RWKV into a bidirectional version using an efficient scanning strategy for image modality and introduces a multi-modal fusion module to integrate language and mask features. The lightweight network achieves state-of-the-art results across visible-infrared, multi-exposure, multi-focus, medical, hyperspectral-multispectral fusion, and pansharpening tasks.",
    date: "2025-07-23",
    tags: "Image",
    img: "images/papers-images/img-20250723-1.png",
    link: "https://ieeexplore.ieee.org/abstract/document/11091495"
  },
  {
    title: "U-RWKV: Lightweight medical image segmentation with direction-adaptive RWKV",
    description: "This paper proposes U-RWKV, a lightweight medical image segmentation framework leveraging the RWKV architecture. It introduces the Direction-Adaptive RWKV Module (DARM) with Dual-RWKV and QuadScan mechanisms to efficiently capture long-range dependencies while mitigating directional bias, and the Stage-Adaptive Squeeze-and-Excitation Module (SASE) to dynamically adapt feature extraction across stages. Experiments demonstrate state-of-the-art segmentation performance with high computational efficiency, making it suitable for resource-constrained healthcare settings.",
    date: "2025-07-15",
    tags: "Image",
    img: "images/papers-images/img-20250715-1.png",
    link: "https://arxiv.org/abs/2507.11415"
  },
  {
    title: "DEVR: Train an Efficient Vision-RWKV Model with Improved Knowledge Distillation",
    description: "This paper, based on RWKV, introduces DEVR, an efficient Vision-RWKV model enhanced through knowledge distillation. The authors redesign the RWKV block for vision tasks by adding 1D convolutions and a reversed-input branch to capture spatial details and improve channel interactions. They propose a novel distillation loss combining contrastive learning and traditional knowledge distillation to align features with a CNN teacher model. Evaluations demonstrate DEVR outperforms vanilla Vision-RWKV and DeiT in image classification, detection, and segmentation tasks while reducing computational costs and accelerating inference.",
    date: "2025-07-15",
    tags: "Image",
    img: "images/papers-images/img-20250715-2.png",
    link: "https://link.springer.com/chapter/10.1007/978-981-96-9794-6_29"
  },
  {
    title: "Scaling Context Requires Rethinking Attention",
    description: "This paper introduces power attention, an architectural layer derived from linear attention principles like those in RWKV, designed for efficient long-context sequence modeling. It addresses limitations of transformers and sub-quadratic architectures by enabling independent adjustment of state size via hyperparameter p, achieving balanced weight-state FLOP ratios. The authors develop optimized GPU kernels and demonstrate power attention's superiority in in-context learning and loss-per-FLOP over exponential and linear attention at long sequences.",
    date: "2025-07-06",
    tags: "General",
    img: "images/papers-images/img-20250706-1.png",
    link: "https://arxiv.org/abs/2507.04239"
  },
  {
    title: "AuroraLong: Bringing RNNs Back to Efficient Open-Ended Video Understanding",
    description: "This paper proposes AURORA LONG, a model that replaces the LLM component in MLLMs with a linear RNN language model (RWKV) to handle long video inputs efficiently. By combining visual token merging with linear RNN models and reordering tokens by size, the model achieves constant memory cost and high throughput. Despite having only 2B parameters and being trained on public data, AURORA LONG matches the performance of larger Transformer-based models on multiple video benchmarks, demonstrating the potential of linear RNNs for democratizing long video understanding.",
    date: "2025-07-03",
    tags: "3D/4D",
    img: "images/papers-images/img-20250703-1.png",
    link: "https://arxiv.org/abs/2507.02591"
  },
  {
    title: "EvRWKV: A RWKV Framework for Effective Event-guided Low-Light Image Enhancement",
    description: "This paper, based on RWKV, introduces EvRWKV, a novel framework for event-guided low-light image enhancement. It leverages a Cross-RWKV module for fine-grained temporal and cross-modal fusion between event and image data, and an Event Image Spectral Fusion Enhancer (EISFE) module for adaptive noise suppression and spatial alignment in frequency and spatial domains. Extensive experiments on real-world datasets demonstrate state-of-the-art performance in suppressing noise and restoring details in challenging low-light conditions.",
    date: "2025-07-01",
    tags: "Image",
    img: "images/papers-images/img-20250701-1.png",
    link: "https://arxiv.org/abs/2507.03184"
  },
  {
    title: "Out-of-Distribution Semantic Occupancy Prediction",
    description: "This paper introduces OccOoD, a framework integrating out-of-distribution (OoD) detection into 3D semantic occupancy prediction for autonomous driving. It proposes a Synthetic Anomaly Integration Pipeline to create datasets (VAA-KITTI and VAA-KITTI-360) and leverages an RWKV-based branch in Voxel-BEV Progressive Fusion to enhance OoD detection. Experimental results demonstrate state-of-the-art OoD detection performance while maintaining competitive occupancy prediction accuracy.",
    date: "2025-06-26",
    tags: "3D/4D",
    img: "images/papers-images/img-20250626-1.png",
    link: "https://arxiv.org/abs/2506.21185"
  },
  {
    title: "Accurate, fast, cheap: Choose three. Replacing Multi-Head-Attention with Bidirectional Recurrent Attention for Long-Form ASR",
    description: "This paper investigates replacing multi-head attention (MHA) with bidirectional recurrent attention (RA) layers, specifically RWKV and Mamba, in long-form automatic speech recognition (ASR) models. It demonstrates that bidirectional RA layers can match MHA accuracy while being more efficient, introduces Direction Dropout for improved accuracy/throughput trade-off, and presents a new alternating directions decoding mode.",
    date: "2025-06-24",
    tags: "Audio",
    img: "images/papers-images/img-20250624-1.png",
    link: "https://arxiv.org/abs/2506.19761"
  },
  {
    title: "SMNet: A Semantic Guided Mamba Network for Remote Sensing Change Detection",
    description: "This paper introduces SMNet, a remote sensing change detection model that integrates RWKV and Mamba architectures to enhance feature representation and global dependency capture. The model employs a Learnable Visual State Space (L-VSS) block, a multi-directional WKV (Mi-WKV) attention mechanism, and a Heterogeneous Pixel Fusion (HPF) module to improve semantic variation detection and feature circulation. Comprehensive evaluations on CD datasets demonstrate SMNet's superior performance compared to leading-edge techniques.",
    date: "2025-06-18",
    tags: "Image",
    img: "images/papers-images/img-20250618-1.png",
    link: "https://ieeexplore.ieee.org/abstract/document/11039697"
  },
  {
    title: "Exploring Diffusion with Test-Time Training on Efficient Image Restoration",
    description: "This paper introduces DiffRWKVIR, a framework integrating Test-Time Training with efficient diffusion for image restoration. It extends RWKV's parameterization to 2D scanning for global context awareness, accelerates processing via chunk-wise parallelism, and extracts compact image priors for faster training/inference. The method outperforms existing models in super-resolution and inpainting tasks.",
    date: "2025-06-17",
    tags: "Image",
    img: "images/papers-images/img-20250617-1.png",
    link: "https://arxiv.org/abs/2506.14541"
  },
  {
    title: "Blind Identification of Collective Motion Criticality using Sequence Model Predictive Entropy Variance",
    description: "This paper proposes a parameter-agnostic method using the RWKV-7 sequence model to detect critical transitions in collective motion systems by analyzing single-agent trajectory data. The model's predictive entropy variance peaks near critical noise levels, demonstrating robustness across system sizes and aligning with finite-size scaling principles.",
    date: "2025-06-16",
    tags: "Sequence",
    img: "images/papers-images/img-20250616-3.png",
    link: "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5297784"
  },
  {
    title: "Personalizable Long-Context Symbolic Music Infilling with MIDI-RWKV",
    description: "This paper introduces MIDI-RWKV, a novel model based on the RWKV-7 architecture, designed for personalizable, multi-track, long-context, and controllable symbolic music infilling. MIDI-RWKV enables efficient and coherent musical cocreation on edge devices and demonstrates effective finetuning for personalization in low-sample regimes. The model is evaluated on quantitative and qualitative metrics, showing superior performance in long-context infilling tasks compared to existing approaches.",
    date: "2025-06-16",
    tags: "Audio",
    img: "images/papers-images/img-20250616-1.png",
    link: "https://arxiv.org/abs/2506.13001"
  },
  {
    title: "A Parallel Processing Architecture for Long-Term Power Load Forecasting",
    description: "This paper proposes MP-RWKV, an enhanced architecture based on RWKV-TS, addressing challenges in long-term power load forecasting through parallel processing paths for temporal modeling. The model demonstrates superior performance over state-of-the-art baselines, maintaining robust accuracy across short-term and long-term horizons.",
    date: "2025-06-16",
    tags: "Sequence",
    img: "images/papers-images/img-20250616-2.png",
    link: "https://www.mdpi.com/2673-4591/97/1/26"
  },
  {
    title: "RWKV-IF: Efficient and Controllable RNA Inverse Folding via Attention-Free Language Modeling",
    description: "This paper introduces RWKV-IF, an attention-free RWKV language model-based framework for RNA inverse folding, treating structure-to-sequence generation as conditional language modeling. It employs Top-k sampling, temperature control, and G-C content biasing to generate accurate and biophysically meaningful sequences. The model demonstrates superior performance over traditional search-based methods, achieving higher accuracy and full match rates while reducing edit distance.",
    date: "2025-06-14",
    tags: "Sequence",
    img: "images/papers-images/img-20250614-1.png",
    link: "https://www.biorxiv.org/content/10.1101/2025.06.13.659654v1"
  },
  {
    title: "Med-URWKV: Pure RWKV With ImageNet Pre-training For Medical Image Segmentation",
    description: "This paper introduces Med-URWKV, a pure RWKV-based architecture for medical image segmentation that incorporates ImageNet pre-training via a pre-trained VRWKV encoder. It demonstrates comparable or superior performance to existing RWKV models trained from scratch, validating the effectiveness of leveraging pre-trained RWKV models in medical segmentation tasks.",
    date: "2025-06-12",
    tags: "Image",
    img: "images/papers-images/img-20250612-1.png",
    link: "https://arxiv.org/abs/2506.10858"
  },
  {
    title: "Vision-QRWKV: Exploring Quantum-Enhanced RWKV Models for Image Classification",
    description: "This paper introduces Vision-QRWKV, a hybrid quantum-classical extension of the RWKV architecture, integrating a variational quantum circuit into its channel mixing component for image classification tasks. The quantum-enhanced model outperforms its classical counterpart on datasets with subtle or noisy class distinctions, demonstrating potential for lightweight and efficient vision tasks.",
    date: "2025-06-07",
    tags: "Image",
    img: "images/papers-images/img-20250607-1.png",
    link: "https://arxiv.org/abs/2506.06633"
  },
  {
    title: "VisualRWKV-HM: Enhancing linear visual-language models via hybrid mixing",
    description: "This paper introduces VisualRWKV-HM, a linear complexity visual-language model that incorporates a hybrid mixing mechanism combining time mixing and cross state mixing. The model achieves state-of-the-art performance across single-image, multi-image, and multi-view benchmarks, demonstrating high computational efficiency and scalability. It significantly outperforms the vanilla VisualRWKV and other Transformer-based models in terms of speed and memory usage.",
    date: "2025-06-06",
    tags: "Image",
    img: "images/papers-images/img-20250606-1.png",
    link: "https://authors.elsevier.com/a/1lDfB5a7-G-6z3"
  },
  {
    title: "FEAT: Full-Dimensional Efficient Attention Transformer for Medical Video Generation",
    description: "Based on the WKV attention mechanism within the RWKV model architecture, this paper introduces the FEAT model, which elegantly addresses the challenges of inadequate channel interaction, prohibitive computational complexity, and crude denoising guidance in medical video generation through a unified spatial-temporal-channel attention framework. FEAT employs a linear complexity attention design that seamlessly integrates global channel attention with residual value guidance modules, delivering efficient and superior-quality medical video generation across diverse datasets.",
    date: "2025-06-05",
    tags: "3D/4D",
    img: "images/papers-images/img-20250605-1.png",
    link: "https://arxiv.org/abs/2506.04956"
  },
  {
    title: "Pan-Sharpening via Causal-Aware Feature Distribution Calibration",
    description: "This paper addresses frequency imbalance in pan-sharpening by leveraging causal inference to identify optimizer momentum as a confounding factor. It proposes a novel RWKV-based architecture with global receptive fields to model long-tailed high-frequency distributions and employs counterfactual reasoning for feature calibration, achieving state-of-the-art performance across benchmark datasets.",
    date: "2025-06-04",
    tags: "Image",
    img: "images/papers-images/img-20250604-1.png",
    link: "https://ieeexplore.ieee.org/abstract/document/11023855"
  },
  {
    title: "Diet-Seg: Dynamic Hardness-Aware Learning for Enhanced Brain Tumor Segmentation",
    description: "This paper introduces Diet-Seg, a novel brain tumor segmentation framework that integrates entropy-based pixel-wise hardness estimation with dynamic learning rate modulation. It employs an RWKV-based U-Net backbone to capture global spatial dependencies and an EdgeNet module to preserve tumor boundaries, achieving superior performance on BraTS datasets compared to state-of-the-art methods.",
    date: "2025-06-03",
    tags: "Image",
    img: "images/papers-images/img-20250603-1.png",
    link: "https://www.biorxiv.org/content/10.1101/2025.05.31.657149v1"
  },
  {
    title: "Relational Context Modeling for Improved Knowledge Graph Completion",
    description: "This paper proposes RCME, a hybrid model integrating RWKV for sequential modeling and dynamic embeddings with TuckER for robust relational decoding. The approach addresses limitations in existing knowledge graph completion methods by capturing contextual nuances and temporal dynamics, achieving superior performance on benchmark datasets.",
    date: "2025-06-01",
    tags: "Language",
    img: "images/papers-images/img-20250601-1.png",
    link: "https://www.engineeringletters.com/issues_v33/issue_6/EL_33_6_28.pdf"
  },
  {
    title: "URWKV: Unified RWKV Model with Multi-state Perspective for Low-light Image Restoration",
    description: "This paper, based on RWKV, introduces a Unified Receptance Weighted Key Value (URWKV) model for low-light image restoration, addressing dynamically coupled degradations through a multi-state perspective. The model customizes the RWKV block to perceive complex degradations using intra- and inter-stage states, featuring Luminance-adaptive Normalization (LAN) for scene-aware luminance modulation and a State-aware Selective Fusion (SSF) module for dynamic feature integration. URWKV outperforms state-of-the-art models with fewer parameters and computational resources.",
    date: "2025-05-29",
    tags: "Image",
    img: "images/papers-images/img-20250529-1.png",
    link: "https://arxiv.org/abs/2505.23068"
  },
  {
    title: "Thyroid nodule segmentation method integrating receiving weighted key-value architecture and spherical geometric features",
    description: "This paper proposes a thyroid nodule segmentation method integrating the receiving weighted key-value (RWKV) architecture and spherical geometry feature (SGF) sampling technology to address high computational complexity and image detail loss in ultrasound thyroid nodule segmentation. The method achieves precise segmentation through two-dimensional offset prediction and pixel-level sampling adjustments, and introduces a patch attention module (PAM) to optimize decoder feature maps, demonstrating superior performance on TN3K and DDTI datasets.",
    date: "2025-05-29",
    tags: "Image",
    img: "images/papers-images/img-20250529-2.png",
    link: "https://pubmed.ncbi.nlm.nih.gov/40566780/"
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
