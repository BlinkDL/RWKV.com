const allProjects = [
  {
    title:
    "StyleRWKV: High-Quality and High-Efficiency Style Transfer with RWKV-like Architecture",
    description:
      "This paper presents StyleRWKV, a new style transfer method. It adopts an architecture inspired by RWKV to resolve the shortcomings of previous approaches, such as high computational complexity. By means of crucial elements like the Re-WKV attention mechanism, it accomplishes efficient and high-quality style transfer.",
    date: "2024-12-27",
    tags: "Image",
    img: "images/papers-images/style-RWKV-architecture.png",
    link: "https://arxiv.org/abs/2412.19535",
  },
  {
    title:
      "L3TC: Leveraging RWKV for Learned Lossless Low-Complexity Text Compression",
    description:
      "This paper presents L3TC, a novel text compression method. It selects RWKV for its fast decoding speed. With an outlier-aware tokenizer and high-rank reparameterization, L3TC achieves 48% bit saving vs gzip, 50× param reduction, and is the fastest learned compressor.",
    date: "2024-12-21",
    tags: "Language",
    img: "images/papers-images/L3TC-architecture.png",
    link: "https://arxiv.org/abs/2412.16642",
  },
  {
    title: "A Survey of RWKV",
    description:
      "A collection of papers and resources related to a survey of RWKV.",
    date: "2024-12-19",
    tags: "General",
    img: "images/papers-images/RWKV_survey.png",
    link: "https://github.com/MLGroupJLU/RWKV-Survey",
  },
  {
    title:
      "PCF-RWKV: Product Carbon Footprint Estimation System Based on Large Language Model",
    description:
      "The paper presents PCF-RWKV, a product carbon footprint assessment model built on the RWKV architecture, featuring stacked residual blocks and three task-specific LoRA adapters. Through Multi-Agents technology integration, the model automates LCI construction for production processes and matches them with emission factors to calculate carbon footprints, enhancing efficiency and security in enterprise carbon footprint assessment while overcoming traditional method limitations.",
    date: "2024-12-18",
    tags: "Language",
    img: "images/papers-images/PCF-RWKV-architecture.png",
    link: "https://www.preprints.org/manuscript/202412.1705/v1",
  },
  {
    title: "RWKV-edge: Deeply Compressed RWKV for Resource-Constrained Devices",
    description:
      "The paper introduces RWKV-edge, a solution for running RWKV models on resource-constrained devices. Using techniques like low-rank approximation, sparsity prediction, and clustered heads, it achieves 4.95-3.8x model compression with only 2.95pp accuracy loss. RWKV-edge provides an effective approach for deploying RWKV models on edge devices.",
    date: "2024-12-14",
    tags: "General",
    img: "images/papers-images/RWKV-edge.png",
    link: "https://arxiv.org/abs/2412.10856",
  },
  {
    title:
      "Linear Attention Based Channel Estimation Scheme for V2X Communications",
    description:
      "This paper proposes an innovative channel estimation scheme for V2X communications. Considering the doubly selective fading and limited pilots in IEEE 802.11p standard, it introduces the RWKV network with linear attention combined with DPA. The RWKV-DPA estimator enhances performance and reduces complexity compared to existing ones.",
    date: "2024-12-13",
    tags: "Sequence",
    img: "images/papers-images/RWkV-DPA-estimator-architecture.png",
    link: "https://ieeexplore.ieee.org/abstract/document/10779439",
  },
  {
    title:
      "Exploring Real&Synthetic Dataset and Linear Attention in Image Restoration",
    description:
      "The paper proposes RWKV-IR, a novel RWKV-based image restoration model that supports both global and local receptive fields. The model demonstrates superior performance on Urban100 x4 benchmark, achieving 0.08dB improvement over SwinIR and 0.03dB over MambaIR, showcasing RWKV-IR's advanced image restoration capabilities and fast convergence.",
    date: "2024-12-11",
    tags: "Image",
    img: "images/papers-images/RWKV-IR-architecture.png",
    link: "https://arxiv.org/abs/2412.03814",
  },
  {
    title:
      "DFT: A Dual-branch Framework of Fluctuation and Trend for Stock Price Prediction",
    description:
      "The paper proposes a Dual-branch Framework of Fluctuation and Trend (DFT) for stock price prediction. The RWKV model is used in the DFT to model time correlations in both the fluctuation and trend branches. It combines the power of RNN and Transformer, maintaining the time sequence of input features and satisfying the causality of the input. This helps in effectively capturing short-term fluctuations and trend information from stocks while explicitly modeling temporal variations and causal correlations, leading to improved performance compared to existing methods",
    date: "2024-11-09",
    tags: "Sequence",
    img: "images/papers-images/DFT.png",
    link: "https://arxiv.org/abs/2411.06065",
  },
  {
    title: "Video RWKV: Video Action Recognition Based RWKV",
    description:
      "The paper proposes LSTM CrossRWKV (LCR) for video understanding. It uses a novel CrossRWKV gate to handle video challenges. LCR stores long-term memory and reduces redundant information. Experiments on datasets show its effectiveness, setting a new benchmark in video understanding with RWKV.",
    date: "2024-11-08",
    tags: "3D/4D",
    img: "images/papers-images/video-rwkv.png",
    link: "https://arxiv.org/abs/2411.05636",
  },
  {
    title:
      "From Explicit Rules to Implicit Reasoning in an Interpretable Violence Monitoring System",
    description:
      "The paper proposes RuleVM for weakly supervised violence monitoring. It uses a dual-branch structure with different designs for images and text. The implicit branch uses visual features for coarse-grained classification, and the explicit branch uses language-image alignment with YOLO-World and data mining. RWKV is used in the lightweight time-series module.",
    date: "2024-10-29",
    tags: "3D/4D",
    img: "images/papers-images/pipeline-of-RuleVM-system.png",
    link: "https://arxiv.org/abs/2410.21991",
  },
  {
    title:
      "Modern Sequence Models in Context of Multi-Agent Reinforcement Learning",
    description:
      "The paper focuses on MARL. It proposes MAM and MARWKV architectures inspired by MAT. Experiments show they perform comparably to MAT. MARWKV offers better inference computational efficiency, especially with more agents. RWKV is used in MARWKV for sequence modeling.",
    date: "2024-10-28",
    tags: "Sequence",
    img: "images/papers-images/marwkv-architecture.png",
    link: "https://epub.jku.at/obvulihs/content/titleinfo/10580112",
  },
  {
    title:
      "MATCC: A Novel Approach for Robust Stock Price Prediction Incorporating Market Trends and Cross-time Correlations",
    description:
      "Stock price prediction is challenging. Existing work has limitations. This paper proposes MATCC, a novel framework. It extracts market trends, decomposes stock data, and mines cross-time correlation. Experiments show MATCC outperforms previous works. It uses RWKV to model inter-temporal correlations",
    date: "2024-10-21",
    tags: "Sequence",
    img: "images/papers-images/matcc-cumulative-return-comparison.png",
    link: "https://dl.acm.org/doi/abs/10.1145/3627673.3679715",
  },
  {
    title:
      "VisualRWKV-HD and UHD: Advancing High-Resolution Processing for Visual Language Models",
    description:
      "The paper presents VisualRWKV-HD and VisualRWKV-UHD for high-resolution visual inputs in visual language models. It details techniques like lossless downsampling and image segmentation. Experiments on benchmarks show their effectiveness, with RWKV models achieving better performance in handling high-resolution tasks.",
    date: "2024-10-15",
    tags: "Image",
    img: "images/papers-images/VisualRWKV-HD-UHD-Architecture_Design.png",
    link: "https://arxiv.org/abs/2410.11665",
  },
  {
    title:
      "AttnInput: Revolutionizing Pinyin Input with Context-Aware RWKV Language Models",
    description:
      "The paper presents AttnInput, a novel approach leveraging RWKV for Pinyin IME. It integrates Pinyin into RWKV's internal state, addressing semantic discontinuity. Using a pre-training strategy, it reduces costs. Experimental results show it achieves state-of-the-art performance on abbreviated Pinyin input.",
    date: "2024-10-13",
    tags: "Language",
    img: "images/papers-images/rwkv-attninput-architecture.png",
    link: "https://openreview.net/forum?id=9OxTqscUwi",
  },
  {
    title:
      "OccRWKV: Rethinking Efficient 3D Semantic Occupancy Prediction with Linear Complexity",
    description:
      "The paper presents OccRWKV, an efficient 3D semantic occupancy network inspired by RWKV. It separates predictions into branches with Sem-RWKV and GeoRWKV blocks. By projecting features to BEV space and using BEV-RWKV block, it achieves real-time inference. It outperforms state-of-the-art methods on SemanticKITTI dataset",
    date: "2024-09-26",
    tags: "3D/4D",
    img: "images/papers-images/occrwkv-architecture.jpg",
    link: "https://www.arxiv.org/abs/2409.19987",
  },
  {
    title:
      "Bone: Block Affine Transformation as Parameter Efficient Fine-tuning Methods for Large Language Models",
    description:
      "The paper introduces Bone, a new PEFT method. It divides LLM weights into subspaces and uses a shared matrix, differing from LoRA. It surpasses LoRA and its variants. The combination with Weight Guide and the development of Bat enhance its performance. Experiments on RWKV and other models confirm its efficacy.",
    date: "2024-09-19",
    tags: "General",
    img: "images/papers-images/bone.png",
    link: "https://arxiv.org/abs/2409.15371",
  },
  {
    title:
      "Multi-scale RWKV with 2-dimensional temporal convolutional network for short-term photovoltaic power forecasting",
    description:
      "The paper proposes MSRWKV-2DTCN for short-term PV power forecasting. It uses FFT to identify periodicity, combines RWKV with a multi-scale 2D TCN, and conducts experiments on real datasets. The model shows high accuracy and strong generalization capabilities.",
    date: "2024-09-06",
    tags: "Sequence",
    img: "images/papers-images/MSRWKV-2DTCN-architecture.png",
    link: "https://www.sciencedirect.com/science/article/abs/pii/S0360544224028433",
  },
  {
    title: "Experimentation in Content Moderation using RWKV",
    description:
      "The paper investigates RWKV's efficacy in content moderation. It creates a novel dataset for distillation, generates responses using LLMs, and fine-tunes RWKV. The study shows RWKV can improve content moderation accuracy and efficiency, and paves the way for more efficient models.",
    date: "2024-09-05",
    tags: "Language",
    img: "images/papers-images/mod-rwkv-architecture.png",
    link: "https://arxiv.org/abs/2409.03939",
  },
  {
    title:
      "Temporal and Interactive Modeling for Efficient Human-Human Motion Generation",
    description:
      "The paper presents TIM for efficient human-human motion generation. It proposes Causal Interactive Injection, Role-Evolving Mixing, and Localized Pattern Amplification. Experiments on InterHuman show TIM's superiority, achieving state-of-the-art results with only 32% of InterGen's trainable parameters, using RWKV",
    date: "2024-08-30",
    tags: "3D/4D",
    img: "images/papers-images/rwkv-tim-architecture.png",
    link: "https://arxiv.org/abs/2408.17135",
  },
  {
    title:
      "OnlySportsLM: Optimizing Sports-Domain Language Models with SOTA Performance under Billion Parameter",
    description:
      "The paper explores a small sports-domain language model. It creates the OnlySports collection (dataset, benchmark, LM). Using 600 billion tokens data, it optimizes RWKV-v6 for sports tasks, training a 196M param model. OnlySportsLM outperforms prior models and rivals larger ones in the sports domain.",
    date: "2024-08-30",
    tags: "Language",
    img: "images/papers-images/onlysportslm-table.png",
    link: "https://arxiv.org/abs/2409.00286",
  },
  {
    title:
      "Why Perturbing Symbolic Music is Necessary: Fitting the Distribution of Never-used Notes through a Joint Probabilistic Diffusion Model",
    description:
      "The paper propose the Music-Diff architecture, which uses a joint probabilistic diffusion model. It improves note distribution fitting and sample diversity compared to language models like RWKV-music, enhancing rhythmic and structural coherence in generated music.",
    date: "2024-08-04",
    tags: "Audio",
    img: "images/papers-images/symb-rwkv-for-music-diff.png",
    link: "https://arxiv.org/abs/2408.01950",
  },
  {
    title:
      "Optimizing Robotic Manipulation with Decision-RWKV: A Recurrent Sequence Modeling Approach for Lifelong Learning",
    description:
      "The paper explores RWKV's integration with decision transformer and experience replay in robotic manipulation. It proposes the Decision-RWKV model, tests it on D4RL and D'Claw platforms, and shows its effectiveness in single-task and lifelong learning, with code open-sourced.",
    date: "2024-07-23",
    tags: "Sequence",
    img: "images/papers-images/Decision-RWKV-block-overview.png",
    link: "https://arxiv.org/abs/2407.16306",
  },
  {
    title:
      "BSBP-RWKV: Background Suppression with Boundary Preservation for Efficient Medical Image Segmentation",
    description:
      "The paper proposes BSBP-RWKV for accurate and efficient medical image segmentation. It combines the advantages of PMD and RWKV, devises DWT-PMD RWKV Block and Multi-Step Runge-Kutta convolutional Block, and proposes a novel loss function. Experiments show its superior accuracy and efficiency.",
    date: "2024-07-21",
    tags: "Image",
    img: "images/papers-images/BSBP-RWKV-architecture.png",
    link: "https://openreview.net/pdf?id=ULD5RCk0oo",
  },
  {
    title:
      "GoldFinch: High Performance RWKV/Transformer Hybrid with Linear Pre-Fill and Extreme KV-Cache Compression",
    description:
      "The paper presents GoldFinch, a hybrid Linear Attention/Transformer model. It uses a new technique to generate a highly compressed KV-Cache. GoldFinch stacks GOLD transformer on an enhanced RWKV-6 (Finch) architecture. It shows improved performance with reduced cache size compared to Finch and Llama.",
    date: "2024-07-16",
    tags: "General",
    img: "images/papers-images/GoldFinch-architecture.png",
    link: "https://arxiv.org/abs/2407.12077",
  },
  {
    title:
      "Restore-RWKV: Efficient and Effective Medical Image Restoration with RWKV",
    description:
      "The paper of this paper proposes Restore-RWKV, the first RWKV-based model for medical image restoration. It modifies RWKV's attention and token shift layers to handle 2D images, capturing global and local dependencies. Experiments show its superiority in various tasks, serving as an efficient and effective backbone.",
    date: "2024-07-14",
    tags: "Image",
    img: "images/papers-images/restore-rwkv-architecture.png",
    link: "https://arxiv.org/abs/2407.11087",
  },
  {
    title:
      "Mamba or RWKV: Exploring High-Quality and High-Efficiency Segment Anything Model",
    description:
      "RThe paper focuses on designing an efficient segment-anything model. It proposes RWKV-SAM with a mixed backbone of convolution and RWKV operation. This model achieves high accuracy and efficiency, outperforming others in benchmarks. It also trains on a combined high-quality dataset for better segmentation.",
    date: "2024-06-27",
    tags: "Image",
    img: "images/papers-images/rwkv-sam-architecture.png",
    link: "https://arxiv.org/abs/2406.19369",
  },
  {
    title:
      "VisualRWKV: Exploring Recurrent Neural Networks for Visual Language Models",
    description:
      "The paper presents VisualRWKV, the first application of the linear RNN model RWKV in multimodal learning. It proposes novel mechanisms like data-dependent recurrence. Experiments show it performs competitively compared to Transformer models, with efficient computation and memory usage.",
    date: "2024-06-19",
    tags: "Image",
    img: "images/papers-images/visual-rwkv-architecture.png",
    link: "https://arxiv.org/abs/2406.13362",
  },
  {
    title: "RWKV-CLIP: A Robust Vision-Language Representation Learner",
    description:
      "The paper explores CLIP from data and model architecture perspectives. It proposes a diverse description generation framework and RWKV-CLIP, the first RWKV-driven vision-language model. Experiments show RWKV-CLIP's robustness and effectiveness, achieving state-of-the-art performance in multiple downstream tasks.",
    date: "2024-06-11",
    tags: "Image",
    img: "images/papers-images/rwkv-clip-architecture.png",
    link: "https://arxiv.org/abs/2406.06973",
  },
  {
    title:
      "PointRWKV: Efficient RWKV-Like Model for Hierarchical Point Cloud Learning",
    description:
      "The paper proposes PointRWKV, a new model with linear complexity adapted from RWKV in NLP for 3D point cloud learning. It uses modified multi-headed matrix-valued states and a dynamic attention recurrence mechanism to explore global processing capabilities and a parallel branch to encode local geometric features, outperforming other models and saving FLOPs.",
    date: "2024-05-24",
    tags: "3D/4D",
    img: "images/papers-images/point-rwkv--architecture.png",
    link: "https://arxiv.org/abs/2405.15214",
  },
  {
    title:
      "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence",
    description:
      "The paper presents Eagle (RWKV-5) and Finch (RWKV-6), improving RWKV-4. Their architectural enhancements include multiheaded matrix-valued states and dynamic recurrence. New multilingual corpus and tokenizer are introduced. Trained models show competitive performance, and all are publicly released.",
    date: "2024-04-08",
    tags: "General",
    img: "images/papers-images/rwkv-5-6-architecture.png",
    link: "https://arxiv.org/abs/2404.05892",
  },
  {
    title:
      "Diffusion-RWKV: Scaling RWKV-Like Architectures for Diffusion Models",
    description:
      "The paper presents Diffusion-RWKV, an architecture adapting RWKV for diffusion models in image generation. It handles long-range hidden states linearly, showing comparable performance to Transformers but with lower complexity, thus being a promising alternative in this field.",
    date: "2024-04-06",
    tags: "Image",
    img: "images/papers-images/Diffusion-RWKV-architecture.png",
    link: "https://arxiv.org/abs/2404.04478",
  },
  {
    title:
      "Onboard deep lossless and near-lossless predictive coding of hyperspectral images with line-based attention",
    description:
      "Deep learning in spacecraft hyperspectral image compression was challenging. This paper designs LineRWKV, a predictive neural network. It uses a novel hybrid operation, combines Transformers & RNNs. LineRWKV outperforms CCSDS-123.0-B-2 in compression and shows good throughput on a 7W system.",
    date: "2024-03-26",
    tags: "Image",
    img: "images/papers-images/LineRWKV-architecture.png",
    link: "https://arxiv.org/abs/2403.17677",
  },
  {
    title:
      "Vision-RWKV: Efficient and Scalable Visual Perception with RWKV-Like Architectures",
    description:
      "The paper presents Vision-RWKV (VRWKV), an adaptation of the RWKV model for vision tasks. It offers efficient handling of sparse inputs and strong global processing, with reduced spatial aggregation complexity. VRWKV outperforms ViT in image classification and shows advantages in dense prediction tasks, being a promising alternative for visual perception.",
    date: "2024-03-07",
    tags: "Image",
    img: "images/papers-images/Vision-RWKV-architecture.png",
    link: "https://arxiv.org/abs/2403.02308",
  },
  {
    title:
      "TLS-RWKV: Real-Time Online Action Detection with Temporal Label Smoothing",
    description:
      "The paper proposes TLS-RWKV for online action detection. It utilizes the RWKV model with temporal label smoothing. Experiments on THUMOS'14 and TVSeries datasets show state-of-the-art performance and high efficiency, making it suitable for real-time applications and resource-constrained devices.",
    date: "2024-2-19",
    tags: "3D/4D",
    img: "images/papers-images/TLS-RWKV-architecture.png",
    link: "https://link.springer.com/article/10.1007/s11063-024-11540-0",
  },
  {
    title: "SDiT: Spiking Diffusion Model with Transformer",
    description:
      "The paper proposes Spiking Diffusion Transformer (SDiT), a novel SNN diffusion model. It uses RWKV for efficient self-attention. SDiT aims to provide a baseline for SNN generative models and shows competitiveness on multiple datasets, generating high-quality images with lower cost and shorter sampling time.",
    date: "2024-02-18",
    tags: "Image",
    img: "images/papers-images/sdit-architecture.png",
    link: "https://arxiv.org/abs/2402.11588",
  },
  {
    title:
      "RWKV-TS: Beyond Traditional Recurrent Neural Network for Time Series Tasks",
    description:
      "Traditional RNNs have declined in time series tasks. This paper presents RWKV-TS, an efficient RNN-based model. It has O(L) complexity, captures long-term info well, and is computationally efficient. RWKV-TS shows competitive performance with reduced latency and memory use in various tasks.",
    date: "2024-01-17",
    tags: "Sequence",
    img: "images/papers-images/rwkv-ts-architecture.png",
    link: "https://arxiv.org/abs/2401.09093",
  },
  {
    title:
      "Advancing VAD Systems Based on Multi-Task Learning with Improved Model Structures",
    description:
      "The paper proposes semantic VAD systems based on multi-task learning with improved models (RWKV for real-time, SAN-M for offline) to address issues in traditional binary VAD. Evaluations show significant improvements in CER, DCF, and NRR metrics compared to DFSMN-based systems.",
    date: "2023-12-19",
    tags: "Audio",
    img: "images/papers-images/rwkv-vad--architecture.png",
    link: "https://arxiv.org/abs/2312.14860",
  },
  {
    title: "RWKV-based Encoder-Decoder Model for Code Completion",
    description:
      "The paper presents an RWKV-based encoder-decoder model for code completion. It aims to address challenges in this area. The model shows good performance and has potential for improving code generation efficiency, but more research is needed for wider application and optimization.",
    date: "2023-11-17",
    tags: "Language",
    img: "images/papers-images/RWKV-Code-Completion.png",
    link: "https://ieeexplore.ieee.org/abstract/document/10442108",
  },
  {
    title:
      "RWKV: A Linear Attention Mechanism for Temperature and Humidity Compensation for Gas Sensors",
    description:
      "The paper presents a novel methodology for a PANI-CeO2 ammonia gas sensor to address temperature and humidity compensation. It uses the RWKV network with a Linear attention mechanism. The process has three stages. The method shows high predictive accuracy, with low mean absolute and relative errors.",
    date: "2023-10-25",
    tags: "Sequence",
    img: "images/papers-images/RWKV-for-Gas-Sensors.png",
    link: "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4612708",
  },
  {
    title: "Exploring RWKV for Memory Efficient and Low Latency Streaming ASR",
    description:
      "The paper proposes applying RWKV, a linear attention transformer variant, to streaming ASR. It combines transformer performance and RNN inference efficiency. Experiments show RWKV-Transducer and RWKV-Boundary-Aware-Transducer achieve good accuracy with minimal latency and memory cost.",
    date: "2023-09-26",
    tags: "Audio",
    img: "images/papers-images/RWKV-ASR-architecture.png",
    link: "https://arxiv.org/abs/2309.14758",
  },
  {
    title: "RWKV: Reinventing RNNs for the Transformer Era",
    description:
      "The paper proposes RWKV, a novel model architecture. It combines the efficient parallelizable training of transformers with the efficient inference of RNNs. RWKV uses a linear attention mechanism, scales to 14 billion parameters, and performs comparably to similar-sized transformers, advancing sequence processing tasks.",
    date: "2023-05-22",
    tags: "General",
    img: "images/papers-images/rwkv-4.png",
    link: "https://arxiv.org/abs/2305.13048",
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
  prevButton.className = `px-2 py-1 text-gray-600 ${
    currentPage === 1 ? "opacity-50 cursor-not-allowed" : ""
  }`;
  prevButton.innerHTML = "&lt;";
  prevButton.onclick = () => {
    if (currentPage > 1) goToPage(currentPage - 1);
  };
  pagination.appendChild(prevButton);

  for (let i = 1; i <= totalPages; i++) {
    const pageButton = document.createElement("button");
    pageButton.className = `px-3 py-1 rounded border ${
      i === currentPage ? "bg-blue-500 text-white" : "bg-gray-200 text-gray-700"
    }`;
    pageButton.innerText = i;
    pageButton.onclick = () => goToPage(i);
    pagination.appendChild(pageButton);
  }

  const nextButton = document.createElement("button");
  nextButton.className = `px-2 py-1 text-gray-600 ${
    currentPage === totalPages ? "opacity-50 cursor-not-allowed" : ""
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
