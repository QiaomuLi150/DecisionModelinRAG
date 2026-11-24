# E5-QA Answer Casebook

This is a deterministic diagnostic casebook, not a factual adjudication. Labels are descriptive and do not assert correctness.

Selected unique qids: **100**; selection seed: **731**.

## qasper:2003.07433:1870f871a5bcea418c44f81f352897a2f53d0971
- Diagnostic label: `UNRESOLVED`
- Paper: `2003.07433` — LAXARY: A Trustworthy Explainable Twitter Analysis Model for Post-Traumatic Stress Disorder Assessment
- Question: Which clinically validated survey tools are used?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["DOSPERT, BSSS and VIAS"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:2003.07433:p3_0", "qasper:1809.05752:p8_1", "qasper:2003.07433:p14_0", "qasper:2003.07433:p0_2", "qasper:2003.07433:p2_1"]
- Jina selected IDs: ["qasper:2003.07433:p3_0", "qasper:2003.07433:p2_1", "qasper:1809.05752:p8_1", "qasper:2003.07433:p5_0", "qasper:2003.07433:p0_2"]
- pplx_decider reader answer: INSUFFICIENT_EVIDENCE
- pplx_decider frozen input hash: `bed17fa8ed7713ddaea4089ca7c445ab68015d7926814ffefa6cfa05d2029c7e`
- jina35 reader answer: Domain-Specific Risk-Taking (DOSPERT) Scale, Berlin Social Support Scales (BSSS), Values In Action Scale (VIAS), NEO Five-Factor Inventory, Positive and Negative Syndrome Scale, Montgomery-Asperg Depression Rating Scale, Young Mania Rating Scale
- jina35 frozen input hash: `e720e1d9ee07c7b45b03e33292595b6c30e01c86c9cc20b5c970f7a72ef8cb85`

No case is classified as factually correct or incorrect by this audit.

## qasper:1912.08960:509af1f11bd6f3db59284258e18fdfebe86cae47
- Diagnostic label: `UNRESOLVED`
- Paper: `1912.08960` — Going Beneath the Surface: Evaluating Image Captioning for Grammaticality, Truthfulness and Diversity
- Question: How is diversity measured?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["diversity score as the ratio of observed number versus optimal number", "we look at language constructions used and compute the corresponding diversity score as the ratio of observed number versus optimal number"]
- Evidence annotations: 2; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1912.08960:p6_1", "qasper:2003.08529:p3_5", "qasper:1911.03350:p7_0", "qasper:2001.05467:p0_2", "qasper:1909.00015:p14_0"]
- Jina selected IDs: ["qasper:2003.08529:p3_5", "qasper:1911.03350:p7_0", "qasper:2001.05467:p13_0", "qasper:2003.08529:p0_3", "qasper:2003.08529:p8_1"]
- pplx_decider reader answer: Diversity is measured by computing the ratio of observed to optimal number of language constructions, standard deviation of embedding vectors with geometric mean, Self-BLEU scores, or Average Output Probability Distribution.
- pplx_decider frozen input hash: `7398dd434bb324f1469a520bf6ae471f3e908ef2c2c4a71ec152182557a489bf`
- jina35 reader answer: By calculating the standard deviation of embedding vectors along each dimension and taking the geometric mean of all calculated values.
- jina35 frozen input hash: `cecc25cbaa683f0ec337eb7ad5a08b04b3204a9272dd833e4e6bd44418aa4211`

No case is classified as factually correct or incorrect by this audit.

## qasper:1909.00338:827c58f6cab6c6fe7a6c43bdc71150b61ba0eed4
- Diagnostic label: `UNRESOLVED`
- Paper: `1909.00338` — Monitoring stance towards vaccination in twitter messages
- Question: What is the agreement score of their annotated dataset?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["Relevance and Subject categorizations are annotated at a percent agreement of $0.71$ and $0.70$, their agreement scores are only fair, at $\\alpha =0.27$ and $\\alpha =0.29$", "Stance and Sentiment, which carry more categories than the former two, is $0.54$ for both. Their agreement scores are also fair, at $\\alpha =0.35$ and $\\alpha =0.34$", "The Negative category yields a mutual F-score of $0.42$, which is higher than the more frequently annotated categories Neutral ($0.23$) and Not clear ($0.31$).", "This holds for the Relevant category ($0.81$), the Vaccine category ($0.79$) and the Positive category ($0.64$)"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:2003.04642:p10_1", "qasper:1608.08738:p5_1", "qasper:2002.02427:p1_1", "qasper:1808.06834:p4_2", "qasper:1612.05310:p4_3"]
- Jina selected IDs: ["qasper:2004.03034:p2_7", "qasper:1909.00338:p3_7", "qasper:1708.09609:p2_12", "qasper:2003.04642:p10_1", "qasper:1911.03243:p9_0"]
- pplx_decider reader answer: 0.82
- pplx_decider frozen input hash: `e9db3960b289f140100f4fcb84d811edf709f248c4889f25024411bd02937950`
- jina35 reader answer: INSUFFICIENT_EVIDENCE
- jina35 frozen input hash: `245116ba3d7d3974bd292ea0eb60d7f02874d7bdfce307ef64b1ae875e95b9c3`

No case is classified as factually correct or incorrect by this audit.

## qasper:1910.09982:fd753ab5177d7bd27db0e0afc12411876ee607df
- Diagnostic label: `UNRESOLVED`
- Paper: `1910.09982` — Findings of the NLP4IF-2019 Shared Task on Fine-Grained Propaganda Detection
- Question: What was the baseline for this task?
- Reference status/types: `ANSWERABLE` / `MIXED_ANNOTATED`
- Reference forms: ["FLC task generates spans and selects one of the 18 techniques randomly", "SLC task is a very simple logistic regression classifier", "The baseline system for the SLC task is a very simple logistic regression classifier with default parameters. The baseline for the FLC task generates spans and selects one of the 18 techniques randomly."]
- Evidence annotations: 2; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1703.10344:p11_1", "qasper:1911.00841:p12_0", "qasper:1703.10344:p5_19", "qasper:1910.09982:p28_0", "qasper:1707.03764:p6_2"]
- Jina selected IDs: ["qasper:1910.09982:p28_0", "qasper:2003.10816:p11_0", "qasper:1910.09982:p28_1", "qasper:1911.00841:p12_0", "qasper:1707.03764:p6_2"]
- pplx_decider reader answer: INSUFFICIENT_EVIDENCE
- pplx_decider frozen input hash: `e236404623113a1edcd0c9e16443646a5332e186dec5db58e31835e91e771de6`
- jina35 reader answer: a very simple logistic regression classifier with default parameters, where we represent the input instances with a single feature: the length of the sentence
- jina35 frozen input hash: `32abf5b0aad152cdd97e614a388013171e242322793db38ba09faacad446e6ca`

No case is classified as factually correct or incorrect by this audit.

## qasper:1906.06045:58a00ca123d67b9be55021493384c0acef4c568d
- Diagnostic label: `UNRESOLVED`
- Paper: `1906.06045` — Learning to Ask Unanswerable Questions for Machine Reading Comprehension
- Question: How do they ensure the generated questions are unanswerable?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["learn to ask unanswerable questions by editing answerable ones with word exchanges, negations, etc"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1906.06045:p0_4", "qasper:1906.06045:p2_0", "qasper:1906.06045:p0_1", "qasper:1906.06045:p0_2", "qasper:1906.06045:p7_4"]
- Jina selected IDs: ["qasper:1906.06045:p0_1", "qasper:1906.06045:p7_8", "qasper:1906.06045:p7_4", "qasper:1906.06045:p2_1", "qasper:1906.06045:p2_0"]
- pplx_decider reader answer: They ensure unanswerability by generating questions that cannot be answered by the given paragraph, while maintaining relevance to the original question and paragraph, and asking for the same type of answer as the original.
- pplx_decider frozen input hash: `75ab077844bcab7184d6ef585096536b099894fea7874ab40131fd43db3f7bce`
- jina35 reader answer: They ensure the generated questions are unanswerable by editing answerable questions and conditioning on the corresponding paragraph that contains the answer.
- jina35 frozen input hash: `ff7547097f33fe1c0f51576b30b14f0a9b7de99f9879956e3373f17f5b9edfbc`

No case is classified as factually correct or incorrect by this audit.

## qasper:1911.11698:26012f57cba21ba44b9a9f7ed8b1ed9e8ee7625d
- Diagnostic label: `UNRESOLVED`
- Paper: `1911.11698` — Doc2Vec on the PubMed corpus: study of a new approach to generate related articles
- Question: What Doc2Vec architectures other than PV-DBOW have been tried?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["PV-DM"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1901.01010:p9_3", "qasper:1911.11698:p6_0", "qasper:1911.11698:p19_0", "qasper:1706.09673:p2_1", "qasper:1911.11698:p3_0"]
- Jina selected IDs: ["qasper:1911.11698:p0_2", "qasper:1909.00183:p4_2", "qasper:1911.11698:p3_0", "qasper:1911.11698:p15_0", "qasper:1903.12495:p2_2"]
- pplx_decider reader answer: PV-DM
- pplx_decider frozen input hash: `dc505f35dc353ee5965e2089dae8672d3e607024c4df5067b0c4c4dd0e196825`
- jina35 reader answer: PV-DM
- jina35 frozen input hash: `25241b8a2b342cdc17b896b5f234ede83bf094e5c8f46fee9c12d5b52fcfbf0d`

No case is classified as factually correct or incorrect by this audit.

## qasper:1908.05828:f59f1f5b528a2eec5cfb1e49c87699e0c536cc45
- Diagnostic label: `UNRESOLVED`
- Paper: `1908.05828` — Named Entity Recognition for Nepali Language
- Question: How big is the new Nepali NER dataset?
- Reference status/types: `ANSWERABLE` / `ABSTRACTIVE`
- Reference forms: ["3606 sentences", "Dataset contains 3606 total sentences and 79087 total entities."]
- Evidence annotations: 2; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1908.05828:p16_2", "qasper:1908.05828:p8_0", "qasper:1908.05828:p9_0", "qasper:1908.05828:p0_3", "qasper:1908.05828:p0_4"]
- Jina selected IDs: ["qasper:1908.05828:p8_0", "qasper:1908.05828:p16_2", "qasper:1908.05828:p0_3", "qasper:1908.05828:p0_1", "qasper:1908.05828:p0_4"]
- pplx_decider reader answer: INSUFFICIENT_EVIDENCE
- pplx_decider frozen input hash: `56fcca1077ffe8896917ca24ac810eaa0c80068ae831018585cd523c07d67c04`
- jina35 reader answer: INSUFFICIENT_EVIDENCE
- jina35 frozen input hash: `70b3873565cada18a9d327e31d621fc2ff686d7970da25b0eef3635b85c57395`

No case is classified as factually correct or incorrect by this audit.

## qasper:1704.06194:c0e341c4d2253eb42c8840381b082aae274eddad
- Diagnostic label: `UNRESOLVED`
- Paper: `1704.06194` — Improved Neural Relation Detection for Knowledge Base Question Answering
- Question: What is te core component for KBQA?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["answer questions by obtaining information from KB tuples", "hierarchical matching between questions and relations with residual learning"]
- Evidence annotations: 2; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1704.06194:p0_3", "qasper:1704.06194:p15_0", "qasper:1707.04662:p1_3", "qasper:1604.00727:p2_0", "qasper:1902.09087:p0_2"]
- Jina selected IDs: ["qasper:1704.06194:p14_2", "qasper:1704.06194:p0_3", "qasper:1704.06194:p0_0", "qasper:2003.08897:p4_0", "qasper:1704.06194:p15_0"]
- pplx_decider reader answer: relation detection
- pplx_decider frozen input hash: `d90a6ef70229f17d5c596613004bb7fc91fca8764f74ac857d5bbfd1c360cddd`
- jina35 reader answer: relation detection
- jina35 frozen input hash: `b7e56b609d3cf4d0874496a67b7ab9a55df7fe54c48db893223061c53e2dd0ae`

No case is classified as factually correct or incorrect by this audit.

## qasper:1910.00825:d5f8707ddc21741d52b3c2a9ab1af2871dc6c90b
- Diagnostic label: `UNRESOLVED`
- Paper: `1910.00825` — Abstractive Dialog Summarization with Semantic Scaffolds
- Question: What automatic and human evaluation metrics are used to compare SPNet to its counterparts?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["ROUGE and CIC", "relevance, conciseness and readability on a 1 to 5 scale, and rank the summary pair"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1910.00825:p14_0", "qasper:1910.00825:p11_1", "qasper:1607.06025:p1_4", "qasper:1910.00825:p12_0", "qasper:1910.00825:p12_1"]
- Jina selected IDs: ["qasper:1910.00825:p14_0", "qasper:1910.00825:p11_1", "qasper:1910.00825:p0_2", "qasper:1811.05085:p15_0", "qasper:1910.00825:p12_1"]
- pplx_decider reader answer: ROUGE, CIC, relevance, conciseness, readability
- pplx_decider frozen input hash: `2b5ed697d53f0cb4f766fbc86ea423817bc541c50ee342be8016d66731f0a906`
- jina35 reader answer: ROUGE, CIC, relevance, readability, conciseness
- jina35 frozen input hash: `d28bac4f4df339dc3484c01c6dd33e363cdca43fd7459daec2048962f9b6f4e1`

No case is classified as factually correct or incorrect by this audit.

## qasper:2001.05970:096f5c59f43f49cab1ef37126341c78f272c0e26
- Diagnostic label: `UNRESOLVED`
- Paper: `2001.05970` — #MeToo on Campus: Studying College Sexual Assault at Scale Using Data Reported on Social Media
- Question: How many followers did they analyze?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["51,104"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1909.10012:p2_1", "qasper:1909.10012:p3_2", "qasper:1909.10012:p6_0", "qasper:2001.05970:p2_0", "qasper:1909.10012:p17_1"]
- Jina selected IDs: ["qasper:1909.10012:p3_2", "qasper:1909.10012:p2_1", "qasper:2001.08845:p16_0", "qasper:1908.00153:p10_4", "qasper:1907.01468:p12_0"]
- pplx_decider reader answer: 55,830,844
- pplx_decider frozen input hash: `3fd8b1c1b76a0c3a277c0810b7bac0d8ee55a4a41e86784307bcc75b8546e2b9`
- jina35 reader answer: 55 million
- jina35 frozen input hash: `6f87729ea4ab3d3cdb45fe9f510905efd464a862258cda279c91f93d2ef6174e`

No case is classified as factually correct or incorrect by this audit.

## qasper:1602.01208:d557752c4706b65dcdb7718272180c59d77fb7a7
- Diagnostic label: `UNRESOLVED`
- Paper: `1602.01208` — Spatial Concept Acquisition for a Mobile Robot that Integrates Self-Localization and Unsupervised Word Discovery from Spoken Sentences
- Question: Which method do they use for word segmentation?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["unsupervised word segmentation method latticelm"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:2001.01589:p0_2", "qasper:2001.01589:p16_1", "qasper:1908.08419:p0_3", "qasper:1711.04457:p5_3", "qasper:1602.01208:p5_9"]
- Jina selected IDs: ["qasper:2001.01589:p1_0", "qasper:1906.07662:p6_0", "qasper:1911.03270:p12_0", "qasper:1708.00214:p5_0", "qasper:1906.07662:p8_0"]
- pplx_decider reader answer: morphological word segmentation
- pplx_decider frozen input hash: `609405717c82a3aa107544b84e53708e16928695856794aa51fcc51a7f84bab4`
- jina35 reader answer: Maximum Matching
- jina35 frozen input hash: `8c781388a6288be8a76cf42570ad8221398e127499290ee1fb6a8a756fe3c879`

No case is classified as factually correct or incorrect by this audit.

## qasper:2003.03044:ce0e2a8675055a5468c4c54dbb099cfd743df8a7
- Diagnostic label: `UNRESOLVED`
- Paper: `2003.03044` — A Corpus for Detecting High-Context Medical Conditions in Intensive Care Patient Notes Focusing on Frequently Readmitted Patients
- Question: What are 10 other phenotypes that are annotated?
- Reference status/types: `ANSWERABLE` / `ABSTRACTIVE`
- Reference forms: ["Adv. Heart Disease, Adv. Lung Disease, Alcohol Abuse, Chronic Neurologic Dystrophies, Dementia, Depression, Developmental Delay, Obesity, Psychiatric disorders and Substance Abuse"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:2003.03044:p2_1", "qasper:2003.03044:p1_1", "qasper:2003.03044:p1_3", "qasper:2003.03044:p9_0", "qasper:2003.03044:p9_1"]
- Jina selected IDs: ["qasper:2003.03044:p2_1", "qasper:2003.03044:p9_0", "qasper:2003.03044:p1_1", "qasper:1906.03538:p2_11", "qasper:1912.09152:p12_1"]
- pplx_decider reader answer: INSUFFICIENT_EVIDENCE
- pplx_decider frozen input hash: `07b0673342799350d204fc956f1c3205dbb93820abf454f5322a46f8a525f9b5`
- jina35 reader answer: INSUFFICIENT_EVIDENCE
- jina35 frozen input hash: `c0af213314698a9bf91ff52bae8c282865b0ef22a77323c1232540ff127b204d`

No case is classified as factually correct or incorrect by this audit.

## qasper:1710.06536:3bf429633ecbbfec3d7ffbcfa61fa90440cc918b
- Diagnostic label: `UNRESOLVED`
- Paper: `1710.06536` — Basic tasks of sentiment analysis
- Question: How are aspects identified in aspect extraction?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["apply an ensemble of deep learning and linguistics t"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1912.07976:p2_0", "qasper:1710.06536:p14_43", "qasper:1710.06536:p6_4", "qasper:1710.06536:p8_0", "qasper:1710.06536:p8_1"]
- Jina selected IDs: ["qasper:1710.06536:p6_3", "qasper:1912.07976:p2_0", "qasper:1710.06536:p14_43", "qasper:1710.06536:p8_0", "qasper:1909.00430:p8_6"]
- pplx_decider reader answer: Using dictionary-based, rule-based, machine-learning, or deep learning methodologies, often incorporating syntactic rules, word embeddings, POS features, or CRFs.
- pplx_decider frozen input hash: `ee629d703b7eb3d5efcd86b65753ca3ccbe891c05237eaea0a8451ae5f707e4f`
- jina35 reader answer: Aspects are identified as explicit aspects (explicitly denoted words) or implicit aspects (inferred from context), using rule-based methods, machine learning, or deep learning approaches, often leveraging syntactic dependency rules, word embeddings, and POS features.
- jina35 frozen input hash: `6c8c9affe698faff80f35eb24a518d759b4dfc366d116f62ff9c305d444820fe`

No case is classified as factually correct or incorrect by this audit.

## qasper:1610.07809:e54257585cc75564341eb02bdc63ff8111992f82
- Diagnostic label: `UNRESOLVED`
- Paper: `1610.07809` — How Document Pre-processing affects Keyphrase Extraction Performance
- Question: what keyphrase extraction models were reassessed?
- Reference status/types: `ANSWERABLE` / `ABSTRACTIVE`
- Reference forms: ["Answer with content missing: (LVL1, LVL2, LVL3) \n- Stanford CoreNLP\n- Optical Character Recognition (OCR) system, ParsCIT \n- further abridge the input text from level 2 preprocessed documents to the following: title, headers, abstract, introduction, related work, background and conclusion."]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1610.07809:p2_0", "qasper:1610.07809:p7_0", "qasper:1610.07809:p0_1", "qasper:1610.07809:p0_2", "qasper:1810.05241:p1_0"]
- Jina selected IDs: ["qasper:1610.07809:p7_0", "qasper:1610.07809:p2_0", "qasper:1610.07809:p0_2", "qasper:1610.07809:p1_2", "qasper:1610.07809:p0_1"]
- pplx_decider reader answer: five keyphrase extraction models
- pplx_decider frozen input hash: `014c4cdae32be733f3f4a4ae4943508432dcf652e71d39a3946c0168f4182cb1`
- jina35 reader answer: several keyphrase extraction models
- jina35 frozen input hash: `83c9d9bf43c15ece5caf5c3266e2695c43254af48bf2c8ccbf19ea6c4e17a709`

No case is classified as factually correct or incorrect by this audit.

## qasper:1912.00667:a74190189a6ced2a2d5b781e445e36f4e527e82a
- Diagnostic label: `UNRESOLVED`
- Paper: `1912.00667` — A Human-AI Loop Approach for Joint Keyword Discovery and Expectation Estimation in Micropost Event Detection
- Question: How are the accuracy merits of the approach demonstrated?
- Reference status/types: `ANSWERABLE` / `MIXED_ANNOTATED`
- Reference forms: ["By evaluating the performance of the approach using accuracy and AUC", "significant improvements clearly demonstrate that our approach is effective at improving model performance"]
- Evidence annotations: 2; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1909.09484:p16_0", "qasper:1910.00194:p10_0", "qasper:1607.06025:p0_8", "qasper:1808.05077:p2_4", "qasper:1904.06941:p4_14"]
- Jina selected IDs: ["qasper:1804.04225:p6_2", "qasper:1912.00667:p7_3", "qasper:1912.00871:p11_4", "qasper:1808.05077:p1_3", "qasper:1808.05077:p2_4"]
- pplx_decider reader answer: By comparing the accuracy of classifiers trained on the generated dataset to those trained on the original dataset.
- pplx_decider frozen input hash: `c43850cf51435a633a437540b4070148eadd8deb3cb530be0554cca1ca5dfe3f`
- jina35 reader answer: By comparing absolute accuracies and showing correlation with BLEU scores.
- jina35 frozen input hash: `bf1084511ffb1b155d965d5703d8ddf52354340fd5b17373510b976fe10028d7`

No case is classified as factually correct or incorrect by this audit.

## qasper:1911.07555:c8541ff10c4e0c8e9eb37d9d7ea408d1914019a9
- Diagnostic label: `UNRESOLVED`
- Paper: `1911.07555` — Short Text Language Identification for Under Resourced Languages
- Question: Which datasets are employed for South African languages LID?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["DSL 2015", "DSL 2017", "JW300 parallel corpus", "NCHLT text corpora"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1911.07555:p1_3", "qasper:1911.07555:p0_4", "qasper:1911.07555:p1_1", "qasper:1911.07555:p1_0", "qasper:1911.07555:p1_2"]
- Jina selected IDs: ["qasper:1911.07555:p1_4", "qasper:1911.07555:p1_3", "qasper:1911.07555:p1_0", "qasper:1911.07555:p1_1", "qasper:1911.07555:p0_4"]
- pplx_decider reader answer: NCHLT text corpora, DSL 2015, DSL 2017
- pplx_decider frozen input hash: `7289450f28edf7af62c9815dd07e6517f467e71b07e046bbeee2d2579e726bf3`
- jina35 reader answer: NCHLT text corpora, DSL 2015, DSL 2017
- jina35 frozen input hash: `d1b5ae5e383719a9d1ac339aa9b0c20f0823fff64dad04f24fd5886c823ac0d0`

No case is classified as factually correct or incorrect by this audit.

## qasper:1909.00100:e24fbcc8be922c43f6b6037cdf2bfd4c0a926c08
- Diagnostic label: `UNRESOLVED`
- Paper: `1909.00100` — Small and Practical BERT Models for Sequence Labeling
- Question: What is the multilingual baseline?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["the Meta-LSTM BIBREF0"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1612.05270:p1_0", "qasper:1909.09524:p6_14", "qasper:1710.09589:p1_0", "qasper:1911.03894:p3_0", "qasper:2003.10816:p11_0"]
- Jina selected IDs: ["qasper:1612.05270:p1_0", "qasper:1710.09589:p1_0", "qasper:1710.09589:p11_2", "qasper:1909.00100:p0_0", "qasper:1909.09524:p6_19"]
- pplx_decider reader answer: B4MSA
- pplx_decider frozen input hash: `ed14596e0c44e538e7f29d8e62e82f3e2ac03a80551caf9237c6514adae46c69`
- jina35 reader answer: A simple, open-source, multilingual sentiment analysis model designed for ease of implementation and proven effectiveness in international competitions.
- jina35 frozen input hash: `bdd84c40422a8269511925f029fbc90e2b62f5516490d0a1ae586d3ea8d6e457`

No case is classified as factually correct or incorrect by this audit.

## qasper:1809.09795:4ec538e114356f72ef82f001549accefaf85e99c
- Diagnostic label: `UNRESOLVED`
- Paper: `1809.09795` — Deep contextualized word representations for detecting sarcasm and irony
- Question: Which morphosyntactic features are thought to indicate irony or sarcasm?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["all caps", "emojis", "emoticons", "hashtags", "quotation marks"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1709.05404:p3_4", "qasper:1709.05404:p0_0", "qasper:1809.09795:p0_0", "qasper:1809.09795:p2_0", "qasper:1709.05404:p1_0"]
- Jina selected IDs: ["qasper:1709.05404:p0_0", "qasper:1709.05404:p3_4", "qasper:1809.09795:p2_1", "qasper:1904.04019:p0_1", "qasper:1809.09795:p2_0"]
- pplx_decider reader answer: rhetorical questions and hyperbole
- pplx_decider frozen input hash: `53ea45828236f47afc1f43a1f59b94954f51bbed60de0d1630e869939e702087`
- jina35 reader answer: intensifiers, elongated words, quotations, false politeness, negative evaluations, emoticons, tag questions, all caps, hashtag
- jina35 frozen input hash: `d4ea61442c45a473e32225b66c39985edc0db0a55ae2869935c6964d3fc645ca`

No case is classified as factually correct or incorrect by this audit.

## qasper:2001.07263:94bee0c58976b58b4fef9e0adf6856fe917232e5
- Diagnostic label: `UNRESOLVED`
- Paper: `2001.07263` — Single headed attention based sequence-to-sequence model for state-of-the-art results on Switchboard-300
- Question: How much bigger is Switchboard-2000 than Switchboard-300 database?
- Reference status/types: `ANSWERABLE` / `ABSTRACTIVE`
- Reference forms: ["Switchboard-2000 contains 1700 more hours of speech data."]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:2001.07263:p8_0", "qasper:2001.07263:p2_0", "qasper:1705.03151:p20_2", "qasper:1904.03288:p8_0", "qasper:1807.00868:p2_0"]
- Jina selected IDs: ["qasper:2001.07263:p8_0", "qasper:2001.07263:p9_0", "qasper:1709.09119:p30_1", "qasper:1912.03627:p3_0", "qasper:2001.11899:p32_0"]
- pplx_decider reader answer: 700 hours
- pplx_decider frozen input hash: `f4e6b3af95d46e96ec72ee58c283c6722d8f8649768959a7155d3f94df27530e`
- jina35 reader answer: 2000-hour
- jina35 frozen input hash: `4f13ea88e74d0363564fcb0c1a497ebf4477049d69cfb947d5166efd378acf9f`

No case is classified as factually correct or incorrect by this audit.

## qasper:1908.11546:b8dea4a98b4da4ef1b9c98a211210e31d6630cf3
- Diagnostic label: `UNRESOLVED`
- Paper: `1908.11546` — Modeling Multi-Action Policy for Task-Oriented Dialogues
- Question: What is specific to gCAS cell?
- Reference status/types: `ANSWERABLE` / `ABSTRACTIVE`
- Reference forms: ["It has three sequentially connected units to output continue, act and slots generating multi-acts in a doble recurrent manner."]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1908.11546:p2_0", "qasper:1908.11546:p5_5", "qasper:1908.11546:p7_0", "qasper:1908.11546:p6_2", "qasper:1908.11546:p6_1"]
- Jina selected IDs: ["qasper:1908.11546:p2_0", "qasper:1908.11546:p5_5", "qasper:1908.11546:p0_2", "qasper:1908.11546:p7_0", "qasper:1908.11546:p6_0"]
- pplx_decider reader answer: It contains three sequentially connected units for outputting continue, act, and slots respectively.
- pplx_decider frozen input hash: `774e769dc5a7debf4f804f50618a8f82daa757bc0a915455e2b526e3c11263de`
- jina35 reader answer: It contains three sequentially connected units for outputting continue, act, and slots respectively.
- jina35 frozen input hash: `eeea1034a2aa839e0d12f94b791c234ff42162cf1034c3110107cdcfc8f3f989`

No case is classified as factually correct or incorrect by this audit.

## qasper:1906.06045:199bdb3a6b1f7c89d95ea6c6ddbbb5eff484fa1f
- Diagnostic label: `UNRESOLVED`
- Paper: `1906.06045` — Learning to Ask Unanswerable Questions for Machine Reading Comprehension
- Question: Does their approach require a dataset of unanswerable questions mapped to similar answerable questions?
- Reference status/types: `ANSWERABLE` / `YES_NO`
- Reference forms: ["yes"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1906.06045:p0_2", "qasper:1906.06045:p7_0", "qasper:1906.06045:p6_0", "qasper:1906.06045:p0_1", "qasper:1906.06045:p9_0"]
- Jina selected IDs: ["qasper:1906.06045:p0_2", "qasper:1906.06045:p7_0", "qasper:1906.06045:p7_1", "qasper:1911.00841:p15_7", "qasper:1906.06045:p6_0"]
- pplx_decider reader answer: No
- pplx_decider frozen input hash: `0aeb46203b51852d2dfebfcb6d53c884ed45ae6f8cd08ca0410df3af3bee0d51`
- jina35 reader answer: No
- jina35 frozen input hash: `db28ebb39eac171175dfdcaf5023b579623d36370489ca34d46df5a1a843c603`

No case is classified as factually correct or incorrect by this audit.

## qasper:1909.09067:9eabb54c2408dac24f00f92cf1061258c7ea2e1a
- Diagnostic label: `UNRESOLVED`
- Paper: `1909.09067` — A Corpus for Automatic Readability Assessment and Text Simplification of German
- Question: Which information about text structure is included in the corpus?
- Reference status/types: `ANSWERABLE` / `MIXED_ANNOTATED`
- Reference forms: ["Information on physical page segmentation (for PDFs only), paragraph segmentation, and line segmentation", "lines", "paragraph, lines, textspan element (paragraph segmentation, line segmentation, Information on physical page segmentation(for PDF only))", "paragraphs"]
- Evidence annotations: 2; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1909.09067:p5_3", "qasper:1909.09067:p0_4", "qasper:1909.09067:p2_3", "qasper:1904.06941:p11_2", "qasper:2003.14026:p0_9"]
- Jina selected IDs: ["qasper:1909.09067:p2_3", "qasper:1909.09067:p0_4", "qasper:1909.09067:p8_0", "qasper:1909.09067:p8_1", "qasper:1909.09067:p5_1"]
- pplx_decider reader answer: Text structure includes paragraph segmentation, line segmentation, and information on physical page segmentation for PDFs.
- pplx_decider frozen input hash: `c3778e9c1ea8647c11616dd8a9fb7e3c1409959c7eaa217fece421cde523ba6c`
- jina35 reader answer: text structure
- jina35 frozen input hash: `4613a8daef3a0153d5ebb384ea2a69109e1d4274fc6e3916f46ce84104a93643`

No case is classified as factually correct or incorrect by this audit.

## qasper:1905.12801:d4e5e3f37679ff68914b55334e822ea18e60a6cf
- Diagnostic label: `UNRESOLVED`
- Paper: `1905.12801` — Reducing Gender Bias in Word-Level Language Models with a Gender-Equalizing Loss Function
- Question: what kinds of male and female words are looked at?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["gendered word pairs like he and she"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1910.14497:p11_1", "qasper:1707.03764:p4_9", "qasper:1910.14497:p4_0", "qasper:1707.03764:p4_1", "qasper:1810.04528:p4_0"]
- Jina selected IDs: ["qasper:1910.14497:p11_1", "qasper:1910.14497:p4_0", "qasper:1905.12801:p0_2", "qasper:1707.03764:p4_9", "qasper:1905.12801:p5_5"]
- pplx_decider reader answer: male socially-biased words and female socially-biased words
- pplx_decider frozen input hash: `a8a410dd8bf3fec88c6c1dd2145f6d217d034e71d4e60f3538522e2208629750`
- jina35 reader answer: male socially-biased words and female socially-biased words
- jina35 frozen input hash: `a1bea241e7e39c2833ca4a1faaa0ff742485e460eb5cb0fd550365aba9afe8af`

No case is classified as factually correct or incorrect by this audit.

## qasper:2003.03612:aaf50a6a9f449389ef212d25d0fae59c10b0df92
- Diagnostic label: `UNRESOLVED`
- Paper: `2003.03612` — Frozen Binomials on the Web: Word Ordering and Language Conventions in Online Text
- Question: What types of various community texts have been investigated for exploring global structure of binomials?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["news publications, wine reviews, and Reddit"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:2003.03612:p0_12", "qasper:2003.03612:p16_0", "qasper:2003.03612:p0_8", "qasper:2003.03612:p0_7", "qasper:2003.03612:p0_5"]
- Jina selected IDs: ["qasper:2003.03612:p0_12", "qasper:2003.03612:p0_4", "qasper:2003.03612:p18_1", "qasper:2003.03612:p1_0", "qasper:2003.03612:p0_7"]
- pplx_decider reader answer: news publications, wine reviews, and Reddit
- pplx_decider frozen input hash: `3e590657312d516a67a5847dc0a72d8bbd81c5a27de5df11aaa99031ed4493fd`
- jina35 reader answer: sports community, politics community, online text data communities, subreddits
- jina35 frozen input hash: `adb3937696c42cf51034e650d373f55efaf9792e801b360867d15dcf7ed7f68d`

No case is classified as factually correct or incorrect by this audit.

## qasper:1912.00667:43f074bacabd0a355b4e0f91a1afd538c0a6244f
- Diagnostic label: `UNRESOLVED`
- Paper: `1912.00667` — A Human-AI Loop Approach for Joint Keyword Discovery and Expectation Estimation in Micropost Event Detection
- Question: How is the keyword specific expectation elicited from the crowd?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["asked to find the keyword that best indicates the class of the microposts", "workers are first asked to find those microposts where the model predictions are deemed correct"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1912.00667:p4_0", "qasper:1912.00667:p1_8", "qasper:1912.00667:p1_14", "qasper:1912.00667:p0_3", "qasper:1912.00667:p2_1"]
- Jina selected IDs: ["qasper:1912.00667:p4_0", "qasper:1912.00667:p0_3", "qasper:1912.00667:p2_1", "qasper:1912.00667:p4_1", "qasper:1912.00667:p1_15"]
- pplx_decider reader answer: By leveraging crowd workers for estimating keyword-specific expectations.
- pplx_decider frozen input hash: `845975dbeaf187bfe22b6560b3d602c1ca5a181eeecea7e9d0515d5151261f0e`
- jina35 reader answer: By leveraging crowd workers to estimate keyword-specific expectations iteratively.
- jina35 frozen input hash: `54faf5dcc74397495f0e893303766338651eabb2c4664a2a4c4394d18f560d1d`

No case is classified as factually correct or incorrect by this audit.

## qasper:1709.05411:65e6a1cc2590b139729e7e44dce6d9af5dd2c3b5
- Diagnostic label: `UNRESOLVED`
- Paper: `1709.05411` — Combining Search with Structured Data to Create a More Engaging User Experience in Open Domain Dialogue
- Question: Why mixed initiative multi-turn dialogs are the greatest challenge in building open-domain conversational agents?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["at each system turn, there are a large number of conversational moves that are possible", "do not follow a particular plan or pursue a particular fixed information need", "integrating content found via search with content from structured data", "live search may not be able to achieve the required speed and efficiency", "most other domains do not have such high quality structured data available"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1709.05411:p0_0", "qasper:1709.05411:p0_8", "qasper:1709.05411:p2_0", "qasper:1709.05411:p1_5", "qasper:1709.05411:p0_3"]
- Jina selected IDs: ["qasper:1709.05411:p0_0", "qasper:1709.05411:p2_0", "qasper:1709.05411:p4_0", "qasper:1705.01214:p1_2", "qasper:1709.05411:p0_3"]
- pplx_decider reader answer: Because they do not follow a particular plan or pursue a fixed information need, requiring real-time balancing of dialogue acts and content availability.
- pplx_decider frozen input hash: `44327d743ef76b22bc73637e6d2d9993a1e75c33bee468484f5fadbd803c0bf4`
- jina35 reader answer: Because they do not follow a particular plan or pursue a fixed information need, making them difficult to model and maintain coherence in open-domain conversations.
- jina35 frozen input hash: `10802483eb10e551b791627ce9ec126d918827e2ee2dfc12604303e8af6ac806`

No case is classified as factually correct or incorrect by this audit.

## qasper:1910.11769:67d8e50ddcc870db71c94ad0ad7f8a59a6c67ca6
- Diagnostic label: `UNRESOLVED`
- Paper: `1910.11769` — DENS: A Dataset for Multi-class Emotion Analysis
- Question: How many annotators were there?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["3"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1912.01673:p7_0", "qasper:1909.00338:p3_6", "qasper:2003.04866:p8_0", "qasper:1910.11769:p5_2", "qasper:1912.08320:p35_43"]
- Jina selected IDs: ["qasper:1912.01673:p7_0", "qasper:1909.00338:p3_6", "qasper:1912.01673:p7_2", "qasper:1912.08320:p14_0", "qasper:1912.08320:p35_43"]
- pplx_decider reader answer: 65
- pplx_decider frozen input hash: `b95cabb8264da7ed3d9d8ea666d01a47c0e2605a14430c5cbcb51b38328b8fe9`
- jina35 reader answer: 65
- jina35 frozen input hash: `1d5998f4cc0933ec21885507a9cd9e27e271468dfe53f9344be23b193f145457`

No case is classified as factually correct or incorrect by this audit.

## qasper:2003.03612:574f17134e4dd041c357ebb75a7ef590da294d22
- Diagnostic label: `UNRESOLVED`
- Paper: `2003.03612` — Frozen Binomials on the Web: Word Ordering and Language Conventions in Online Text
- Question: What new model is proposed for binomial lists?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["null model"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:2003.03612:p3_0", "qasper:2003.03612:p0_8", "qasper:2003.03612:p0_11", "qasper:2003.03612:p6_0", "qasper:2003.03612:p0_6"]
- Jina selected IDs: ["qasper:2003.03612:p3_0", "qasper:2003.03612:p16_0", "qasper:2003.03612:p18_1", "qasper:2003.03612:p0_13", "qasper:2003.03612:p0_1"]
- pplx_decider reader answer: A new framework based on asymmetry, movement, and agreement visualized as a cube.
- pplx_decider frozen input hash: `a5a074644eb9b7e78a454903c0dcea0c7eb62b5dc8b0d5026bad5c1c222716b2`
- jina35 reader answer: a new framework based on asymmetry, movement, and agreement visualized as a cube
- jina35 frozen input hash: `09f30880929500e7b982d01397065d2acd223afbdc9bd8a685580a65a15fa222`

No case is classified as factually correct or incorrect by this audit.

## qasper:1805.10824:22ccee453e37536ddb0c1c1d17b0dbac04c6c607
- Diagnostic label: `UNRESOLVED`
- Paper: `1805.10824` — UG18 at SemEval-2018 Task 1: Generating Additional Training Data for Predicting Emotion Intensity in Spanish
- Question: What other languages did they translate the data from?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["English"]
- Evidence annotations: 2; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1709.05563:p2_1", "qasper:1910.05154:p1_0", "qasper:2003.10816:p14_1", "qasper:2001.11899:p11_4", "qasper:1910.05456:p9_2"]
- Jina selected IDs: ["qasper:1805.10824:p4_0", "qasper:2003.10816:p14_1", "qasper:1709.05563:p2_1", "qasper:1906.00378:p10_0", "qasper:1708.01464:p10_2"]
- pplx_decider reader answer: German, French, and Italian
- pplx_decider frozen input hash: `6ad8345026a010baa3c853af0539494d208cdd72421bf0487a3d813c618135d2`
- jina35 reader answer: English, Farsi, French, German, Italian, Japanese
- jina35 frozen input hash: `02d6a69cc0a56d351f1861384ff71ed95d20572340a065ff28a429713c34ef59`

No case is classified as factually correct or incorrect by this audit.

## qasper:1908.06941:d46c0ea1ba68c649cc64d2ebb6af20202a74a3c7
- Diagnostic label: `UNRESOLVED`
- Paper: `1908.06941` — Why So Down? The Role of Negative (and Positive) Pointwise Mutual Information in Distributional Semantics
- Question: What are the disadvantages to clipping negative PMI?
- Reference status/types: `AMBIGUOUS_REFERENCE` / `AMBIGUOUS_REFERENCE`
- Reference forms: ["It may lead to poor rare word representations and word analogies."]
- Evidence annotations: 2; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1908.06941:p0_1", "qasper:1908.06941:p4_5", "qasper:1908.06941:p4_3", "qasper:1704.02686:p3_3", "qasper:1908.06941:p2_7"]
- Jina selected IDs: ["qasper:1908.06941:p0_1", "qasper:1908.06941:p2_9", "qasper:1908.06941:p2_1", "qasper:1908.06941:p4_5", "qasper:1908.06941:p1_1"]
- pplx_decider reader answer: Loss of information about negative cooccurrence, which may be useful for rare words and analogies.
- pplx_decider frozen input hash: `ccb1b12add4b67fbaf61da8a79a8754d5f50fbdb527e98b2291a53d85889078d`
- jina35 reader answer: Loss of information from the negative PMI spectrum.
- jina35 frozen input hash: `6a8ec8f28c1cd46a9c7d34ddc6958ab75ba1442e2d9203d538d4da641e155243`

No case is classified as factually correct or incorrect by this audit.

## qasper:1908.09246:3d7a982c718ea6bc7e770d8c5da564fbb9d11951
- Diagnostic label: `UNRESOLVED`
- Paper: `1908.09246` — Open Event Extraction from Online Text using a Generative Adversarial Network
- Question: How does this model overcome the assumption that all words in a document are generated from a single event?
- Reference status/types: `ANSWERABLE` / `MIXED_ANNOTATED`
- Reference forms: ["by learning a projection function between the document-event distribution and four event related word distributions", "flexibility of neural networks, the generator is capable of learning complicated nonlinear distributions", "supervision signal provided by the discriminator will help generator to capture the event-related patterns"]
- Evidence annotations: 2; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1908.09246:p0_3", "qasper:1908.09246:p0_2", "qasper:1908.09246:p8_0", "qasper:1910.01863:p17_1", "qasper:1910.01863:p9_0"]
- Jina selected IDs: ["qasper:1908.09246:p0_2", "qasper:1908.09246:p11_4", "qasper:1908.09246:p8_0", "qasper:1908.09246:p6_1", "qasper:1910.01863:p15_2"]
- pplx_decider reader answer: By using a generator network to learn complex nonlinear distributions and a discriminator to provide supervision signals that help capture event-related patterns, the AEM model can handle multiple events in a document without assuming all words originate from a single event.
- pplx_decider frozen input hash: `b0096d0a16b47878f468dbbd956ba75a1e273db42f95f4cb9433968208c76e5a`
- jina35 reader answer: By modeling multiple events and generating event-related word distributions for each, the model avoids assuming all words in a document stem from a single event.
- jina35 frozen input hash: `2a6ccd1786d7f41d6a40e8e8c3873c74ca1d0db5fca6c4627351a313e6d565a0`

No case is classified as factually correct or incorrect by this audit.

## qasper:1808.08850:99c50d51a428db09edaca0d07f4dab0503af1b94
- Diagnostic label: `UNRESOLVED`
- Paper: `1808.08850` — WiSeBE: Window-based Sentence Boundary Evaluation
- Question: What kind of Youtube video transcripts did they use?
- Reference status/types: `ANSWERABLE` / `ABSTRACTIVE`
- Reference forms: ["youtube video transcripts on news covering different topics like technology, human rights, terrorism and politics"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:2002.11268:p7_1", "qasper:1802.00923:p6_2", "qasper:1808.08850:p8_0", "qasper:1912.01046:p10_2", "qasper:2002.11268:p8_1"]
- Jina selected IDs: ["qasper:1912.01046:p1_2", "qasper:1912.01046:p1_0", "qasper:2002.11268:p7_1", "qasper:2002.11268:p7_2", "qasper:1912.04961:p13_1"]
- pplx_decider reader answer: transcripts obtained from semi-supervised caption filtering
- pplx_decider frozen input hash: `22402f298e82dca6d69830346bf43697ded8ddba02d19bfe1a9a7b238beb660a`
- jina35 reader answer: YouTube video transcripts from 120M segmented utterances with semi-supervised caption filtering.
- jina35 frozen input hash: `744ffea644ef93c0ad6e418df1f17aec691460f72a245734a6f14a78e0baa845`

No case is classified as factually correct or incorrect by this audit.

## qasper:1910.07181:d6e2b276390bdc957dfa7e878de80cee1f41fbca
- Diagnostic label: `UNRESOLVED`
- Paper: `1910.07181` — BERTRAM: Improved Word Embeddings Have Big Impact on Contextualized Model Performance
- Question: What models other than standalone BERT is new model compared to?
- Reference status/types: `ANSWERABLE` / `ABSTRACTIVE`
- Reference forms: ["Only Bert base and Bert large are compared to proposed approach."]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:2001.06286:p10_0", "qasper:2001.06286:p7_2", "qasper:2001.00137:p4_0", "qasper:2002.00652:p19_0", "qasper:1902.09314:p9_19"]
- Jina selected IDs: ["qasper:2001.00137:p4_0", "qasper:2002.00652:p19_0", "qasper:2001.06286:p0_0", "qasper:1910.07181:p10_2", "qasper:1908.10084:p1_1"]
- pplx_decider reader answer: RNN-based approach
- pplx_decider frozen input hash: `8d103ed861a64ecb6cd8fd77686215bdebba78ec4a0da0eff603b9764ab27936`
- jina35 reader answer: Concat, Attentive Mimicking, Bertram, RoBERTa, XLNet
- jina35 frozen input hash: `fa0d3f64082642dde619b49b273cc21ca2d443eb2d187d7a8685e12040b9de27`

No case is classified as factually correct or incorrect by this audit.

## qasper:1911.03597:b44ce9aae8b1479820555b99ce234443168dc1fe
- Diagnostic label: `UNRESOLVED`
- Paper: `1911.03597` — Zero-Shot Paraphrase Generation with Multilingual Language Models
- Question: What multilingual parallel data is used for training proposed model?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["MultiUN BIBREF20", "OpenSubtitles BIBREF21"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:2004.04721:p0_0", "qasper:1809.04686:p4_0", "qasper:1911.03597:p6_0", "qasper:1611.04798:p6_0", "qasper:1612.06897:p1_3"]
- Jina selected IDs: ["qasper:1809.04686:p4_0", "qasper:1907.02636:p13_0", "qasper:1809.04686:p4_1", "qasper:1611.04798:p6_0", "qasper:1910.11768:p0_6"]
- pplx_decider reader answer: MultiUN and OpenSubtitles
- pplx_decider frozen input hash: `e448c4700eabf219d0695f04cf2f8c302a1cd1a890ff3e81ebc86bbd0be58c8d`
- jina35 reader answer: WMT 2014 En ↔ Fr parallel corpus and larger corpora including EPPS, NC, and CommonCrawl
- jina35 frozen input hash: `f2516a2c725a41a7c3fe476cc52c1048f332c31c9afebda3294cdd05d5ac1f4c`

No case is classified as factually correct or incorrect by this audit.

## qasper:1910.08502:871c34219eb623bde9ac3937aa0f28fc3ad69445
- Diagnostic label: `UNRESOLVED`
- Paper: `1910.08502` — End-to-End Speech Recognition: A review for the French Language
- Question: Which model have the smallest Character Error Rate and which have the smallest Word Error Rate?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["For subword units, classic RNN-transducer, RNN-transducer with attention and joint CTC-attention show comparable performance", "character unit the RNN-transducer with additional attention module"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1910.08502:p11_0", "qasper:1910.08502:p14_0", "qasper:1910.08502:p13_9", "qasper:1910.08502:p13_22", "qasper:1708.01464:p8_3"]
- Jina selected IDs: ["qasper:2002.01320:p9_0", "qasper:1810.03459:p2_2", "qasper:1910.08502:p11_0", "qasper:1910.08502:p13_9", "qasper:1905.11268:p0_6"]
- pplx_decider reader answer: RNN-transducer with attention module has the smallest CER, and classic RNN-transducer has the smallest WER.
- pplx_decider frozen input hash: `3c68f3097c105ee56ce00ed2178dbfe2d263b176bb655b49dbd348907b428c7e`
- jina35 reader answer: INSUFFICIENT_EVIDENCE
- jina35 frozen input hash: `1fe5b5a1ff3ba7228a0715b6f7a6a360b63a09bc9e4a479ac967b95a1676088e`

No case is classified as factually correct or incorrect by this audit.

## qasper:1911.07555:c598028815066089cc1e131b96d6966d2610467a
- Diagnostic label: `UNRESOLVED`
- Paper: `1911.07555` — Short Text Language Identification for Under Resourced Languages
- Question: Is the lexicon the same for all languages?
- Reference status/types: `ANSWERABLE` / `YES_NO`
- Reference forms: ["yes"]
- Evidence annotations: 2; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:2003.04866:p3_0", "qasper:1909.06522:p0_1", "qasper:2003.14026:p4_4", "qasper:2001.11899:p7_0", "qasper:1909.06522:p3_0"]
- Jina selected IDs: ["qasper:1812.01704:p2_8", "qasper:2003.04866:p3_0", "qasper:1909.06522:p0_1", "qasper:1911.07555:p2_2", "qasper:2003.14026:p4_4"]
- pplx_decider reader answer: No
- pplx_decider frozen input hash: `e785b37715181fdadaa035bc3a04917f240c5abf9230b830788aa7300ca4eaf8`
- jina35 reader answer: No
- jina35 frozen input hash: `3209dc40ab0b2f68b3f6665d6a9582f94eb515b3161daadfc239a2cb6d2f8673`

No case is classified as factually correct or incorrect by this audit.

## qasper:1611.02550:1d791713d1aa77358f11501f05c108045f53c8aa
- Diagnostic label: `UNRESOLVED`
- Paper: `1611.02550` — Discriminative Acoustic Word Embeddings: Recurrent Neural Network-Based Approaches
- Question: Which dimensionality do they use for their embeddings?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["1061"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1911.07228:p4_2", "qasper:1907.04152:p8_8", "qasper:1704.02686:p9_3", "qasper:1706.08032:p8_4", "qasper:1706.09147:p3_1"]
- Jina selected IDs: ["qasper:1704.02686:p9_3", "qasper:1907.04152:p8_8", "qasper:1611.02550:p5_0", "qasper:1912.13109:p9_3", "qasper:1611.02550:p9_0"]
- pplx_decider reader answer: 300
- pplx_decider frozen input hash: `ce72c2e168b53954799f38ca4c7d6209cf9e7cf65269356c6c38ff6425136b89`
- jina35 reader answer: 300-dimensional
- jina35 frozen input hash: `9b342eb8ea3ce96b27f29fe9580e6f0ea2c7729c4e10eba7c8465ba1da966b8f`

No case is classified as factually correct or incorrect by this audit.

## qasper:1601.06068:ad362365656b0b218ba324ae60701eb25fe664c1
- Diagnostic label: `UNRESOLVED`
- Paper: `1601.06068` — Paraphrase Generation from Latent-Variable PCFGs for Semantic Parsing
- Question: What latent variables are modeled in the PCFG?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["semantic and topical information", "syntactic information"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1601.06068:p1_0", "qasper:1601.06068:p3_1", "qasper:1906.10225:p2_12", "qasper:1808.09111:p4_0", "qasper:1906.10225:p0_2"]
- Jina selected IDs: ["qasper:1906.10225:p8_4", "qasper:1906.10225:p2_12", "qasper:1906.10225:p1_2", "qasper:1601.06068:p1_0", "qasper:1906.10225:p0_2"]
- pplx_decider reader answer: syntactic, semantic, and topical information
- pplx_decider frozen input hash: `43d52abf771c46cd6397cc93d8fa5882dadd59477b2757e7da61255eb2068cf4`
- jina35 reader answer: latent vector
- jina35 frozen input hash: `a3c14c1c81aac60767c83d92f99d05b4ce4c6357094ac2268cc698ee00a01b7a`

No case is classified as factually correct or incorrect by this audit.

## qasper:1812.06705:6333845facb22f862ffc684293eccc03002a4830
- Diagnostic label: `UNRESOLVED`
- Paper: `1812.06705` — Conditional BERT Contextual Augmentation
- Question: Do the authors report performance of conditional bert on tasks without data augmentation?
- Reference status/types: `ANSWERABLE` / `YES_NO`
- Reference forms: ["yes"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1812.06705:p8_7", "qasper:1812.06705:p9_0", "qasper:1812.06705:p8_6", "qasper:1812.06705:p0_3", "qasper:1811.05085:p9_7"]
- Jina selected IDs: ["qasper:1812.06705:p8_7", "qasper:1812.06705:p6_0", "qasper:1811.05085:p12_5", "qasper:1812.06705:p10_0", "qasper:1811.05085:p9_7"]
- pplx_decider reader answer: Yes
- pplx_decider frozen input hash: `94401d7db899c46e37a8202020d517d8f22ac57a1fff0b90d147f39be51c89aa`
- jina35 reader answer: INSUFFICIENT_EVIDENCE
- jina35 frozen input hash: `0540ec6320e4ca38f80915fa23f60dc88a19496b34d237f12a38902a06083e0e`

No case is classified as factually correct or incorrect by this audit.

## qasper:1904.02357:5c70fdd3d6b67031768d3e28336942e49bf9a500
- Diagnostic label: `UNRESOLVED`
- Paper: `1904.02357` — Plan, Write, and Revise: an Interactive System for Open-Domain Story Generation
- Question: How is human interaction consumed by the model?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["displays three different versions of a story written by three distinct models for a human to compare", "human can select the model to interact with (potentially after having chosen it via cross-model), and can collaborate at all stages"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1910.11949:p2_0", "qasper:1808.04614:p15_0", "qasper:2001.07615:p0_9", "qasper:1805.00460:p1_2", "qasper:1904.02357:p4_0"]
- Jina selected IDs: ["qasper:1805.00460:p1_2", "qasper:1904.02357:p4_0", "qasper:1904.02357:p0_1", "qasper:1904.02357:p2_0", "qasper:1910.11949:p2_0"]
- pplx_decider reader answer: Human interaction is consumed by the model to learn the user's interest and to ensure the system returns correct formal-queries.
- pplx_decider frozen input hash: `5db979fd7c5a158d0f9b05ad21700c34e3664d852f811ea2d221b59d16dc7170`
- jina35 reader answer: The model consumes human interaction by using it to learn user interests and allowing users to edit, reject, or request new generations, which influences the model's output and enables iterative refinement.
- jina35 frozen input hash: `73799c4b98476bc4ae585a54d78de4345bd8f736ce7817e929c2a30cdcfd0c6e`

No case is classified as factually correct or incorrect by this audit.

## qasper:2003.07433:bdd8368debcb1bdad14c454aaf96695ac5186b09
- Diagnostic label: `UNRESOLVED`
- Paper: `2003.07433` — LAXARY: A Trustworthy Explainable Twitter Analysis Model for Post-Traumatic Stress Disorder Assessment
- Question: How is the intensity of the PTSD established?
- Reference status/types: `ANSWERABLE` / `ABSTRACTIVE`
- Reference forms: ["Given we have four intensity, No PTSD, Low Risk PTSD, Moderate Risk PTSD and High Risk PTSD with a score of 0, 1, 2 and 3 respectively, the estimated intensity  is established as mean  squared error.", "defined into four categories from high risk, moderate risk, to low risk"]
- Evidence annotations: 2; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:2003.07433:p3_0", "qasper:2003.07433:p3_1", "qasper:2003.07433:p5_0", "qasper:2003.07433:p3_3", "qasper:2003.07433:p3_2"]
- Jina selected IDs: ["qasper:2003.07433:p3_0", "qasper:2003.07433:p0_4", "qasper:2003.07433:p11_0", "qasper:2003.07433:p13_0", "qasper:2003.07433:p0_5"]
- pplx_decider reader answer: The intensity of PTSD is established by assessing weekly scores on three tools (DOSPERT, BSSS, and VIAS) and categorizing based on how many tools exceed their thresholds: low risk (one tool), moderate risk (two tools), or high risk (all three tools).
- pplx_decider frozen input hash: `fdbffb21776ba9ca88b9315fc6f12cdc92251d8f6e6b7931a94a2af971a71854`
- jina35 reader answer: By calculating the cumulative sum of s-scores derived from α-scores scaled according to survey systems like DOSPERT, BSSS, and VIAS.
- jina35 frozen input hash: `6781d3ba8fd239692d5bc822d21cdc5832f0582b431e2ebf6a22168e48f722d3`

No case is classified as factually correct or incorrect by this audit.

## qasper:1603.01417:1ce26783f0ff38925bfc07bbbb65d206e52c2d21
- Diagnostic label: `UNRESOLVED`
- Paper: `1603.01417` — Dynamic Memory Networks for Visual and Textual Question Answering
- Question: Does the DMN+ model establish state-of-the-art ?
- Reference status/types: `ANSWERABLE` / `YES_NO`
- Reference forms: ["yes"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1603.01417:p12_6", "qasper:1603.01417:p14_0", "qasper:1603.01417:p12_3", "qasper:1603.01417:p11_4", "qasper:1603.01417:p0_2"]
- Jina selected IDs: ["qasper:1603.01417:p12_6", "qasper:1603.01417:p0_1", "qasper:1603.01417:p12_3", "qasper:1603.01417:p11_4", "qasper:1603.01417:p0_2"]
- pplx_decider reader answer: Yes
- pplx_decider frozen input hash: `f667ad5c60fb304b72356de581e88305acdce7a08540aee151f30cde8be0ef46`
- jina35 reader answer: Yes
- jina35 frozen input hash: `c813f9cc85b53f7e51fcb7f0bca61a6f1b9f9c8b857afd90c55edb8938af336d`

No case is classified as factually correct or incorrect by this audit.

## qasper:2002.06675:8a5254ca726a2914214a4c0b6b42811a007ecfc6
- Diagnostic label: `UNRESOLVED`
- Paper: `2002.06675` — Speech Corpus of Ainu Folklore and End-to-end Speech Recognition for Ainu Language
- Question: How much transcribed data is available for for Ainu language?
- Reference status/types: `ANSWERABLE` / `ABSTRACTIVE`
- Reference forms: ["Transcribed data is available for duration of 38h 54m 38s for 8 speakers."]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:2002.06675:p5_0", "qasper:2002.06675:p0_1", "qasper:2002.06675:p5_1", "qasper:2002.06675:p2_0", "qasper:2002.06675:p7_0"]
- Jina selected IDs: ["qasper:2002.06675:p2_0", "qasper:2002.06675:p0_1", "qasper:2002.06675:p1_0", "qasper:2002.06675:p3_1", "qasper:1804.11297:p0_0"]
- pplx_decider reader answer: INSUFFICIENT_EVIDENCE
- pplx_decider frozen input hash: `b8d709f9c1beeb69acdf9683da8692b906609b873e5ed4fab686c0955dd42949`
- jina35 reader answer: INSUFFICIENT_EVIDENCE
- jina35 frozen input hash: `8188de25fcca650efd9899f3334d956e21176c1c4fadb053428318b144c4012c`

No case is classified as factually correct or incorrect by this audit.

## qasper:1909.00279:d484a71e23d128f146182dccc30001df35cdf93f
- Diagnostic label: `UNRESOLVED`
- Paper: `1909.00279` — Generating Classical Chinese Poems from Vernacular Chinese
- Question: How much is proposed model better in perplexity and BLEU score than typical UMT models?
- Reference status/types: `ANSWERABLE` / `ABSTRACTIVE`
- Reference forms: ["Perplexity of the best model is 65.58 compared to best baseline 105.79.\nBleu of the best model is 6.57 compared to best baseline 5.50."]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1701.06538:p11_1", "qasper:1811.11365:p10_1", "qasper:1905.01715:p8_2", "qasper:1701.06538:p10_2", "qasper:1701.06538:p18_6"]
- Jina selected IDs: ["qasper:1701.06538:p10_2", "qasper:1811.11365:p10_1", "qasper:1701.06538:p18_6", "qasper:1701.06538:p11_1", "qasper:1909.00279:p10_0"]
- pplx_decider reader answer: INSUFFICIENT_EVIDENCE
- pplx_decider frozen input hash: `a1a42f2f02f27e9afd4b3c508d93a24deb5dce462067d0fa177c8e9ce61edc00`
- jina35 reader answer: INSUFFICIENT_EVIDENCE
- jina35 frozen input hash: `06cede2d43cf0fdd83784ba309e1075556215ad0009fb45af9acca2039b8e99b`

No case is classified as factually correct or incorrect by this audit.

## qasper:1909.00326:93beae291b455e5d3ecea6ac73b83632a3ae7ec7
- Diagnostic label: `UNRESOLVED`
- Paper: `1909.00326` — Towards Understanding Neural Machine Translation with Word Importance
- Question: How do their models decide how much improtance to give to the output words?
- Reference status/types: `ANSWERABLE` / `MIXED_ANNOTATED`
- Reference forms: ["Given the contribution matrix, we can obtain the word importance of each input word to the entire output sentence.", "They compute the gradient of the output at each time step with respect to the input words to decide the importance."]
- Evidence annotations: 2; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1912.00667:p6_5", "qasper:1901.03859:p7_19", "qasper:1910.07181:p2_1", "qasper:1811.12889:p0_1", "qasper:1611.08661:p6_3"]
- Jina selected IDs: ["qasper:1909.00326:p6_4", "qasper:1901.03859:p7_19", "qasper:1909.00326:p5_0", "qasper:1907.05664:p10_1", "qasper:1909.00326:p8_0"]
- pplx_decider reader answer: The model uses a gating mechanism to decide how much attention to pay to each pair of form and context embeddings.
- pplx_decider frozen input hash: `d23a866a0c4d9eb7a8e074ac196b8826cb54526db0afcae2d8302ca5f3949c0f`
- jina35 reader answer: They aggregate the contribution values of each input word to all output words by summing them and then normalize the sums using the Softmax function.
- jina35 frozen input hash: `c776d77a4735416e6f094ac64a1c79f055b1afaf705cd5da852d85f83125903f`

No case is classified as factually correct or incorrect by this audit.

## qasper:1805.10824:d00bbeda2a45495e6261548710afa6b21ea32870
- Diagnostic label: `UNRESOLVED`
- Paper: `1805.10824` — UG18 at SemEval-2018 Task 1: Generating Additional Training Data for Predicting Emotion Intensity in Spanish
- Question: What semi-supervised learning is applied?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["This silver data is then simply added to our training set, after which the model is retrained", "first a model is trained on the training set and then this model is used to predict the labels of the silver data"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1805.10824:p6_2", "qasper:1908.01294:p0_6", "qasper:1701.09123:p8_0", "qasper:1911.00069:p20_3", "qasper:1809.00530:p0_4"]
- Jina selected IDs: ["qasper:1908.01294:p0_6", "qasper:1701.09123:p4_7", "qasper:1805.10824:p6_0", "qasper:1809.00530:p1_2", "qasper:1710.01492:p7_4"]
- pplx_decider reader answer: Cross-View Training
- pplx_decider frozen input hash: `ccc7916823d75f583f0695b644cc6496c7097e62462018342dbabafa28b683b9`
- jina35 reader answer: Cross-View Training
- jina35 frozen input hash: `f88c67cd6cbe5ac40b0fe63c023981ab3cdc14dc6a4a412f019c75f91e345e6e`

No case is classified as factually correct or incorrect by this audit.

## qasper:1909.05246:bd7a95b961af7caebf0430a7c9f675816c9c527f
- Diagnostic label: `UNRESOLVED`
- Paper: `1909.05246` — Self-Attentional Models Application in Task-Oriented Dialogue Generation Systems
- Question: What are the three datasets used?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["DSTC2", "M2M-sim-M", "M2M-sim-R"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1806.00722:p6_0", "qasper:1706.02027:p8_0", "qasper:1611.02988:p2_0", "qasper:1711.05345:p5_0", "qasper:1709.10367:p4_1"]
- Jina selected IDs: ["qasper:1806.00722:p6_0", "qasper:1709.02271:p4_0", "qasper:1909.05246:p8_0", "qasper:1706.02027:p8_0", "qasper:1603.01417:p7_0"]
- pplx_decider reader answer: IWSLT14 German-English, Turkish-English, and WMT14 English-German
- pplx_decider frozen input hash: `2a15168c56d8ed9b24cf1e81b30d37a99dc1ac18d4ac24212d598848fab027ce`
- jina35 reader answer: IWSLT14 German-English, Turkish-English, and WMT14 English-German
- jina35 frozen input hash: `96af99bdf4e40de8dd68d36d215929efea1cd6e261544de1bf5d5401e0f204c7`

No case is classified as factually correct or incorrect by this audit.

## qasper:2001.01269:cf171fad0bea5ab985c53d11e48e7883c23cdc44
- Diagnostic label: `UNRESOLVED`
- Paper: `2001.01269` — Generating Word and Document Embeddings for Sentiment Analysis
- Question: What details are given about the Twitter dataset?
- Reference status/types: `ANSWERABLE` / `MIXED_ANNOTATED`
- Reference forms: ["Those tweets are mostly much noisier and shorter compared to the reviews in the movie corpus. In total, there are 1,716 tweets. 973 of them are negative and 743 of them are positive.", "one of the Twitter datasets is about Turkish mobile network operators, there are positive, neutral and negative labels and provide the total amount plus the distribution of labels"]
- Evidence annotations: 2; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1809.09795:p3_1", "qasper:1803.09745:p1_3", "qasper:1907.01468:p3_11", "qasper:1701.04653:p2_2", "qasper:2001.01269:p8_2"]
- Jina selected IDs: ["qasper:1803.09745:p1_3", "qasper:1910.06748:p4_0", "qasper:1809.09795:p3_1", "qasper:1909.10012:p3_3", "qasper:1910.06748:p0_1"]
- pplx_decider reader answer: The Twitter dataset used for SemEval 2018 Task 3 is manually annotated with binary labels for irony detection, includes a manually annotated dataset for sarcasm, and a user self-annotated corpus of tweets with the #sarcasm hashtag. Another Twitter dataset consists of a random sample of roughly 1
- pplx_decider frozen input hash: `93d9821056a3d6c6c5d108cf10189cbba3f39cf836a047194bd886087fc9f437`
- jina35 reader answer: The Twitter dataset consists of a random sample of roughly 10% of all tweets posted between 9 September 2008 and 22 October 2017, comprising more than 106 billion messages from about 750 million unique accounts. It includes verb forms of interest,
- jina35 frozen input hash: `d76651563109b1d80da431c5e53245f7a89bed035ba24d79db736bcf496e7ec2`

No case is classified as factually correct or incorrect by this audit.

## qasper:1911.08673:9aa52b898d029af615b95b18b79078e9bed3d766
- Diagnostic label: `UNRESOLVED`
- Paper: `1911.08673` — Global Greedy Dependency Parsing
- Question: How faster is training and decoding compared to former models?
- Reference status/types: `ANSWERABLE` / `ABSTRACTIVE`
- Reference forms: ["Proposed vs best baselin<redacted-path> 8541 vs 8532 tokens/sec\nTraining: 8h vs 8h"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1809.02731:p5_0", "qasper:1705.01991:p11_3", "qasper:1911.08673:p13_0", "qasper:1911.08673:p16_0", "qasper:1705.01991:p0_2"]
- Jina selected IDs: ["qasper:1705.01991:p11_3", "qasper:1705.01991:p0_2", "qasper:1911.08673:p13_0", "qasper:1703.07090:p0_6", "qasper:1705.01991:p9_1"]
- pplx_decider reader answer: faster
- pplx_decider frozen input hash: `063086e5ee4e4191077e23f229daaa4dba54b048124744f17ef2e1759fd6eb81`
- jina35 reader answer: 3x faster to decode and 1.5x faster to train compared to former models.
- jina35 frozen input hash: `229ac17926c39f5967f24b1a9fcf9a0ae31efe3fcffd4a314ce0fa9b54f143e4`

No case is classified as factually correct or incorrect by this audit.

## qasper:1809.01060:6b7d76c1e1a2490beb69609ba5652476dde8831b
- Diagnostic label: `UNRESOLVED`
- Paper: `1809.01060` — The Effect of Context on Metaphor Paraphrase Aptness Judgments
- Question: What provisional explanation do the authors give for the impact of document context?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["adding context causes speakers to focus on broader semantic and pragmatic issues of discourse coherence"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1811.00266:p5_0", "qasper:1809.01060:p2_3", "qasper:1906.03338:p5_8", "qasper:1909.02776:p1_7", "qasper:2004.03034:p13_0"]
- Jina selected IDs: ["qasper:1809.01060:p2_4", "qasper:1909.02776:p1_7", "qasper:1809.01060:p2_3", "qasper:2004.03034:p2_10", "qasper:2004.03034:p3_0"]
- pplx_decider reader answer: Document context can improve or reduce human ratings depending on the sentence's acceptability, and models relying on discourse-based indicators may be misled in cross-document settings.
- pplx_decider frozen input hash: `ecb87cd5e0c380699b4cdd201572374e3cd6339ec18e1deabf5c29f2296cf55c`
- jina35 reader answer: Adding context causes speakers to focus on broader semantic and pragmatic issues of discourse coherence, rather than simply judging syntactic well formedness.
- jina35 frozen input hash: `56d97b447e813eef725e26351b9823c7501a5c82dcf59c6961ed7be6ce3f6f3d`

No case is classified as factually correct or incorrect by this audit.

## qasper:2002.06675:526ae24fa861d52536b66bcc2d2ddfce483511d6
- Diagnostic label: `UNRESOLVED`
- Paper: `2002.06675` — Speech Corpus of Ainu Folklore and End-to-end Speech Recognition for Ainu Language
- Question: How big are improvements with multilingual ASR training vs single language training?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["relative WER improvement of 10%."]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:2002.06675:p20_5", "qasper:1909.06522:p12_0", "qasper:1809.01541:p5_4", "qasper:1909.06522:p14_0", "qasper:1909.06522:p1_0"]
- Jina selected IDs: ["qasper:1809.01541:p5_4", "qasper:1909.06522:p14_0", "qasper:1909.06522:p12_3", "qasper:1909.06522:p12_0", "qasper:1909.06522:p13_3"]
- pplx_decider reader answer: Improvements range from 4.6% to 10.3% relative WER reduction.
- pplx_decider frozen input hash: `fa280d2d03e86e169f5aada388a3f068e47590ac8991363cbda7e2d000ce7ff8`
- jina35 reader answer: 7.96% higher on average
- jina35 frozen input hash: `42b8a94fa031f3b446e3d04a60059cec9e3d41b653c5d254615381462dc85f1f`

No case is classified as factually correct or incorrect by this audit.

## qasper:1701.06538:0cd90e5b79ea426ada0203177c28812a7fc86be5
- Diagnostic label: `UNRESOLVED`
- Paper: `1701.06538` — Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
- Question: How is the correct number of experts to use decided?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["varied the number of experts between models"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1701.06538:p18_7", "qasper:1907.01468:p13_2", "qasper:1701.06538:p5_3", "qasper:1701.06538:p19_4", "qasper:1701.06538:p3_2"]
- Jina selected IDs: ["qasper:1701.06538:p18_7", "qasper:1811.01299:p9_6", "qasper:1701.06538:p16_1", "qasper:1701.06538:p18_2", "qasper:1907.01468:p13_2"]
- pplx_decider reader answer: The correct number is determined by the needs of the user, not by the collection.
- pplx_decider frozen input hash: `d0277e84a78391898395a3a1e898380a05b637212fdef743caf206e9c0e11366`
- jina35 reader answer: The correct number of experts is decided based on performance improvement, as increasing the number of experts continued to improve test perplexity up to 2048.
- jina35 frozen input hash: `936b068db0884b391ab572087a427f5abda549a11b00ae86de5c8c61fad3110d`

No case is classified as factually correct or incorrect by this audit.

## qasper:1911.03894:5471766ca7c995dd7f0f449407902b32ac9db269
- Diagnostic label: `UNRESOLVED`
- Paper: `1911.03894` — CamemBERT: a Tasty French Language Model
- Question: How much better was results of CamemBERT than previous results on these tasks?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["2.36 point increase in the F1 score with respect to the best SEM architecture", "For POS tagging, we observe error reductions of respectively 0.71% for GSD, 0.81% for Sequoia, 0.7% for Spoken and 0.28% for ParTUT", "For parsing, we observe error reductions in LAS of 2.96% for GSD, 3.33% for Sequoia, 1.70% for Spoken and 1.65% for ParTUT", "lags behind models trained on the original English training set in the TRANSLATE-TEST setting, 81.2 vs. 82.91 for RoBERTa", "on the TRANSLATE-TRAIN setting (81.2 vs. 80.2 for XLM)"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1911.03894:p22_0", "qasper:1911.03894:p19_0", "qasper:1911.03894:p21_0", "qasper:1911.03894:p20_0", "qasper:1911.03894:p0_2"]
- Jina selected IDs: ["qasper:1911.03894:p22_0", "qasper:1911.03894:p19_1", "qasper:1911.03894:p21_0", "qasper:1911.03894:p0_2", "qasper:1911.03894:p20_0"]
- pplx_decider reader answer: CamemBERT improved state-of-the-art performance for most tasks over previous monolingual and multilingual approaches.
- pplx_decider frozen input hash: `3be4d14502e50266b6059bad1316f6e6f8525a8372818ee49b108dc768a6ccb9`
- jina35 reader answer: CamemBERT improved performance by 0.71% to 3.5 points and 2.36 points F1 score increase over previous models on specific tasks.
- jina35 frozen input hash: `cd71170e72b502c4358e06ee241c7e76e9685ef9296ec5efe070ade80e9ba525`

No case is classified as factually correct or incorrect by this audit.

## qasper:1907.02636:ef872807cb0c9974d18bbb886a7836e793727c3d
- Diagnostic label: `UNRESOLVED`
- Paper: `1907.02636` — Collecting Indicators of Compromise from Unstructured Text of Cybersecurity Articles using Neural-Based Sequence Labelling
- Question: What contextual features are used?
- Reference status/types: `ANSWERABLE` / `ABSTRACTIVE`
- Reference forms: ["The words that can indicate the characteristics of the neighbor words as contextual keywords and generate it from the automatically extracted contextual keywords."]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:2003.08553:p7_0", "qasper:1602.06291:p0_8", "qasper:1912.03234:p2_0", "qasper:2003.08553:p7_3", "qasper:1907.02636:p7_0"]
- Jina selected IDs: ["qasper:2003.08553:p7_3", "qasper:1907.02636:p7_2", "qasper:1907.02636:p7_5", "qasper:1907.02636:p12_0", "qasper:1602.06291:p0_8"]
- pplx_decider reader answer: topics based on different segments of text, contextual keywords, co-occurring words indicating characteristics of neighbor words
- pplx_decider frozen input hash: `a18cefa17b0a0947c20a4ac850f1153654182ea0f0865f77209e22add494e7a9`
- jina35 reader answer: contextual keywords, window size, token frequency, corpus frequency
- jina35 frozen input hash: `559c89c1f9623cdc996933eb416b090654dcbfd97008902b83a728f6b77e1e66`

No case is classified as factually correct or incorrect by this audit.

## qasper:1806.00738:c37f65c9f0d543a35c784263b79236ccf1c44fac
- Diagnostic label: `UNRESOLVED`
- Paper: `1806.00738` — Contextualize, Show and Tell: A Neural Visual Storyteller
- Question: What model is used to encode the images?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["LSTM", "a Convolutional Neural Network (CNN)"]
- Evidence annotations: 2; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1711.06288:p7_2", "qasper:1906.00378:p3_2", "qasper:1811.11365:p10_0", "qasper:1909.00279:p11_3", "qasper:1812.10479:p13_5"]
- Jina selected IDs: ["qasper:1809.01341:p5_4", "qasper:1806.00738:p3_0", "qasper:1906.00378:p3_2", "qasper:1711.06288:p7_2", "qasper:1811.11365:p10_0"]
- pplx_decider reader answer: VGG-16
- pplx_decider frozen input hash: `fb641c52da3a5a1638ad79f9c37caa6d45410bb031ab1d1c0a62c487b4467955`
- jina35 reader answer: ResNet-152
- jina35 frozen input hash: `96ec2e840215f24005154ac580dc94b519cb2b49473fe910d21117123282b407`

No case is classified as factually correct or incorrect by this audit.

## qasper:2002.12612:dd20d93166c14f1e57644cd7fa7b5e5738025cd0
- Diagnostic label: `UNRESOLVED`
- Paper: `2002.12612` — A multi-layer approach to disinformation detection on Twitter
- Question: Which two news domains are country-independent?
- Reference status/types: `ANSWERABLE` / `MIXED_ANNOTATED`
- Reference forms: ["mainstream and disinformation news", "mainstream news and disinformation"]
- Evidence annotations: 2; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:2002.12612:p10_5", "qasper:2002.12612:p12_2", "qasper:2002.12612:p1_2", "qasper:2002.12612:p0_7", "qasper:2002.12612:p3_1"]
- Jina selected IDs: ["qasper:2002.12612:p10_5", "qasper:2002.12612:p12_2", "qasper:2002.12612:p0_7", "qasper:1910.09295:p12_0", "qasper:2002.12612:p10_7"]
- pplx_decider reader answer: mainstream and disinformation news
- pplx_decider frozen input hash: `e0d8179cf22547d0eab5218ab206af20c2b638adb41d9147004b75e141bad807`
- jina35 reader answer: political news and disinformation news
- jina35 frozen input hash: `5d8563da662f72e1f65155848b091ff2949d3a38f4cc3758bf15572d6a6f1a3b`

No case is classified as factually correct or incorrect by this audit.

## qasper:2001.06888:0106bd9d54e2f343cc5f30bb09a5dbdd171e964b
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `2001.06888` — A multimodal deep learning approach for named entity recognition from social media
- Question: Which social media platform is explored?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["twitter"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1908.00153:p3_2", "qasper:1710.01492:p3_2", "qasper:1608.04917:p1_0", "qasper:2001.08845:p0_1", "qasper:1908.04531:p3_12"]
- Jina selected IDs: ["qasper:2001.08845:p0_1", "qasper:1710.01492:p3_2", "qasper:1908.00153:p3_2", "qasper:1908.04531:p3_12", "qasper:1610.04377:p0_0"]
- pplx_decider reader answer: Twitter
- pplx_decider frozen input hash: `c128d3e1d924965fc96f5635cacb1797d8d6bdcc2528132a2c53f9dd93bf8a84`
- jina35 reader answer: Sina Weibo
- jina35 frozen input hash: `f355230147dd2e37407d94d87a8e85c8d1d13a2b3724c2eb54b74f0ddc3c3981`

No case is classified as factually correct or incorrect by this audit.

## qasper:1908.06138:8b43201e7e648c670c02e16ba189230820879228
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1908.06138` — UDS--DFKI Submission to the WMT2019 Similar Language Translation Shared Task
- Question: Does the use of out-of-domain data improve the performance of the method?
- Reference status/types: `ANSWERABLE` / `YES_NO`
- Reference forms: ["no"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1705.01265:p4_5", "qasper:1810.05334:p7_1", "qasper:1904.04019:p17_3", "qasper:1701.03214:p10_3", "qasper:1909.02764:p15_1"]
- Jina selected IDs: ["qasper:1804.07789:p14_0", "qasper:1612.06897:p6_1", "qasper:1701.03214:p5_4", "qasper:1701.09123:p1_5", "qasper:1705.01265:p4_5"]
- pplx_decider reader answer: Yes
- pplx_decider frozen input hash: `619e1e559570c0a306d4c6db5a8fdcaa848b4a52cc16d7a165867d33c429ab9d`
- jina35 reader answer: No
- jina35 frozen input hash: `3af0c9663d1fde2bcd7103497c5d5b0bfaaab77fa472d037319e2313e9724ab4`

No case is classified as factually correct or incorrect by this audit.

## qasper:2001.02885:78536da059b884d6ad04680baeb894895458055c
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `2001.02885` — Resolving the Scope of Speculation and Negation using Transformer-Based Architectures
- Question: What were the baselines?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["Recursive Neural Networks (BIBREF9,BIBREF10), Convolutional Neural Networks (BIBREF11) and most recently transfer learning-based architectures like Bidirectional Encoder Representation from Transformers (BERT) (BIBREF12)", "varied from Maximum Entropy Classifiers (BIBREF4) to Support Vector Machines (BIBREF5,BIBREF6,BIBREF7,BIBREF8)"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1908.09246:p10_8", "qasper:1811.00625:p8_0", "qasper:1811.04604:p9_0", "qasper:1901.01010:p9_0", "qasper:1909.09484:p14_0"]
- Jina selected IDs: ["qasper:1811.04604:p9_0", "qasper:1811.00625:p8_0", "qasper:1703.10344:p5_19", "qasper:2001.07786:p5_0", "qasper:1909.09524:p6_12"]
- pplx_decider reader answer: The baselines were models chosen for comparison in the experiments.
- pplx_decider frozen input hash: `6942a25fa12702630bb35a6921b8a0519f3b3698d8a2ceec9cc8ffbe41e4b382`
- jina35 reader answer: The baselines were models used for comparison in the task.
- jina35 frozen input hash: `291aea943200c2f5b1d655a1e28db33a2c482ffb890b64502abe0968cdb527df`

No case is classified as factually correct or incorrect by this audit.

## qasper:1906.00378:9e805020132d950b54531b1a2620f61552f06114
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1906.00378` — Unsupervised Bilingual Lexicon Induction from Mono-lingual Multimodal Data
- Question: What baseline is used for the experimental setup?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["CNN-avgmax", "CNN-mean"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1909.07593:p8_0", "qasper:2002.01207:p7_1", "qasper:2002.01207:p11_1", "qasper:1811.04604:p9_0", "qasper:1809.00530:p9_0"]
- Jina selected IDs: ["qasper:2002.01207:p11_1", "qasper:1909.07593:p8_0", "qasper:2002.01207:p7_1", "qasper:1804.07789:p5_0", "qasper:1908.09246:p9_0"]
- pplx_decider reader answer: CHAR and word
- pplx_decider frozen input hash: `b0db5262254cafa350fbc5ce77d0835dc2d155798ef195f1cbca951d36da8de0`
- jina35 reader answer: word surface forms only
- jina35 frozen input hash: `5e25f2cc376a6a7124bbba714756666499ba336478387c98519e441b5392290d`

No case is classified as factually correct or incorrect by this audit.

## qasper:1901.00439:fac052c4ad6b19a64d7db32fd08df38ad2e22118
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1901.00439` — Deep Representation Learning for Clustering of Health Tweets
- Question: How do they evaluate their method?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["Calinski-Harabasz score", "UMAP", "t-SNE"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1911.03350:p0_7", "qasper:1808.03738:p6_0", "qasper:1601.02543:p0_1", "qasper:1911.03350:p5_0", "qasper:2003.04642:p0_3"]
- Jina selected IDs: ["qasper:1910.10781:p1_0", "qasper:1708.01464:p5_2", "qasper:1911.09845:p10_0", "qasper:2004.01694:p2_2", "qasper:1601.02543:p0_1"]
- pplx_decider reader answer: They evaluate their method through a human evaluation.
- pplx_decider frozen input hash: `0de986226f3960c6d03b17f0db53660e436bd80ddf7fa05f1b2958d42f8f491a`
- jina35 reader answer: They compute automatic metrics on their test set.
- jina35 frozen input hash: `b8e028f5c484d45075652d5d665fe0f837584f932cb9b655ccef3f3f894af650`

No case is classified as factually correct or incorrect by this audit.

## qasper:1905.10702:52e8c9ed66ace1780e41815260af1309064d20de
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1905.10702` — MDE: Multi Distance Embeddings for Link Prediction in Knowledge Graphs
- Question: What datasets are used to evaluate the model?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["WN18 and FB15k"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1901.03866:p8_0", "qasper:1912.11602:p9_0", "qasper:1911.05343:p3_0", "qasper:1910.14537:p5_0", "qasper:2002.06424:p7_0"]
- Jina selected IDs: ["qasper:1901.03866:p8_0", "qasper:1911.05343:p3_0", "qasper:1811.09353:p6_0", "qasper:1912.11602:p9_0", "qasper:1909.05246:p8_0"]
- pplx_decider reader answer: QuasarT, TriviaQA, SearchQA, NYT, XSum, CNN/DailyMail, DUC-2003, DUC-2004, PTB, E2E, PKU, MSR, AS, CITYU
- pplx_decider frozen input hash: `70515af1a7af707e72e37e0d43c82e93729fbc673af919ea6f61155ed5526871`
- jina35 reader answer: QuasarT, TriviaQA, SearchQA, Penn Treebank, E2E, NYT, XSum, CNN/DailyMail, DUC-2003, DUC-2004, DSTC2, M2M-sim-M, M2M-sim-R
- jina35 frozen input hash: `01cca7637847fa8efc2a452dc9ac4f086c1a85350a0543a8e42907815a230f30`

No case is classified as factually correct or incorrect by this audit.

## qasper:1901.03866:efe49829725cfe54de01405c76149a4fe4d18747
- Diagnostic label: `UNRESOLVED`
- Paper: `1901.03866` — HAS-QA: Hierarchical Answer Spans Model for Open-domain Question Answering
- Question: How much does HAS-QA improve over baselines?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["For example, in QuasarT, it improves 16.8% in EM score and 20.4% in F1 score.", "For example, in QuasarT, it improves 4.6% in EM score and 3.5% in F1 score."]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1901.03866:p10_2", "qasper:1901.03866:p10_1", "qasper:1901.03866:p12_0", "qasper:1901.03866:p0_12", "qasper:1901.03866:p11_2"]
- Jina selected IDs: ["qasper:1901.03866:p10_2", "qasper:1901.03866:p10_1", "qasper:1901.03866:p0_12", "qasper:1901.03866:p11_2", "qasper:1901.03866:p12_0"]
- pplx_decider reader answer: 4.6% in EM score and 3.5% in F1 score
- pplx_decider frozen input hash: `fea664389ab7eb9be789a088e1ae250086a6cd8c11f048ead548f72e88b92a9f`
- jina35 reader answer: 4.6% in EM score and 3.5% in F1 score
- jina35 frozen input hash: `7237b37eae40aeeba48a193b16d5744a3416d7eb63b12418a586368243fcc2be`

No case is classified as factually correct or incorrect by this audit.

## qasper:1911.01188:96c20af8bbef435d0d534d10c42ae15ff2f926f8
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1911.01188` — Analysing Coreference in Transformer Outputs
- Question: What translationese effects are seen in the analysis?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["explicitation effect", "potentially indicating a shining through effect"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1911.01188:p2_2", "qasper:1903.03467:p5_0", "qasper:2004.01694:p17_5", "qasper:2004.01694:p17_1", "qasper:2004.01694:p17_2"]
- Jina selected IDs: ["qasper:2004.04721:p4_0", "qasper:2004.01694:p17_2", "qasper:1911.01188:p2_2", "qasper:2004.01694:p17_5", "qasper:2004.01694:p17_1"]
- pplx_decider reader answer: simpler, more explicit, and more normalised
- pplx_decider frozen input hash: `c49cbd4a989ea8234f802b83c8aa4d72e1ee85805cd73000201877c8fef7c5d4`
- jina35 reader answer: simplification, explicitation, normalization, interference
- jina35 frozen input hash: `d6fa39faa9e420b64cbdea738800abf0c6d3b0b6deb084e42a1b27d00c8d61f8`

No case is classified as factually correct or incorrect by this audit.

## qasper:2003.04032:269b05b74d5215b09c16e95a91ae50caedd9e2aa
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `2003.04032` — Shallow Discourse Annotation for Chinese TED Talks
- Question: Which inter-annotator metric do they use?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["Kappa value", "agreement rates"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:2003.04866:p8_17", "qasper:2003.04642:p10_1", "qasper:1809.05752:p4_0", "qasper:1808.06834:p4_1", "qasper:1912.03184:p27_0"]
- Jina selected IDs: ["qasper:1809.05752:p4_0", "qasper:1912.00819:p5_2", "qasper:1912.08320:p35_73", "qasper:1601.02403:p7_8", "qasper:1912.08320:p18_0"]
- pplx_decider reader answer: average pairwise inter-annotator agreement (APIAA)
- pplx_decider frozen input hash: `2fddc077d546ead080fabbfd896e0b589a84aa01451ef5ab78fbdbabeeadeae7`
- jina35 reader answer: Fleiss's Kappa
- jina35 frozen input hash: `cb39cc5a4431d9db5859be38d3d9e6cb1e70d4bba4dad26f5498b1dea05bc4dd`

No case is classified as factually correct or incorrect by this audit.

## qasper:2003.08529:9174aded45bc36915f2e2adb6f352f3c7d9ada8b
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `2003.08529` — Diversity, Density, and Homogeneity: Quantitative Characteristic Metrics for Text Collections
- Question: Which real-world datasets did they use?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["SST-2", "SST-2 (Stanford Sentiment Treebank, version 2)", "Snips"]
- Evidence annotations: 2; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1703.03097:p7_0", "qasper:2001.05493:p9_0", "qasper:1912.00667:p0_7", "qasper:2004.01670:p1_8", "qasper:1705.04153:p8_5"]
- Jina selected IDs: ["qasper:1804.09301:p0_4", "qasper:1703.03097:p7_0", "qasper:2001.05493:p9_0", "qasper:1912.00667:p0_7", "qasper:1909.09551:p6_7"]
- pplx_decider reader answer: TRAC 2018 Dataset, Kaggle Dataset
- pplx_decider frozen input hash: `b9b285f4f7742792a41671bdb68747f602325aff2bfacf25d729e96c2d500543`
- jina35 reader answer: TRAC 2018 Dataset, Kaggle Dataset, four real-world human trafficking datasets, large real-world LBSN dataset
- jina35 frozen input hash: `6e28031b696a375748780173865a50facd238d5cf4eb5f2404361706d956f0a5`

No case is classified as factually correct or incorrect by this audit.

## qasper:1908.06083:c84590ba32df470a7c5343d8b99e541b217f10cf
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1908.06083` — Build it Break it Fix it for Dialogue Safety: Robustness from Adversarial Human Attack
- Question: What datasets are used?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["The Wikipedia Toxic Comments dataset"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:2001.05493:p9_0", "qasper:1909.05246:p8_0", "qasper:1804.08139:p7_1", "qasper:1910.10781:p5_0", "qasper:1911.02855:p17_0"]
- Jina selected IDs: ["qasper:1909.05246:p8_0", "qasper:2001.05493:p9_0", "qasper:1809.00530:p6_5", "qasper:1911.02855:p17_0", "qasper:1904.04019:p0_21"]
- pplx_decider reader answer: TRAC 2018 Dataset, Kaggle Dataset, DSTC2, M2M-sim-M, M2M-sim-R, 16 review corpora, SQuAD v1.1, SQuAD v2.0, Quoref
- pplx_decider frozen input hash: `f3d88a5a8e0c6ad777694a09e81ff9e2bf2f3eaf7351cfede66cb90792b59354`
- jina35 reader answer: DSTC2, M2M-sim-M, M2M-sim-R, TRAC 2018 Dataset, Kaggle Dataset, IMDB, Yelp2014, Cell Phone, Baby, SQuAD v1.1, SQuAD v2.0, Quoref
- jina35 frozen input hash: `aaa2a6143c35584fa04a2a0244d0b9b0da50611778363bbac3926eaad39d7efd`

No case is classified as factually correct or incorrect by this audit.

## qasper:1910.11768:356e462f7966e30665a387ed7a9ad2e830479da6
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1910.11768` — Exploring Multilingual Syntactic Sentence Representations
- Question: Which corpus do they use?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["The dataset was created by using translations provided by Tatoeba and OpenSubtitles BIBREF16."]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:2001.02885:p0_4", "qasper:1911.07228:p3_0", "qasper:1710.06536:p14_16", "qasper:1904.04019:p14_3", "qasper:1710.00341:p8_3"]
- Jina selected IDs: ["qasper:2001.02885:p0_4", "qasper:1904.04019:p14_3", "qasper:1909.01247:p25_1", "qasper:1911.07228:p3_0", "qasper:1908.10322:p8_3"]
- pplx_decider reader answer: BioScope Abstracts, BioScope Full Papers, and SFU Review Corpus
- pplx_decider frozen input hash: `8bac846f5e49878ec1e11acbd90938c5e6ccbe4c2b5b936ca64a05285ec31a90`
- jina35 reader answer: BioScope Corpus
- jina35 frozen input hash: `34d792eebd613815d22afbcf3d245cbe37effbfca436e3463b41b1d1dd249b2a`

No case is classified as factually correct or incorrect by this audit.

## qasper:2001.08845:fc65f19a30150a0e981fb69c1f5720f0136325b0
- Diagnostic label: `UNRESOLVED`
- Paper: `2001.08845` — Linguistic Fingerprints of Internet Censorship: the Case of SinaWeibo
- Question: Is is known whether Sina Weibo posts are censored by humans or some automatic classifier?
- Reference status/types: `ANSWERABLE` / `YES_NO`
- Reference forms: ["no"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:2001.08845:p0_2", "qasper:2001.08845:p0_1", "qasper:2001.08845:p18_18", "qasper:2001.08845:p0_5", "qasper:2001.08845:p1_0"]
- Jina selected IDs: ["qasper:2001.08845:p5_0", "qasper:2001.08845:p0_1", "qasper:2001.08845:p0_2", "qasper:2001.08845:p0_5", "qasper:2001.08845:p18_18"]
- pplx_decider reader answer: Yes
- pplx_decider frozen input hash: `452121252ec07e563bd2b7383fe67722245e04cd61e110847f6ce5818db40d3d`
- jina35 reader answer: Yes
- jina35 frozen input hash: `b12da3f9c97acec2cd5e46b3ca2aa76274f0fde14727acc0b1b309b75a7e9957`

No case is classified as factually correct or incorrect by this audit.

## qasper:1808.03894:2efdcebebeb970021233553104553205ce5d6567
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1808.03894` — Interpreting Recurrent and Attention-Based Neural Models: a Case Study on Natural Language Inference
- Question: How many layers are there in their model?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["two LSTM layers"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1712.03556:p1_1", "qasper:1812.10479:p13_1", "qasper:1809.03449:p7_0", "qasper:1902.00756:p12_0", "qasper:1909.00100:p13_5"]
- Jina selected IDs: ["qasper:1810.00663:p4_2", "qasper:1902.00756:p12_0", "qasper:1712.03556:p1_1", "qasper:1902.00756:p5_5", "qasper:1812.10479:p13_1"]
- pplx_decider reader answer: Four
- pplx_decider frozen input hash: `bb3389a3177eda5c71db1ef6629918f93761f1b4c236fee2c6ffecc1b4f1428f`
- jina35 reader answer: six
- jina35 frozen input hash: `4bfec34cdc769b67909515c93d1710e6618d9242eb0bba2a374735a2227729f6`

No case is classified as factually correct or incorrect by this audit.

## qasper:1803.05223:ad1be65c4f0655ac5c902d17f05454c0d4c4a15d
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1803.05223` — MCScript: A Novel Dataset for Assessing Machine Comprehension Using Script Knowledge
- Question: what dataset statistics are provided?
- Reference status/types: `ANSWERABLE` / `ABSTRACTIVE`
- Reference forms: ["Distribution of category labels, number of answerable-not answerable questions, number of text-based and script-based questions, average text, question, and answer length, number of questions per text", "More than 2,100 texts were paired with 15 questions each, resulting in a total number of approx. 32,000 annotated questions. 13% of the questions are not answerable.  Out of the answerable questions, 10,160 could be answered from the text directly (text-based) and 3,914 questions required the use of commonsense knowledge (script-based).  The final dataset comprises 13,939 questions, 3,827 of which require commonsense knowledge (i.e. 27.4%)."]
- Evidence annotations: 2; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1812.07023:p2_1", "qasper:1908.04042:p2_0", "qasper:1906.03538:p12_0", "qasper:1811.09353:p13_0", "qasper:1907.08937:p10_0"]
- Jina selected IDs: ["qasper:1811.09353:p13_0", "qasper:2004.04315:p0_3", "qasper:1908.04042:p2_0", "qasper:2001.06888:p5_0", "qasper:1906.03538:p12_0"]
- pplx_decider reader answer: number of dialogues, turns, and words; statistics on sources of different content; dataset statistics
- pplx_decider frozen input hash: `1718725cbe523e67f4917582305c8d2703f8826cad38bfccd0e91b12d81cc807`
- jina35 reader answer: Dataset statistics are provided in Table TABREF34 and Table TABREF46.
- jina35 frozen input hash: `075734dcf82c1f2672442431416faa34c21ff24e5ec501a430b44009a7c5205b`

No case is classified as factually correct or incorrect by this audit.

## qasper:1909.00183:ee9b95d773e060dced08705db8d79a0a6ef353da
- Diagnostic label: `UNRESOLVED`
- Paper: `1909.00183` — Extracting information from free text through unsupervised graph-based clustering: an application to patient incident records
- Question: How are content clusters used to improve the prediction of incident severity?
- Reference status/types: `ANSWERABLE` / `ABSTRACTIVE`
- Reference forms: ["they are used as additional features in a supervised classification task"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1909.00183:p13_0", "qasper:1909.00183:p15_3", "qasper:1909.00183:p9_0", "qasper:1909.00183:p15_0", "qasper:1909.00183:p0_5"]
- Jina selected IDs: ["qasper:1909.00183:p0_5", "qasper:1909.00183:p15_3", "qasper:1909.00183:p9_0", "qasper:1909.00183:p11_3", "qasper:1909.00183:p13_0"]
- pplx_decider reader answer: Content clusters improve the prediction of incident severity by serving as additional features in a supervised classifier, enhancing performance when combined with free-text descriptions and other hand-coded features.
- pplx_decider frozen input hash: `1b0afcd4453ebde77b6bbab4258ccab0d8a5be97b3a9c07f7c7fd86783ea1e20`
- jina35 reader answer: Content clusters improve prediction by serving as additional features in supervised classification, enhancing performance when combined with text embeddings.
- jina35 frozen input hash: `3be5c46d0d7f241964fbe71464412fa37b2e523a54f50e2268e831d3ba9decca`

No case is classified as factually correct or incorrect by this audit.

## qasper:1710.09589:cb20aebfedad1a306e82966d6e9e979129fcd9f9
- Diagnostic label: `UNRESOLVED`
- Paper: `1710.09589` — ALL-IN-1: Short Text Classification with One Model for All Languages
- Question: what evaluation metrics were used?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["weighted F1-score"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1805.11535:p18_1", "qasper:1708.01464:p8_0", "qasper:1910.04601:p1_3", "qasper:1910.06036:p10_1", "qasper:1912.05066:p10_0"]
- Jina selected IDs: ["qasper:1708.01464:p8_0", "qasper:1805.08241:p4_3", "qasper:1805.11535:p18_1", "qasper:2002.12612:p7_1", "qasper:1912.05066:p10_0"]
- pplx_decider reader answer: BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, ROUGE-L
- pplx_decider frozen input hash: `3cacb4a88461ec74898f64e600166c203e75327de00e19f09493e20a23262684`
- jina35 reader answer: tokenized BLEU, METEOR, and two new metrics for over and under-translation
- jina35 frozen input hash: `cad533640934bebf1ead95efbf0148e0c5d2add9980271cd262846532a7419c2`

No case is classified as factually correct or incorrect by this audit.

## qasper:1909.07512:2e3265d83d2a595293ed458152d3ee76ad19e244
- Diagnostic label: `UNRESOLVED`
- Paper: `1909.07512` — Short-Text Classification Using Unsupervised Keyword Expansion
- Question: What news dataset was used?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["collection of headlines published by HuffPost BIBREF12 between 2012 and 2018"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1909.07512:p2_0", "qasper:1707.06806:p9_0", "qasper:1909.00088:p13_0", "qasper:1908.09246:p10_6", "qasper:1910.09295:p12_0"]
- Jina selected IDs: ["qasper:1707.06806:p9_0", "qasper:1909.07512:p2_0", "qasper:2004.01980:p15_0", "qasper:1910.01863:p0_5", "qasper:2003.03612:p2_5"]
- pplx_decider reader answer: News Category Dataset
- pplx_decider frozen input hash: `06245e55126f2657ad7302e4cb9da0fd93e0c5bad99020a77849a6d7abb76934`
- jina35 reader answer: The NowThisNews dataset
- jina35 frozen input hash: `c13f2104d8e2a4d54e56dbbbfeb1af0236625dd550eff30e3be36d3d99ada535`

No case is classified as factually correct or incorrect by this audit.

## qasper:2001.07820:92bb41cf7bd1f7886784796a8220ed5aa07bc49b
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `2001.07820` — Elephant in the Room: An Evaluation Framework for Assessing Adversarial Examples in NLP
- Question: What other factors affect the performance?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["architecture of the classifier", "input domain", "sentence length"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1906.10519:p17_4", "qasper:1912.00582:p17_0", "qasper:1807.09671:p8_10", "qasper:1908.08717:p16_0", "qasper:1807.09671:p8_0"]
- Jina selected IDs: ["qasper:1906.10519:p17_4", "qasper:1912.00582:p17_0", "qasper:1807.09671:p9_3", "qasper:1807.09671:p8_10", "qasper:1807.09671:p8_0"]
- pplx_decider reader answer: sentence length, perplexity, probability of the good sentence
- pplx_decider frozen input hash: `4eaa22d7107e5e87a6d6eb414aa77be12db8b24b46e2e3ec997e5c882adaae88`
- jina35 reader answer: sentence length, perplexity, probability of the good sentence
- jina35 frozen input hash: `3a40daeb6e214060eba7e2c4f5a7bd09ece331d3447580595b2396abe113c45e`

No case is classified as factually correct or incorrect by this audit.

## qasper:1704.04539:8fa7011e7beaa9fb4083bf7dd75d1216f9c7b2eb
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1704.04539` — Cross-lingual Abstract Meaning Representation Parsing
- Question: Do the authors test their annotation projection techniques on tasks other than AMR?
- Reference status/types: `ANSWERABLE` / `YES_NO`
- Reference forms: ["no"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1704.04539:p9_0", "qasper:1704.04539:p0_2", "qasper:1704.04539:p2_0", "qasper:1704.04539:p8_0", "qasper:1704.04539:p5_4"]
- Jina selected IDs: ["qasper:1704.04539:p9_0", "qasper:1704.04539:p2_0", "qasper:1704.04539:p5_0", "qasper:1704.04539:p5_4", "qasper:1704.04539:p0_2"]
- pplx_decider reader answer: Yes
- pplx_decider frozen input hash: `3bfa19ca625799c53968da80cd5fb75247d4cc2a82c0306b32f1f562e9c4c739`
- jina35 reader answer: Yes
- jina35 frozen input hash: `bc92e849672ae9dc00be568884e9db82fa4c6ab7ca9faa6db44f5f5a96e09045`

No case is classified as factually correct or incorrect by this audit.

## qasper:1711.06351:bd1a3c651ca2b27f283d3f36df507ed4eb24c2b0
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1711.06351` — Question Asking as Program Generation
- Question: Is it a neural model? How is it trained?
- Reference status/types: `ANSWERABLE` / `ABSTRACTIVE`
- Reference forms: ["No, it is a probabilistic model trained by finding feature weights through gradient ascent"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1805.06966:p7_0", "qasper:1809.04960:p6_0", "qasper:1707.00110:p1_6", "qasper:1711.11221:p5_0", "qasper:1808.07625:p10_0"]
- Jina selected IDs: ["qasper:1909.09270:p18_2", "qasper:1805.06966:p7_0", "qasper:1912.00582:p1_0", "qasper:1707.00110:p1_6", "qasper:1809.04960:p6_0"]
- pplx_decider reader answer: Yes, it is a neural model; trained to maximize log probability or negative log likelihood using stochastic gradient descent.
- pplx_decider frozen input hash: `8546519cf764e83248caa530a4928bf14633c91304af9b14a5e0f74e8c075f87`
- jina35 reader answer: Yes, it is a neural model. It is trained to maximize the log probability of user utterances or by minimizing the negative log likelihood using stochastic gradient descent.
- jina35 frozen input hash: `d9519cc236fe003cb804b87f4ae094c9e98d82375fb0d626bb1a6bc3b1fc0ae1`

No case is classified as factually correct or incorrect by this audit.

## qasper:2003.08553:fd0ef5a7b6f62d07776bf672579a99c67e61a568
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `2003.08553` — QnAMaker: Data to Bot in 2 Minutes
- Question: What experiments do the authors present to validate their system?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["evaluations are done by managed judges who understands the knowledge base and then judge user queries relevance to the QA pairs", "we measure our system's performance for datasets across various domains"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1910.11204:p0_7", "qasper:1909.08089:p10_0", "qasper:1906.10551:p11_4", "qasper:1804.08782:p8_0", "qasper:1909.07593:p0_8"]
- Jina selected IDs: ["qasper:1910.06748:p3_1", "qasper:1909.08089:p10_0", "qasper:1804.08782:p8_0", "qasper:1909.07593:p0_8", "qasper:1710.07960:p3_3"]
- pplx_decider reader answer: experiments on the two scientific paper datasets (arXiv and PubMed)
- pplx_decider frozen input hash: `6d41544e4ef0577c525d551cc6f9f54b4d50e74b84ce1321419f3402bc575fbd`
- jina35 reader answer: experiments on the two scientific paper datasets (arXiv and PubMed)
- jina35 frozen input hash: `04ce8af7ac035514e84ede226386c3d8f88b812dac85e0317c121a945b0b43c9`

No case is classified as factually correct or incorrect by this audit.

## qasper:2002.01207:2740e3d7d33173664c1c5ab292c7ec75ff6e0802
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `2002.01207` — Arabic Diacritic Recovery Using a Feature-Rich biLSTM Model
- Question: what datasets were used?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["WikiNews", "WikiNews test set BIBREF31", "a large collection of fully diacritized classical texts", "diacritized corpus that was used to train the RDI BIBREF7 diacritizer and the Farasa diacritizer BIBREF31", "large collection of fully diacritized classical texts (2.7M tokens) from a book publisher", "the diacritized corpus that was used to train the RDI BIBREF7 diacritizer"]
- Evidence annotations: 2; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1909.05246:p8_0", "qasper:1909.02560:p10_0", "qasper:1910.10781:p5_0", "qasper:1905.07471:p5_0", "qasper:1804.08139:p7_1"]
- Jina selected IDs: ["qasper:1909.05246:p8_0", "qasper:1809.00530:p6_5", "qasper:1611.02988:p2_0", "qasper:1804.08139:p7_1", "qasper:1912.00864:p9_0"]
- pplx_decider reader answer: DSTC2, M2M-sim-M, M2M-sim-R, SQuADv1.1, NewsQA, 16 review corpora datasets
- pplx_decider frozen input hash: `cf0437b3a15c9d54f4aa74b005d39bb41b149cdc84ae5bb2bbf2d364fbb4e05d`
- jina35 reader answer: DSTC2, M2M-sim-M, M2M-sim-R, IMDB, Yelp2014, Cell Phone, Baby, Affective Text, Fairy Tales, ISEAR, 16 review corpora datasets, two CQA datasets
- jina35 frozen input hash: `158bd388ad6a16f13867770d5c13ab54c48f6d73e811ae9944bfb93f69ef4dc7`

No case is classified as factually correct or incorrect by this audit.

## qasper:1911.13066:2c88b46c7e3a632cfa10b7574276d84ecec7a0af
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1911.13066` — A Multi-cascaded Deep Model for Bilingual SMS Classification
- Question: What is their baseline model?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["the model proposed in BIBREF3"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1701.03051:p7_0", "qasper:1905.06512:p1_1", "qasper:1805.00460:p2_16", "qasper:1811.00625:p8_0", "qasper:1909.08824:p7_0"]
- Jina selected IDs: ["qasper:1701.03051:p7_0", "qasper:1701.03051:p8_0", "qasper:1905.06512:p1_1", "qasper:1912.10435:p9_5", "qasper:1809.04960:p10_5"]
- pplx_decider reader answer: The baseline model uses Naive Bayes, SVM, and Maximum Entropy classifiers with feature vectors composed of Unigrams, Bigrams, Unigrams + Bigrams, or Unigrams + POS tags.
- pplx_decider frozen input hash: `b257cba7c8715f3fadc49081e31b35cf19e25a18cdacf702d3c7ef0a63fe792d`
- jina35 reader answer: Naive Bayes, SVM, and Maximum Entropy classifiers with feature vectors of Unigrams, Bigrams, Unigrams + Bigrams, or Unigrams + POS tags
- jina35 frozen input hash: `08d2df546c0abf3fd2e303129404535329a1118287c7818e7aa352de93a40e3c`

No case is classified as factually correct or incorrect by this audit.

## qasper:1709.05404:6608f171b3e0dcdcd51b3e0c697d6e5003ab5f02
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1709.05404` — Creating and Characterizing a Diverse Corpus of Sarcasm in Dialogue
- Question: What lexico-syntactic cues are used to retrieve sarcastic utterances?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["adjective and adverb patterns", "verb, subject, and object arguments", "verbal patterns"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1709.05404:p3_4", "qasper:1709.05404:p0_0", "qasper:1709.05404:p6_1", "qasper:1809.09795:p0_3", "qasper:1709.05404:p3_5"]
- Jina selected IDs: ["qasper:1709.05404:p3_4", "qasper:1709.05404:p0_0", "qasper:1709.05404:p6_1", "qasper:1809.09795:p0_3", "qasper:1709.05404:p3_0"]
- pplx_decider reader answer: rhetorical questions, hyperbole, intensifiers, elongated words, quotations, false politeness, negative evaluations, emoticons, tag questions, exclamations, interjections, all caps, typographic markers, emojis, hashtags
- pplx_decider frozen input hash: `bca057ebab059b52f41aafc780b6d3cf8ba1aa6ed31b973b422a130633935c1b`
- jina35 reader answer: rhetorical questions, hyperbole, intensifiers, elongated words, quotations, false politeness, negative evaluations, emoticons, tag questions, exclamations, interjections, all caps, hashtag, emojis
- jina35 frozen input hash: `1f7c7c7e69857418e039eec074ffa99146f8095a8cab08b5dc996b9ae11e7d2e`

No case is classified as factually correct or incorrect by this audit.

## qasper:1809.04686:545ff2f76913866304bfacdb4cc10d31dbbd2f37
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1809.04686` — Zero-Shot Cross-lingual Classification Using Multilingual Neural Machine Translation
- Question: What data were they used to train the multilingual encoder?
- Reference status/types: `ANSWERABLE` / `ABSTRACTIVE`
- Reference forms: ["WMT 2014 En-Fr parallel corpus"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:2002.06675:p16_1", "qasper:1912.01214:p2_1", "qasper:1911.03597:p6_0", "qasper:1807.00868:p1_7", "qasper:1911.03597:p7_1"]
- Jina selected IDs: ["qasper:2002.06675:p16_1", "qasper:1911.03597:p6_0", "qasper:1807.00868:p1_7", "qasper:1708.01464:p2_1", "qasper:1809.04686:p5_2"]
- pplx_decider reader answer: MultiUN and OpenSubtitles corpora
- pplx_decider frozen input hash: `1ef81f669cb206a70ff07d96d0c116d14e0271e5cc532efd2e7c76981b2a06b4`
- jina35 reader answer: MultiUN and OpenSubtitles corpora
- jina35 frozen input hash: `f2c9160449235614375eb3ab7c88399d7dd6d5ef9577e2183bd4b3597bbe6797`

No case is classified as factually correct or incorrect by this audit.

## qasper:1803.07771:2d307b43746be9cedf897adac06d524419b0720b
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1803.07771` — $\rho$-hot Lexicon Embedding-based Two-level LSTM for Sentiment Analysis
- Question: How long are the datasets?
- Reference status/types: `ANSWERABLE` / `ABSTRACTIVE`
- Reference forms: ["Travel dataset contains 4100 raw samples, 11291 clauses, Hotel dataset contains 3825 raw samples, 11264 clauses, and the Mobile dataset contains 3483 raw samples and 8118 clauses"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1909.08306:p2_0", "qasper:1908.01294:p7_4", "qasper:1901.03859:p8_2", "qasper:1912.07976:p17_1", "qasper:1705.04153:p8_5"]
- Jina selected IDs: ["qasper:1909.08306:p2_0", "qasper:1908.01294:p7_4", "qasper:1901.03859:p8_2", "qasper:1811.09353:p13_0", "qasper:1705.04153:p8_5"]
- pplx_decider reader answer: INSUFFICIENT_EVIDENCE
- pplx_decider frozen input hash: `b64b48a7392a773495f03bb1829529f34c5b94385a604bf961aecc27b613b683`
- jina35 reader answer: INSUFFICIENT_EVIDENCE
- jina35 frozen input hash: `a59bbbd6d05e18ff80fd76a5024a6c1af76558f342784c8efb88d43924320182`

No case is classified as factually correct or incorrect by this audit.

## qasper:1909.03405:bdc91d1283a82226aeeb7a2f79dbbc57d3e84a1a
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1909.03405` — Symmetric Regularization based BERT for Pair-wise Semantic Reasoning
- Question: How much is performance improved on NLI?
- Reference status/types: `ANSWERABLE` / `MIXED_ANNOTATED`
- Reference forms: ["The average score improved by 1.4 points over the previous best result.", "improvement on the RTE dataset is significant, i.e., 4% absolute gain over the BERTBase"]
- Evidence annotations: 2; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1908.10084:p6_2", "qasper:1804.11346:p0_2", "qasper:1907.04380:p0_3", "qasper:1804.11346:p1_0", "qasper:1810.09774:p5_6"]
- Jina selected IDs: ["qasper:1908.10084:p6_2", "qasper:1901.02222:p0_4", "qasper:1810.09774:p5_6", "qasper:1911.03894:p14_1", "qasper:1809.02279:p8_0"]
- pplx_decider reader answer: 1-2 points
- pplx_decider frozen input hash: `e9cc1e61e83a08e7a8115d6aa945f0e162ae7f1ff6a1b1e121eaf20cda8d858b`
- jina35 reader answer: 1-2 points
- jina35 frozen input hash: `64c031347c65e77a8850d75a563361feb7986f14cc06e1c11cdc93b520ed34e3`

No case is classified as factually correct or incorrect by this audit.

## qasper:1704.05572:e215fa142102f7f9eeda9c9eb8d2aeff7f2a33ed
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1704.05572` — Answering Complex Questions Using Open Information Extraction
- Question: What method was used to generate the OpenIE extractions?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["for each multiple-choice question $(q,A) \\in Q_\\mathit {tr}$ and each choice $a \\in A$ , we use all non-stopword tokens in $q$ and $a$ as an ElasticSearch query against S", "take the top 200 hits, run Open IE v4, and aggregate the resulting tuples over all $a \\in A$ and over all questions in $Q_\\mathit {tr}$"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1905.07471:p9_0", "qasper:1802.05574:p2_0", "qasper:1910.06036:p3_0", "qasper:1905.13413:p0_2", "qasper:1802.05574:p2_2"]
- Jina selected IDs: ["qasper:1905.07471:p9_0", "qasper:1905.13413:p0_0", "qasper:1905.07471:p0_0", "qasper:1905.13413:p0_2", "qasper:1905.07471:p14_0"]
- pplx_decider reader answer: sequence to sequence model with attention
- pplx_decider frozen input hash: `1c2198bc08c5036f232830faff191b7ca1207159272c8ca2f97e10291af20fc3`
- jina35 reader answer: sequence to sequence model with attention
- jina35 frozen input hash: `0123daee65e74ae6d3fde7abd450bf2f199fb94c3c799448e9005c94a27c0a5d`

No case is classified as factually correct or incorrect by this audit.

## qasper:1901.04085:25c4fa78299481788a19d0c25ae07dfd8cb6315c
- Diagnostic label: `UNRESOLVED`
- Paper: `1901.04085` — Passage Re-ranking with BERT
- Question: What is the TREC-CAR dataset?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["in this dataset, the input query is the concatenation of a Wikipedia article title with the title of one of its section. The relevant passages are the paragraphs within that section"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1901.04085:p3_0", "qasper:1901.04085:p1_0", "qasper:1901.04085:p3_1", "qasper:1901.04085:p3_2", "qasper:1705.04153:p8_8"]
- Jina selected IDs: ["qasper:1901.04085:p3_0", "qasper:1901.04085:p3_2", "qasper:1901.04085:p3_1", "qasper:1901.04085:p1_0", "qasper:1812.06705:p7_5"]
- pplx_decider reader answer: A dataset where input queries are Wikipedia article titles concatenated with section titles, and relevant passages are paragraphs within those sections.
- pplx_decider frozen input hash: `d69999296e0e437ae89af27863b5c4b9fea4ed498c1e2f7ed2ea65207bc4bf24`
- jina35 reader answer: A dataset where input queries are Wikipedia article titles concatenated with section titles, and relevant passages are paragraphs within those sections.
- jina35 frozen input hash: `bb93e659a66fb30ddc563e2cab75cd159f0a7b156a695581a20e57ce27cf3d17`

No case is classified as factually correct or incorrect by this audit.

## qasper:1701.03051:6b55b558ed581759425ede5d3a6fcdf44b8082ac
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1701.03051` — Efficient Twitter Sentiment Classification using Subjective Distant Supervision
- Question: What previously proposed methods is this method compared against?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["Maximum Entropy classifiers", "Naive Bayes", "SVM"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1909.07575:p13_0", "qasper:1709.06671:p0_10", "qasper:1810.05334:p4_5", "qasper:1908.11860:p18_0", "qasper:1701.03214:p11_0"]
- Jina selected IDs: ["qasper:1706.00139:p12_0", "qasper:1709.06671:p0_10", "qasper:1709.06671:p8_3", "qasper:1808.03738:p7_2", "qasper:1909.08824:p7_0"]
- pplx_decider reader answer: fine tuning and multi domain methods
- pplx_decider frozen input hash: `131403bbfc79327546549f8cb82fd789229cae4a08f3878e50f1cdc0260014c1`
- jina35 reader answer: 1TON and 1TON+
- jina35 frozen input hash: `6dcc93b6710580ff524e8ae2e30009283f0eb6a2726e9edeadc686ba171c6202`

No case is classified as factually correct or incorrect by this audit.

## qasper:1909.09268:1244cf6d75e3aa6d605a0f4b141c015923a3f2e7
- Diagnostic label: `UNRESOLVED`
- Paper: `1909.09268` — Towards Neural Language Evaluators
- Question: What is the criteria for a good metric?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["The first one is that it should be highly correlated with human judgement of similarity. The second one is that it should be able to distinguish sentences which are in logical contradiction, logically unrelated or in logical agreement. The third one is that a robust evaluator should also be able to identify unintelligible sentences. The last criteria is that a good evaluation metric should not give high scores to semantically distant sentences and low scores to semantically related sentences."]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1909.09268:p10_0", "qasper:2003.04866:p5_2", "qasper:1906.01910:p8_0", "qasper:2003.08529:p8_1", "qasper:2001.07820:p0_4"]
- Jina selected IDs: ["qasper:1909.09268:p10_0", "qasper:1708.09609:p4_0", "qasper:2002.05058:p7_1", "qasper:2003.04866:p5_2", "qasper:2001.07820:p0_1"]
- pplx_decider reader answer: Highly correlated with human judgement, distinguishes logical relations, identifies unintelligible sentences, and avoids scoring semantically distant sentences highly or related ones low.
- pplx_decider frozen input hash: `f02ac17568862b330171b1cabed1b30b8052f50d5e57cb7a99cc89c1bb2a4333`
- jina35 reader answer: Highly correlated with human judgement, distinguishes logical relations, identifies unintelligible sentences, and avoids rewarding semantically distant sentences.
- jina35 frozen input hash: `f92ba91975ebb7f0cd484948eb27307fba49550779261655a39632bb1d2db1df`

No case is classified as factually correct or incorrect by this audit.

## qasper:1909.09270:869aaf397c9b4da7ab52d6dd0961887ae08da9ae
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1909.09270` — Named Entity Recognition with Partially Annotated Training Data
- Question: Which languages are evaluated?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["Amharic", "Arabic", "Bengali", "English, German, Spanish, Dutch", "Hindi", "Somali"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1911.02116:p4_1", "qasper:2002.07306:p8_0", "qasper:1604.05372:p0_4", "qasper:1911.10049:p5_0", "qasper:1910.05456:p9_2"]
- Jina selected IDs: ["qasper:2002.07306:p8_0", "qasper:1911.02116:p4_1", "qasper:1911.10049:p5_0", "qasper:1910.07481:p4_2", "qasper:1910.11768:p4_1"]
- pplx_decider reader answer: English, French, German, Russian, Chinese, Swahili, Urdu, Arabic, Hindi, Vietnamese, Basque, Hungarian, Italian, Navajo, Turkish, Quechua, and others including romanized Hindi and traditional Chinese.
- pplx_decider frozen input hash: `e6e6c4ef262fd8b3172711f22d5383ad16c143c92323d3482be3ff9793cf8957`
- jina35 reader answer: French, Russian, Arabic, Chinese, Hindi, Vietnamese, English, German, Swahili, Urdu, Spanish, Dutch, Korean, Japanese
- jina35 frozen input hash: `be90667cadf18dbbd17b092819e78bed99da2ad2729e04257355518fdfc6d1e7`

No case is classified as factually correct or incorrect by this audit.

## qasper:1905.07464:3752bbc5367973ab5b839ded08c57f51336b5c3d
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1905.07464` — A Multi-Task Learning Framework for Extracting Drugs and Their Interactions from Drug Labels
- Question: What training data did they use?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["NLM-180", "Training-22"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:2002.11268:p7_0", "qasper:1910.11949:p5_0", "qasper:1710.06536:p14_16", "qasper:1810.09774:p3_2", "qasper:1710.06536:p14_13"]
- Jina selected IDs: ["qasper:2002.11268:p7_0", "qasper:1910.11949:p5_0", "qasper:1903.11283:p1_6", "qasper:1710.06536:p14_16", "qasper:2004.01670:p2_0"]
- pplx_decider reader answer: A dataset that maps pictures with questions, and an open-domain conversation dataset
- pplx_decider frozen input hash: `592d93fb43bfcf151c7c5ba2a72c88a68bbfa868d63f43f9b8eaa1bd1afc971b`
- jina35 reader answer: A dataset that maps pictures with questions, and an open-domain conversation dataset
- jina35 frozen input hash: `e491d58db40f950ae8b7eb53bc435efd6c84b1a3b0a7a979c536bbc0a54c84c1`

No case is classified as factually correct or incorrect by this audit.

## qasper:1603.07044:942eb1f7b243cdcfd47f176bcc71de2ef48a17c4
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1603.07044` — Recurrent Neural Network Encoder with Attention for Community Question Answering
- Question: Did they experimnet in other languages?
- Reference status/types: `ANSWERABLE` / `YES_NO`
- Reference forms: ["yes"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1708.09157:p6_0", "qasper:1906.00378:p10_0", "qasper:1805.03122:p4_0", "qasper:1906.10519:p12_0", "qasper:1909.00326:p7_0"]
- Jina selected IDs: ["qasper:1708.09157:p6_0", "qasper:1805.03122:p4_0", "qasper:1906.00378:p10_0", "qasper:1909.00326:p7_0", "qasper:1708.01464:p9_2"]
- pplx_decider reader answer: Yes
- pplx_decider frozen input hash: `38e482bac96f6c2dc3867c3bfd7c32f6f996cd5083c670b603c157fffb57150e`
- jina35 reader answer: Yes
- jina35 frozen input hash: `f3d56622b85fb3276a21a5407441760531d52ed77605c478c9a2de7bcd2e0004`

No case is classified as factually correct or incorrect by this audit.

## qasper:1708.08615:95d8368b1055d97250df38d1e8c4a2b283d2b57e
- Diagnostic label: `UNRESOLVED`
- Paper: `1708.08615` — Comparing Human and Machine Errors in Conversational Speech Transcription
- Question: what standard speech transcription pipeline was used?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["pipeline that is used at Microsoft for production data"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1908.08717:p6_0", "qasper:1708.08615:p1_1", "qasper:1708.08615:p0_1", "qasper:1912.01772:p5_0", "qasper:2003.01472:p1_0"]
- Jina selected IDs: ["qasper:1708.08615:p1_1", "qasper:1708.08615:p0_1", "qasper:2003.01472:p1_0", "qasper:1908.08717:p6_0", "qasper:1912.01772:p5_0"]
- pplx_decider reader answer: two-pass transcription with error correction
- pplx_decider frozen input hash: `09c9e432146a3994ea5f1a6cc4618fe19b8fdcf7527e54fb31fc5712096a4d9b`
- jina35 reader answer: two-pass transcription
- jina35 frozen input hash: `d2a4f00ecc22d88cc53be763d972fd592a154e80b1cb643e1a1dd4e559870447`

No case is classified as factually correct or incorrect by this audit.

## qasper:1810.08732:7f2fd7ab968de720082133c42c2052d351589a67
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1810.08732` — Named Entity Recognition on Twitter for Turkish using Semi-supervised Learning with Word Embeddings
- Question: What type and size of word embeddings were used?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["200 as the dimension of the obtained word vectors", "word2vec"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1901.03866:p4_1", "qasper:1908.05434:p14_0", "qasper:1710.06536:p14_24", "qasper:1911.00547:p6_2", "qasper:1912.02481:p11_1"]
- Jina selected IDs: ["qasper:1901.03866:p4_1", "qasper:1611.03599:p8_0", "qasper:1704.02686:p9_3", "qasper:1611.04361:p5_0", "qasper:1912.02481:p11_1"]
- pplx_decider reader answer: 300-dimensional GloVe embeddings
- pplx_decider frozen input hash: `65da59325325d95a60c9bb19c968049dfe2887461d427858b2d972625ef987e7`
- jina35 reader answer: 300-dimensional GloVe word embeddings
- jina35 frozen input hash: `42f7d32a3f3ae901904dcc62174508969dc37736794cf31eb3f02ca7daf72d0c`

No case is classified as factually correct or incorrect by this audit.

## qasper:1811.00383:a313e98994fc039a82aa2447c411dda92c65a470
- Diagnostic label: `UNRESOLVED`
- Paper: `1811.00383` — Addressing word-order Divergence in Multilingual Neural Machine Translation for extremely Low Resource Languages
- Question: How do they match words before reordering them?
- Reference status/types: `AMBIGUOUS_REFERENCE` / `AMBIGUOUS_REFERENCE`
- Reference forms: ["CFILT-preorder system"]
- Evidence annotations: 2; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1608.01084:p2_3", "qasper:1811.00383:p8_2", "qasper:1808.03738:p2_2", "qasper:1608.01084:p8_5", "qasper:1803.02839:p3_21"]
- Jina selected IDs: ["qasper:1608.01084:p0_0", "qasper:1608.01084:p2_3", "qasper:1811.00383:p8_2", "qasper:1808.03738:p2_2", "qasper:1903.12495:p6_1"]
- pplx_decider reader answer: They match words based on their relationships in the dependency parse tree, specifically head, sibling, and child words.
- pplx_decider frozen input hash: `f17aa6838293540cd09e59ee2d75d8af4b203290a72972ba1a2d0001ff6fcc16`
- jina35 reader answer: They match words based on their relations in the dependency parse tree, focusing on head, sibling, and child words.
- jina35 frozen input hash: `9e8a61a5e1a4d8a6410bb32184ff972be223b3f280f447a2af042c870376ae97`

No case is classified as factually correct or incorrect by this audit.

## qasper:1811.11365:cf15c4652e23829d8fb4cf2a25e64408c18734c1
- Diagnostic label: `UNRESOLVED`
- Paper: `1811.11365` — Unsupervised Multi-modal Neural Machine Translation
- Question: Why is this work different from text-only UNMT?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["the image can play the role of a pivot “language\" to bridge the two languages without paralleled corpus"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1811.11365:p0_4", "qasper:1811.11365:p1_1", "qasper:1811.11365:p10_4", "qasper:1811.11365:p13_0", "qasper:1811.11365:p12_5"]
- Jina selected IDs: ["qasper:1811.11365:p0_4", "qasper:1909.01383:p19_2", "qasper:1811.11365:p10_4", "qasper:1811.11365:p12_5", "qasper:1909.00279:p1_1"]
- pplx_decider reader answer: It uses multi-modal attention and images as additional information to improve translation, unlike text-only UNMT which relies solely on text data.
- pplx_decider frozen input hash: `a79b32d1941ce051d9116aa78f02a2fc3c62776e8295173b39ac9de0302a28ea`
- jina35 reader answer: It uses images as a pivot language to align languages, unlike text-only UNMT which is fundamentally ill-posed due to ambiguous associations.
- jina35 frozen input hash: `ba9e7edfdb1dde8f41e59e12438de71a2b592a91b2c3a3dec7850c8c14dc9470`

No case is classified as factually correct or incorrect by this audit.

## qasper:1908.01060:e6c163f80a11bd057bbd0b6e1451ac82edddc78d
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1908.01060` — Multilingual Speech Recognition with Corpus Relatedness Sampling
- Question: Do they test their approach on large-resource tasks?
- Reference status/types: `ANSWERABLE` / `YES_NO`
- Reference forms: ["yes"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1805.07513:p6_2", "qasper:1601.01705:p6_0", "qasper:1812.10860:p2_0", "qasper:1909.06434:p6_3", "qasper:1708.01464:p12_0"]
- Jina selected IDs: ["qasper:1708.01464:p5_2", "qasper:1909.04493:p0_9", "qasper:1708.01464:p12_0", "qasper:1809.04686:p0_1", "qasper:1912.13337:p0_4"]
- pplx_decider reader answer: Yes
- pplx_decider frozen input hash: `1b548c66c925a5797d20c2a70b817f892ac82702b3d775ded068e5cf8e01e1d3`
- jina35 reader answer: Yes
- jina35 frozen input hash: `5c633492b261227d23cc3766eaa97f08d863777510c6eecf295ce7cecd9f1db1`

No case is classified as factually correct or incorrect by this audit.

## qasper:1710.07395:569ad21441e99ae782d325d5f5e1ac19e08d5e76
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1710.07395` — Detecting Online Hate Speech Using Context Aware Models
- Question: What context do they use?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["screen name of the user", "title of the news article"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1809.01060:p1_9", "qasper:2001.07786:p15_0", "qasper:1809.01060:p1_7", "qasper:1902.09393:p9_4", "qasper:1912.09713:p26_8"]
- Jina selected IDs: ["qasper:2001.07786:p15_0", "qasper:1906.07662:p8_0", "qasper:1809.01060:p1_9", "qasper:1809.01060:p1_7", "qasper:1912.09152:p3_9"]
- pplx_decider reader answer: natural context
- pplx_decider frozen input hash: `e1f9f5b2b468bbe3e582ac3a2aef5a70a07a31e8d7bc4dec7c196dedc821bee3`
- jina35 reader answer: natural context
- jina35 frozen input hash: `21ebc0b9fca0f54e0590fe9bafb2abcfa243d22b0275ff58dc3ba7604bc5e591`

No case is classified as factually correct or incorrect by this audit.

## qasper:1912.00871:095888f6e10080a958d9cd3f779a339498f3a109
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1912.00871` — Solving Arithmetic Word Problems Automatically Using Transformer and Unambiguous Representations
- Question: What datasets do they use?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["AI2 BIBREF2", "CC BIBREF19", "IL BIBREF4", "MAWPS BIBREF20"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:2001.05493:p9_0", "qasper:1710.06536:p14_13", "qasper:1910.10781:p5_0", "qasper:1705.04153:p8_5", "qasper:1609.01962:p3_26"]
- Jina selected IDs: ["qasper:2001.05493:p9_0", "qasper:1912.08320:p35_81", "qasper:1909.02560:p10_0", "qasper:1609.01962:p3_26", "qasper:1910.10781:p5_0"]
- pplx_decider reader answer: TRAC 2018 Dataset and Kaggle Dataset
- pplx_decider frozen input hash: `9f168a25acabd902dc09f4fb7a1a1e487524ec93190dcf38ff2651cf092a0e7d`
- jina35 reader answer: TRAC 2018 Dataset and Kaggle Dataset
- jina35 frozen input hash: `cd04cabba50bccbb5d907bc500d9ada0d80dac15135bc8c157b4cf1dfce1f834`

No case is classified as factually correct or incorrect by this audit.

## qasper:1910.13215:537a786794604ecc473fb3ef6222e0c3cb81f772
- Diagnostic label: `EVIDENCE_MISSING`
- Paper: `1910.13215` — Transformer-based Cascaded Multimodal Speech Translation
- Question: What dataset was used in this work?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["How2"]
- Evidence annotations: 1; valid Top-20 support present: `False`
- PPLX selected IDs: ["qasper:1808.05077:p2_0", "qasper:2001.05493:p9_0", "qasper:1912.08320:p35_81", "qasper:1909.02560:p10_0", "qasper:1910.10781:p8_0"]
- Jina selected IDs: ["qasper:1808.05077:p2_0", "qasper:1905.07471:p5_0", "qasper:2004.04315:p0_3", "qasper:2001.05493:p9_0", "qasper:1802.05574:p3_0"]
- pplx_decider reader answer: Persian movie reviews from 2014 to 2016
- pplx_decider frozen input hash: `79921d4d69b992614d49a0f5f07d0068de52561aec1449d2b6ca871981ad7962`
- jina35 reader answer: Persian movie reviews from 2014 to 2016
- jina35 frozen input hash: `0fc494c5aab5d1fa99ccff609e71541656887b12dd7f89385ae237553089bf3e`

No case is classified as factually correct or incorrect by this audit.

## qasper:1707.06945:6b8a3100895f2192e08973006474428319dc298e
- Diagnostic label: `UNRESOLVED`
- Paper: `1707.06945` — Cross-Lingual Induction and Transfer of Verb Classes Based on Word Vector Space Specialisation
- Question: What clustering algorithm is used on top of the VerbNet-specialized representations?
- Reference status/types: `ANSWERABLE` / `EXTRACTIVE`
- Reference forms: ["MNCut spectral clustering algorithm BIBREF58"]
- Evidence annotations: 1; valid Top-20 support present: `True`
- PPLX selected IDs: ["qasper:1707.06945:p2_0", "qasper:1701.09123:p8_16", "qasper:1701.09123:p8_2", "qasper:1908.10422:p4_2", "qasper:1701.09123:p8_19"]
- Jina selected IDs: ["qasper:1707.06945:p2_0", "qasper:1701.00185:p8_0", "qasper:1701.09123:p1_4", "qasper:1907.04152:p6_0", "qasper:2003.06651:p7_10"]
- pplx_decider reader answer: MNCut spectral clustering
- pplx_decider frozen input hash: `5b91cfff3a2a090a13b99cb5ac617b725ab58341cf9ff55f108429032c39cb60`
- jina35 reader answer: MNCut spectral clustering
- jina35 frozen input hash: `0e1918702df643bbc1501ca65fd803693af95dd6c8d936b97aaa8a6e0a56a548`

No case is classified as factually correct or incorrect by this audit.
