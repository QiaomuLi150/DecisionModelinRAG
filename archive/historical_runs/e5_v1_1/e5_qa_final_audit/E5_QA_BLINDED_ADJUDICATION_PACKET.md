# E5-QA Blinded Adjudication Packet

Frozen offline packet for a later independent review. No adjudication was performed here. Arm labels are deterministic opaque A/B labels; PPLX/Jina names, selector scores and E5-A success labels are withheld.

Cases: 100; packet seed: 731.

## Case 001
- Case ID: `case-001`
- Question: Which clinically validated survey tools are used?
- Reference forms for adjudication: ["DOSPERT, BSSS and VIAS"]
### Arm A
- Answer: Domain-Specific Risk-Taking (DOSPERT) Scale, Berlin Social Support Scales (BSSS), Values In Action Scale (VIAS), NEO Five-Factor Inventory, Positive and Negative Syndrome Scale, Montgomery-Asperg Depression Rating Scale, Young Mania Rating Scale
- Selected passage IDs: ["qasper:2003.07433:p3_0", "qasper:2003.07433:p2_1", "qasper:1809.05752:p8_1", "qasper:2003.07433:p5_0", "qasper:2003.07433:p0_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: INSUFFICIENT_EVIDENCE
- Selected passage IDs: ["qasper:2003.07433:p3_0", "qasper:1809.05752:p8_1", "qasper:2003.07433:p14_0", "qasper:2003.07433:p0_2", "qasper:2003.07433:p2_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 002
- Case ID: `case-002`
- Question: How is diversity measured?
- Reference forms for adjudication: ["diversity score as the ratio of observed number versus optimal number", "we look at language constructions used and compute the corresponding diversity score as the ratio of observed number versus optimal number"]
### Arm A
- Answer: Diversity is measured by computing the ratio of observed to optimal number of language constructions, standard deviation of embedding vectors with geometric mean, Self-BLEU scores, or Average Output Probability Distribution.
- Selected passage IDs: ["qasper:1912.08960:p6_1", "qasper:2003.08529:p3_5", "qasper:1911.03350:p7_0", "qasper:2001.05467:p0_2", "qasper:1909.00015:p14_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: By calculating the standard deviation of embedding vectors along each dimension and taking the geometric mean of all calculated values.
- Selected passage IDs: ["qasper:2003.08529:p3_5", "qasper:1911.03350:p7_0", "qasper:2001.05467:p13_0", "qasper:2003.08529:p0_3", "qasper:2003.08529:p8_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 003
- Case ID: `case-003`
- Question: What is the agreement score of their annotated dataset?
- Reference forms for adjudication: ["Relevance and Subject categorizations are annotated at a percent agreement of $0.71$ and $0.70$, their agreement scores are only fair, at $\\alpha =0.27$ and $\\alpha =0.29$", "Stance and Sentiment, which carry more categories than the former two, is $0.54$ for both. Their agreement scores are also fair, at $\\alpha =0.35$ and $\\alpha =0.34$", "The Negative category yields a mutual F-score of $0.42$, which is higher than the more frequently annotated categories Neutral ($0.23$) and Not clear ($0.31$).", "This holds for the Relevant category ($0.81$), the Vaccine category ($0.79$) and the Positive category ($0.64$)"]
### Arm A
- Answer: INSUFFICIENT_EVIDENCE
- Selected passage IDs: ["qasper:2004.03034:p2_7", "qasper:1909.00338:p3_7", "qasper:1708.09609:p2_12", "qasper:2003.04642:p10_1", "qasper:1911.03243:p9_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: 0.82
- Selected passage IDs: ["qasper:2003.04642:p10_1", "qasper:1608.08738:p5_1", "qasper:2002.02427:p1_1", "qasper:1808.06834:p4_2", "qasper:1612.05310:p4_3"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 004
- Case ID: `case-004`
- Question: What was the baseline for this task?
- Reference forms for adjudication: ["FLC task generates spans and selects one of the 18 techniques randomly", "SLC task is a very simple logistic regression classifier", "The baseline system for the SLC task is a very simple logistic regression classifier with default parameters. The baseline for the FLC task generates spans and selects one of the 18 techniques randomly."]
### Arm A
- Answer: INSUFFICIENT_EVIDENCE
- Selected passage IDs: ["qasper:1703.10344:p11_1", "qasper:1911.00841:p12_0", "qasper:1703.10344:p5_19", "qasper:1910.09982:p28_0", "qasper:1707.03764:p6_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: a very simple logistic regression classifier with default parameters, where we represent the input instances with a single feature: the length of the sentence
- Selected passage IDs: ["qasper:1910.09982:p28_0", "qasper:2003.10816:p11_0", "qasper:1910.09982:p28_1", "qasper:1911.00841:p12_0", "qasper:1707.03764:p6_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 005
- Case ID: `case-005`
- Question: How do they ensure the generated questions are unanswerable?
- Reference forms for adjudication: ["learn to ask unanswerable questions by editing answerable ones with word exchanges, negations, etc"]
### Arm A
- Answer: They ensure unanswerability by generating questions that cannot be answered by the given paragraph, while maintaining relevance to the original question and paragraph, and asking for the same type of answer as the original.
- Selected passage IDs: ["qasper:1906.06045:p0_4", "qasper:1906.06045:p2_0", "qasper:1906.06045:p0_1", "qasper:1906.06045:p0_2", "qasper:1906.06045:p7_4"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: They ensure the generated questions are unanswerable by editing answerable questions and conditioning on the corresponding paragraph that contains the answer.
- Selected passage IDs: ["qasper:1906.06045:p0_1", "qasper:1906.06045:p7_8", "qasper:1906.06045:p7_4", "qasper:1906.06045:p2_1", "qasper:1906.06045:p2_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 006
- Case ID: `case-006`
- Question: What Doc2Vec architectures other than PV-DBOW have been tried?
- Reference forms for adjudication: ["PV-DM"]
### Arm A
- Answer: PV-DM
- Selected passage IDs: ["qasper:1911.11698:p0_2", "qasper:1909.00183:p4_2", "qasper:1911.11698:p3_0", "qasper:1911.11698:p15_0", "qasper:1903.12495:p2_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: PV-DM
- Selected passage IDs: ["qasper:1901.01010:p9_3", "qasper:1911.11698:p6_0", "qasper:1911.11698:p19_0", "qasper:1706.09673:p2_1", "qasper:1911.11698:p3_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 007
- Case ID: `case-007`
- Question: How big is the new Nepali NER dataset?
- Reference forms for adjudication: ["3606 sentences", "Dataset contains 3606 total sentences and 79087 total entities."]
### Arm A
- Answer: INSUFFICIENT_EVIDENCE
- Selected passage IDs: ["qasper:1908.05828:p16_2", "qasper:1908.05828:p8_0", "qasper:1908.05828:p9_0", "qasper:1908.05828:p0_3", "qasper:1908.05828:p0_4"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: INSUFFICIENT_EVIDENCE
- Selected passage IDs: ["qasper:1908.05828:p8_0", "qasper:1908.05828:p16_2", "qasper:1908.05828:p0_3", "qasper:1908.05828:p0_1", "qasper:1908.05828:p0_4"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 008
- Case ID: `case-008`
- Question: What is te core component for KBQA?
- Reference forms for adjudication: ["answer questions by obtaining information from KB tuples", "hierarchical matching between questions and relations with residual learning"]
### Arm A
- Answer: relation detection
- Selected passage IDs: ["qasper:1704.06194:p0_3", "qasper:1704.06194:p15_0", "qasper:1707.04662:p1_3", "qasper:1604.00727:p2_0", "qasper:1902.09087:p0_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: relation detection
- Selected passage IDs: ["qasper:1704.06194:p14_2", "qasper:1704.06194:p0_3", "qasper:1704.06194:p0_0", "qasper:2003.08897:p4_0", "qasper:1704.06194:p15_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 009
- Case ID: `case-009`
- Question: What automatic and human evaluation metrics are used to compare SPNet to its counterparts?
- Reference forms for adjudication: ["ROUGE and CIC", "relevance, conciseness and readability on a 1 to 5 scale, and rank the summary pair"]
### Arm A
- Answer: ROUGE, CIC, relevance, readability, conciseness
- Selected passage IDs: ["qasper:1910.00825:p14_0", "qasper:1910.00825:p11_1", "qasper:1910.00825:p0_2", "qasper:1811.05085:p15_0", "qasper:1910.00825:p12_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: ROUGE, CIC, relevance, conciseness, readability
- Selected passage IDs: ["qasper:1910.00825:p14_0", "qasper:1910.00825:p11_1", "qasper:1607.06025:p1_4", "qasper:1910.00825:p12_0", "qasper:1910.00825:p12_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 010
- Case ID: `case-010`
- Question: How many followers did they analyze?
- Reference forms for adjudication: ["51,104"]
### Arm A
- Answer: 55 million
- Selected passage IDs: ["qasper:1909.10012:p3_2", "qasper:1909.10012:p2_1", "qasper:2001.08845:p16_0", "qasper:1908.00153:p10_4", "qasper:1907.01468:p12_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: 55,830,844
- Selected passage IDs: ["qasper:1909.10012:p2_1", "qasper:1909.10012:p3_2", "qasper:1909.10012:p6_0", "qasper:2001.05970:p2_0", "qasper:1909.10012:p17_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 011
- Case ID: `case-011`
- Question: Which method do they use for word segmentation?
- Reference forms for adjudication: ["unsupervised word segmentation method latticelm"]
### Arm A
- Answer: morphological word segmentation
- Selected passage IDs: ["qasper:2001.01589:p0_2", "qasper:2001.01589:p16_1", "qasper:1908.08419:p0_3", "qasper:1711.04457:p5_3", "qasper:1602.01208:p5_9"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: Maximum Matching
- Selected passage IDs: ["qasper:2001.01589:p1_0", "qasper:1906.07662:p6_0", "qasper:1911.03270:p12_0", "qasper:1708.00214:p5_0", "qasper:1906.07662:p8_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 012
- Case ID: `case-012`
- Question: What are 10 other phenotypes that are annotated?
- Reference forms for adjudication: ["Adv. Heart Disease, Adv. Lung Disease, Alcohol Abuse, Chronic Neurologic Dystrophies, Dementia, Depression, Developmental Delay, Obesity, Psychiatric disorders and Substance Abuse"]
### Arm A
- Answer: INSUFFICIENT_EVIDENCE
- Selected passage IDs: ["qasper:2003.03044:p2_1", "qasper:2003.03044:p1_1", "qasper:2003.03044:p1_3", "qasper:2003.03044:p9_0", "qasper:2003.03044:p9_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: INSUFFICIENT_EVIDENCE
- Selected passage IDs: ["qasper:2003.03044:p2_1", "qasper:2003.03044:p9_0", "qasper:2003.03044:p1_1", "qasper:1906.03538:p2_11", "qasper:1912.09152:p12_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 013
- Case ID: `case-013`
- Question: How are aspects identified in aspect extraction?
- Reference forms for adjudication: ["apply an ensemble of deep learning and linguistics t"]
### Arm A
- Answer: Aspects are identified as explicit aspects (explicitly denoted words) or implicit aspects (inferred from context), using rule-based methods, machine learning, or deep learning approaches, often leveraging syntactic dependency rules, word embeddings, and POS features.
- Selected passage IDs: ["qasper:1710.06536:p6_3", "qasper:1912.07976:p2_0", "qasper:1710.06536:p14_43", "qasper:1710.06536:p8_0", "qasper:1909.00430:p8_6"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: Using dictionary-based, rule-based, machine-learning, or deep learning methodologies, often incorporating syntactic rules, word embeddings, POS features, or CRFs.
- Selected passage IDs: ["qasper:1912.07976:p2_0", "qasper:1710.06536:p14_43", "qasper:1710.06536:p6_4", "qasper:1710.06536:p8_0", "qasper:1710.06536:p8_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 014
- Case ID: `case-014`
- Question: what keyphrase extraction models were reassessed?
- Reference forms for adjudication: ["Answer with content missing: (LVL1, LVL2, LVL3) \n- Stanford CoreNLP\n- Optical Character Recognition (OCR) system, ParsCIT \n- further abridge the input text from level 2 preprocessed documents to the following: title, headers, abstract, introduction, related work, background and conclusion."]
### Arm A
- Answer: several keyphrase extraction models
- Selected passage IDs: ["qasper:1610.07809:p7_0", "qasper:1610.07809:p2_0", "qasper:1610.07809:p0_2", "qasper:1610.07809:p1_2", "qasper:1610.07809:p0_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: five keyphrase extraction models
- Selected passage IDs: ["qasper:1610.07809:p2_0", "qasper:1610.07809:p7_0", "qasper:1610.07809:p0_1", "qasper:1610.07809:p0_2", "qasper:1810.05241:p1_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 015
- Case ID: `case-015`
- Question: How are the accuracy merits of the approach demonstrated?
- Reference forms for adjudication: ["By evaluating the performance of the approach using accuracy and AUC", "significant improvements clearly demonstrate that our approach is effective at improving model performance"]
### Arm A
- Answer: By comparing the accuracy of classifiers trained on the generated dataset to those trained on the original dataset.
- Selected passage IDs: ["qasper:1909.09484:p16_0", "qasper:1910.00194:p10_0", "qasper:1607.06025:p0_8", "qasper:1808.05077:p2_4", "qasper:1904.06941:p4_14"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: By comparing absolute accuracies and showing correlation with BLEU scores.
- Selected passage IDs: ["qasper:1804.04225:p6_2", "qasper:1912.00667:p7_3", "qasper:1912.00871:p11_4", "qasper:1808.05077:p1_3", "qasper:1808.05077:p2_4"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 016
- Case ID: `case-016`
- Question: Which datasets are employed for South African languages LID?
- Reference forms for adjudication: ["DSL 2015", "DSL 2017", "JW300 parallel corpus", "NCHLT text corpora"]
### Arm A
- Answer: NCHLT text corpora, DSL 2015, DSL 2017
- Selected passage IDs: ["qasper:1911.07555:p1_4", "qasper:1911.07555:p1_3", "qasper:1911.07555:p1_0", "qasper:1911.07555:p1_1", "qasper:1911.07555:p0_4"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: NCHLT text corpora, DSL 2015, DSL 2017
- Selected passage IDs: ["qasper:1911.07555:p1_3", "qasper:1911.07555:p0_4", "qasper:1911.07555:p1_1", "qasper:1911.07555:p1_0", "qasper:1911.07555:p1_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 017
- Case ID: `case-017`
- Question: What is the multilingual baseline?
- Reference forms for adjudication: ["the Meta-LSTM BIBREF0"]
### Arm A
- Answer: A simple, open-source, multilingual sentiment analysis model designed for ease of implementation and proven effectiveness in international competitions.
- Selected passage IDs: ["qasper:1612.05270:p1_0", "qasper:1710.09589:p1_0", "qasper:1710.09589:p11_2", "qasper:1909.00100:p0_0", "qasper:1909.09524:p6_19"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: B4MSA
- Selected passage IDs: ["qasper:1612.05270:p1_0", "qasper:1909.09524:p6_14", "qasper:1710.09589:p1_0", "qasper:1911.03894:p3_0", "qasper:2003.10816:p11_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 018
- Case ID: `case-018`
- Question: Which morphosyntactic features are thought to indicate irony or sarcasm?
- Reference forms for adjudication: ["all caps", "emojis", "emoticons", "hashtags", "quotation marks"]
### Arm A
- Answer: rhetorical questions and hyperbole
- Selected passage IDs: ["qasper:1709.05404:p3_4", "qasper:1709.05404:p0_0", "qasper:1809.09795:p0_0", "qasper:1809.09795:p2_0", "qasper:1709.05404:p1_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: intensifiers, elongated words, quotations, false politeness, negative evaluations, emoticons, tag questions, all caps, hashtag
- Selected passage IDs: ["qasper:1709.05404:p0_0", "qasper:1709.05404:p3_4", "qasper:1809.09795:p2_1", "qasper:1904.04019:p0_1", "qasper:1809.09795:p2_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 019
- Case ID: `case-019`
- Question: How much bigger is Switchboard-2000 than Switchboard-300 database?
- Reference forms for adjudication: ["Switchboard-2000 contains 1700 more hours of speech data."]
### Arm A
- Answer: 2000-hour
- Selected passage IDs: ["qasper:2001.07263:p8_0", "qasper:2001.07263:p9_0", "qasper:1709.09119:p30_1", "qasper:1912.03627:p3_0", "qasper:2001.11899:p32_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: 700 hours
- Selected passage IDs: ["qasper:2001.07263:p8_0", "qasper:2001.07263:p2_0", "qasper:1705.03151:p20_2", "qasper:1904.03288:p8_0", "qasper:1807.00868:p2_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 020
- Case ID: `case-020`
- Question: What is specific to gCAS cell?
- Reference forms for adjudication: ["It has three sequentially connected units to output continue, act and slots generating multi-acts in a doble recurrent manner."]
### Arm A
- Answer: It contains three sequentially connected units for outputting continue, act, and slots respectively.
- Selected passage IDs: ["qasper:1908.11546:p2_0", "qasper:1908.11546:p5_5", "qasper:1908.11546:p0_2", "qasper:1908.11546:p7_0", "qasper:1908.11546:p6_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: It contains three sequentially connected units for outputting continue, act, and slots respectively.
- Selected passage IDs: ["qasper:1908.11546:p2_0", "qasper:1908.11546:p5_5", "qasper:1908.11546:p7_0", "qasper:1908.11546:p6_2", "qasper:1908.11546:p6_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 021
- Case ID: `case-021`
- Question: Does their approach require a dataset of unanswerable questions mapped to similar answerable questions?
- Reference forms for adjudication: ["yes"]
### Arm A
- Answer: No
- Selected passage IDs: ["qasper:1906.06045:p0_2", "qasper:1906.06045:p7_0", "qasper:1906.06045:p6_0", "qasper:1906.06045:p0_1", "qasper:1906.06045:p9_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: No
- Selected passage IDs: ["qasper:1906.06045:p0_2", "qasper:1906.06045:p7_0", "qasper:1906.06045:p7_1", "qasper:1911.00841:p15_7", "qasper:1906.06045:p6_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 022
- Case ID: `case-022`
- Question: Which information about text structure is included in the corpus?
- Reference forms for adjudication: ["Information on physical page segmentation (for PDFs only), paragraph segmentation, and line segmentation", "lines", "paragraph, lines, textspan element (paragraph segmentation, line segmentation, Information on physical page segmentation(for PDF only))", "paragraphs"]
### Arm A
- Answer: text structure
- Selected passage IDs: ["qasper:1909.09067:p2_3", "qasper:1909.09067:p0_4", "qasper:1909.09067:p8_0", "qasper:1909.09067:p8_1", "qasper:1909.09067:p5_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: Text structure includes paragraph segmentation, line segmentation, and information on physical page segmentation for PDFs.
- Selected passage IDs: ["qasper:1909.09067:p5_3", "qasper:1909.09067:p0_4", "qasper:1909.09067:p2_3", "qasper:1904.06941:p11_2", "qasper:2003.14026:p0_9"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 023
- Case ID: `case-023`
- Question: what kinds of male and female words are looked at?
- Reference forms for adjudication: ["gendered word pairs like he and she"]
### Arm A
- Answer: male socially-biased words and female socially-biased words
- Selected passage IDs: ["qasper:1910.14497:p11_1", "qasper:1910.14497:p4_0", "qasper:1905.12801:p0_2", "qasper:1707.03764:p4_9", "qasper:1905.12801:p5_5"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: male socially-biased words and female socially-biased words
- Selected passage IDs: ["qasper:1910.14497:p11_1", "qasper:1707.03764:p4_9", "qasper:1910.14497:p4_0", "qasper:1707.03764:p4_1", "qasper:1810.04528:p4_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 024
- Case ID: `case-024`
- Question: What types of various community texts have been investigated for exploring global structure of binomials?
- Reference forms for adjudication: ["news publications, wine reviews, and Reddit"]
### Arm A
- Answer: sports community, politics community, online text data communities, subreddits
- Selected passage IDs: ["qasper:2003.03612:p0_12", "qasper:2003.03612:p0_4", "qasper:2003.03612:p18_1", "qasper:2003.03612:p1_0", "qasper:2003.03612:p0_7"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: news publications, wine reviews, and Reddit
- Selected passage IDs: ["qasper:2003.03612:p0_12", "qasper:2003.03612:p16_0", "qasper:2003.03612:p0_8", "qasper:2003.03612:p0_7", "qasper:2003.03612:p0_5"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 025
- Case ID: `case-025`
- Question: How is the keyword specific expectation elicited from the crowd?
- Reference forms for adjudication: ["asked to find the keyword that best indicates the class of the microposts", "workers are first asked to find those microposts where the model predictions are deemed correct"]
### Arm A
- Answer: By leveraging crowd workers to estimate keyword-specific expectations iteratively.
- Selected passage IDs: ["qasper:1912.00667:p4_0", "qasper:1912.00667:p0_3", "qasper:1912.00667:p2_1", "qasper:1912.00667:p4_1", "qasper:1912.00667:p1_15"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: By leveraging crowd workers for estimating keyword-specific expectations.
- Selected passage IDs: ["qasper:1912.00667:p4_0", "qasper:1912.00667:p1_8", "qasper:1912.00667:p1_14", "qasper:1912.00667:p0_3", "qasper:1912.00667:p2_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 026
- Case ID: `case-026`
- Question: Why mixed initiative multi-turn dialogs are the greatest challenge in building open-domain conversational agents?
- Reference forms for adjudication: ["at each system turn, there are a large number of conversational moves that are possible", "do not follow a particular plan or pursue a particular fixed information need", "integrating content found via search with content from structured data", "live search may not be able to achieve the required speed and efficiency", "most other domains do not have such high quality structured data available"]
### Arm A
- Answer: Because they do not follow a particular plan or pursue a fixed information need, making them difficult to model and maintain coherence in open-domain conversations.
- Selected passage IDs: ["qasper:1709.05411:p0_0", "qasper:1709.05411:p2_0", "qasper:1709.05411:p4_0", "qasper:1705.01214:p1_2", "qasper:1709.05411:p0_3"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: Because they do not follow a particular plan or pursue a fixed information need, requiring real-time balancing of dialogue acts and content availability.
- Selected passage IDs: ["qasper:1709.05411:p0_0", "qasper:1709.05411:p0_8", "qasper:1709.05411:p2_0", "qasper:1709.05411:p1_5", "qasper:1709.05411:p0_3"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 027
- Case ID: `case-027`
- Question: How many annotators were there?
- Reference forms for adjudication: ["3"]
### Arm A
- Answer: 65
- Selected passage IDs: ["qasper:1912.01673:p7_0", "qasper:1909.00338:p3_6", "qasper:2003.04866:p8_0", "qasper:1910.11769:p5_2", "qasper:1912.08320:p35_43"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: 65
- Selected passage IDs: ["qasper:1912.01673:p7_0", "qasper:1909.00338:p3_6", "qasper:1912.01673:p7_2", "qasper:1912.08320:p14_0", "qasper:1912.08320:p35_43"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 028
- Case ID: `case-028`
- Question: What new model is proposed for binomial lists?
- Reference forms for adjudication: ["null model"]
### Arm A
- Answer: A new framework based on asymmetry, movement, and agreement visualized as a cube.
- Selected passage IDs: ["qasper:2003.03612:p3_0", "qasper:2003.03612:p0_8", "qasper:2003.03612:p0_11", "qasper:2003.03612:p6_0", "qasper:2003.03612:p0_6"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: a new framework based on asymmetry, movement, and agreement visualized as a cube
- Selected passage IDs: ["qasper:2003.03612:p3_0", "qasper:2003.03612:p16_0", "qasper:2003.03612:p18_1", "qasper:2003.03612:p0_13", "qasper:2003.03612:p0_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 029
- Case ID: `case-029`
- Question: What other languages did they translate the data from?
- Reference forms for adjudication: ["English"]
### Arm A
- Answer: English, Farsi, French, German, Italian, Japanese
- Selected passage IDs: ["qasper:1805.10824:p4_0", "qasper:2003.10816:p14_1", "qasper:1709.05563:p2_1", "qasper:1906.00378:p10_0", "qasper:1708.01464:p10_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: German, French, and Italian
- Selected passage IDs: ["qasper:1709.05563:p2_1", "qasper:1910.05154:p1_0", "qasper:2003.10816:p14_1", "qasper:2001.11899:p11_4", "qasper:1910.05456:p9_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 030
- Case ID: `case-030`
- Question: What are the disadvantages to clipping negative PMI?
- Reference forms for adjudication: ["It may lead to poor rare word representations and word analogies."]
### Arm A
- Answer: Loss of information about negative cooccurrence, which may be useful for rare words and analogies.
- Selected passage IDs: ["qasper:1908.06941:p0_1", "qasper:1908.06941:p4_5", "qasper:1908.06941:p4_3", "qasper:1704.02686:p3_3", "qasper:1908.06941:p2_7"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: Loss of information from the negative PMI spectrum.
- Selected passage IDs: ["qasper:1908.06941:p0_1", "qasper:1908.06941:p2_9", "qasper:1908.06941:p2_1", "qasper:1908.06941:p4_5", "qasper:1908.06941:p1_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 031
- Case ID: `case-031`
- Question: How does this model overcome the assumption that all words in a document are generated from a single event?
- Reference forms for adjudication: ["by learning a projection function between the document-event distribution and four event related word distributions", "flexibility of neural networks, the generator is capable of learning complicated nonlinear distributions", "supervision signal provided by the discriminator will help generator to capture the event-related patterns"]
### Arm A
- Answer: By modeling multiple events and generating event-related word distributions for each, the model avoids assuming all words in a document stem from a single event.
- Selected passage IDs: ["qasper:1908.09246:p0_2", "qasper:1908.09246:p11_4", "qasper:1908.09246:p8_0", "qasper:1908.09246:p6_1", "qasper:1910.01863:p15_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: By using a generator network to learn complex nonlinear distributions and a discriminator to provide supervision signals that help capture event-related patterns, the AEM model can handle multiple events in a document without assuming all words originate from a single event.
- Selected passage IDs: ["qasper:1908.09246:p0_3", "qasper:1908.09246:p0_2", "qasper:1908.09246:p8_0", "qasper:1910.01863:p17_1", "qasper:1910.01863:p9_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 032
- Case ID: `case-032`
- Question: What kind of Youtube video transcripts did they use?
- Reference forms for adjudication: ["youtube video transcripts on news covering different topics like technology, human rights, terrorism and politics"]
### Arm A
- Answer: YouTube video transcripts from 120M segmented utterances with semi-supervised caption filtering.
- Selected passage IDs: ["qasper:1912.01046:p1_2", "qasper:1912.01046:p1_0", "qasper:2002.11268:p7_1", "qasper:2002.11268:p7_2", "qasper:1912.04961:p13_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: transcripts obtained from semi-supervised caption filtering
- Selected passage IDs: ["qasper:2002.11268:p7_1", "qasper:1802.00923:p6_2", "qasper:1808.08850:p8_0", "qasper:1912.01046:p10_2", "qasper:2002.11268:p8_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 033
- Case ID: `case-033`
- Question: What models other than standalone BERT is new model compared to?
- Reference forms for adjudication: ["Only Bert base and Bert large are compared to proposed approach."]
### Arm A
- Answer: Concat, Attentive Mimicking, Bertram, RoBERTa, XLNet
- Selected passage IDs: ["qasper:2001.00137:p4_0", "qasper:2002.00652:p19_0", "qasper:2001.06286:p0_0", "qasper:1910.07181:p10_2", "qasper:1908.10084:p1_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: RNN-based approach
- Selected passage IDs: ["qasper:2001.06286:p10_0", "qasper:2001.06286:p7_2", "qasper:2001.00137:p4_0", "qasper:2002.00652:p19_0", "qasper:1902.09314:p9_19"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 034
- Case ID: `case-034`
- Question: What multilingual parallel data is used for training proposed model?
- Reference forms for adjudication: ["MultiUN BIBREF20", "OpenSubtitles BIBREF21"]
### Arm A
- Answer: WMT 2014 En ↔ Fr parallel corpus and larger corpora including EPPS, NC, and CommonCrawl
- Selected passage IDs: ["qasper:1809.04686:p4_0", "qasper:1907.02636:p13_0", "qasper:1809.04686:p4_1", "qasper:1611.04798:p6_0", "qasper:1910.11768:p0_6"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: MultiUN and OpenSubtitles
- Selected passage IDs: ["qasper:2004.04721:p0_0", "qasper:1809.04686:p4_0", "qasper:1911.03597:p6_0", "qasper:1611.04798:p6_0", "qasper:1612.06897:p1_3"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 035
- Case ID: `case-035`
- Question: Which model have the smallest Character Error Rate and which have the smallest Word Error Rate?
- Reference forms for adjudication: ["For subword units, classic RNN-transducer, RNN-transducer with attention and joint CTC-attention show comparable performance", "character unit the RNN-transducer with additional attention module"]
### Arm A
- Answer: RNN-transducer with attention module has the smallest CER, and classic RNN-transducer has the smallest WER.
- Selected passage IDs: ["qasper:1910.08502:p11_0", "qasper:1910.08502:p14_0", "qasper:1910.08502:p13_9", "qasper:1910.08502:p13_22", "qasper:1708.01464:p8_3"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: INSUFFICIENT_EVIDENCE
- Selected passage IDs: ["qasper:2002.01320:p9_0", "qasper:1810.03459:p2_2", "qasper:1910.08502:p11_0", "qasper:1910.08502:p13_9", "qasper:1905.11268:p0_6"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 036
- Case ID: `case-036`
- Question: Is the lexicon the same for all languages?
- Reference forms for adjudication: ["yes"]
### Arm A
- Answer: No
- Selected passage IDs: ["qasper:2003.04866:p3_0", "qasper:1909.06522:p0_1", "qasper:2003.14026:p4_4", "qasper:2001.11899:p7_0", "qasper:1909.06522:p3_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: No
- Selected passage IDs: ["qasper:1812.01704:p2_8", "qasper:2003.04866:p3_0", "qasper:1909.06522:p0_1", "qasper:1911.07555:p2_2", "qasper:2003.14026:p4_4"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 037
- Case ID: `case-037`
- Question: Which dimensionality do they use for their embeddings?
- Reference forms for adjudication: ["1061"]
### Arm A
- Answer: 300
- Selected passage IDs: ["qasper:1911.07228:p4_2", "qasper:1907.04152:p8_8", "qasper:1704.02686:p9_3", "qasper:1706.08032:p8_4", "qasper:1706.09147:p3_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: 300-dimensional
- Selected passage IDs: ["qasper:1704.02686:p9_3", "qasper:1907.04152:p8_8", "qasper:1611.02550:p5_0", "qasper:1912.13109:p9_3", "qasper:1611.02550:p9_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 038
- Case ID: `case-038`
- Question: What latent variables are modeled in the PCFG?
- Reference forms for adjudication: ["semantic and topical information", "syntactic information"]
### Arm A
- Answer: syntactic, semantic, and topical information
- Selected passage IDs: ["qasper:1601.06068:p1_0", "qasper:1601.06068:p3_1", "qasper:1906.10225:p2_12", "qasper:1808.09111:p4_0", "qasper:1906.10225:p0_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: latent vector
- Selected passage IDs: ["qasper:1906.10225:p8_4", "qasper:1906.10225:p2_12", "qasper:1906.10225:p1_2", "qasper:1601.06068:p1_0", "qasper:1906.10225:p0_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 039
- Case ID: `case-039`
- Question: Do the authors report performance of conditional bert on tasks without data augmentation?
- Reference forms for adjudication: ["yes"]
### Arm A
- Answer: Yes
- Selected passage IDs: ["qasper:1812.06705:p8_7", "qasper:1812.06705:p9_0", "qasper:1812.06705:p8_6", "qasper:1812.06705:p0_3", "qasper:1811.05085:p9_7"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: INSUFFICIENT_EVIDENCE
- Selected passage IDs: ["qasper:1812.06705:p8_7", "qasper:1812.06705:p6_0", "qasper:1811.05085:p12_5", "qasper:1812.06705:p10_0", "qasper:1811.05085:p9_7"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 040
- Case ID: `case-040`
- Question: How is human interaction consumed by the model?
- Reference forms for adjudication: ["displays three different versions of a story written by three distinct models for a human to compare", "human can select the model to interact with (potentially after having chosen it via cross-model), and can collaborate at all stages"]
### Arm A
- Answer: Human interaction is consumed by the model to learn the user's interest and to ensure the system returns correct formal-queries.
- Selected passage IDs: ["qasper:1910.11949:p2_0", "qasper:1808.04614:p15_0", "qasper:2001.07615:p0_9", "qasper:1805.00460:p1_2", "qasper:1904.02357:p4_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: The model consumes human interaction by using it to learn user interests and allowing users to edit, reject, or request new generations, which influences the model's output and enables iterative refinement.
- Selected passage IDs: ["qasper:1805.00460:p1_2", "qasper:1904.02357:p4_0", "qasper:1904.02357:p0_1", "qasper:1904.02357:p2_0", "qasper:1910.11949:p2_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 041
- Case ID: `case-041`
- Question: How is the intensity of the PTSD established?
- Reference forms for adjudication: ["Given we have four intensity, No PTSD, Low Risk PTSD, Moderate Risk PTSD and High Risk PTSD with a score of 0, 1, 2 and 3 respectively, the estimated intensity  is established as mean  squared error.", "defined into four categories from high risk, moderate risk, to low risk"]
### Arm A
- Answer: The intensity of PTSD is established by assessing weekly scores on three tools (DOSPERT, BSSS, and VIAS) and categorizing based on how many tools exceed their thresholds: low risk (one tool), moderate risk (two tools), or high risk (all three tools).
- Selected passage IDs: ["qasper:2003.07433:p3_0", "qasper:2003.07433:p3_1", "qasper:2003.07433:p5_0", "qasper:2003.07433:p3_3", "qasper:2003.07433:p3_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: By calculating the cumulative sum of s-scores derived from α-scores scaled according to survey systems like DOSPERT, BSSS, and VIAS.
- Selected passage IDs: ["qasper:2003.07433:p3_0", "qasper:2003.07433:p0_4", "qasper:2003.07433:p11_0", "qasper:2003.07433:p13_0", "qasper:2003.07433:p0_5"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 042
- Case ID: `case-042`
- Question: Does the DMN+ model establish state-of-the-art ?
- Reference forms for adjudication: ["yes"]
### Arm A
- Answer: Yes
- Selected passage IDs: ["qasper:1603.01417:p12_6", "qasper:1603.01417:p0_1", "qasper:1603.01417:p12_3", "qasper:1603.01417:p11_4", "qasper:1603.01417:p0_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: Yes
- Selected passage IDs: ["qasper:1603.01417:p12_6", "qasper:1603.01417:p14_0", "qasper:1603.01417:p12_3", "qasper:1603.01417:p11_4", "qasper:1603.01417:p0_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 043
- Case ID: `case-043`
- Question: How much transcribed data is available for for Ainu language?
- Reference forms for adjudication: ["Transcribed data is available for duration of 38h 54m 38s for 8 speakers."]
### Arm A
- Answer: INSUFFICIENT_EVIDENCE
- Selected passage IDs: ["qasper:2002.06675:p2_0", "qasper:2002.06675:p0_1", "qasper:2002.06675:p1_0", "qasper:2002.06675:p3_1", "qasper:1804.11297:p0_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: INSUFFICIENT_EVIDENCE
- Selected passage IDs: ["qasper:2002.06675:p5_0", "qasper:2002.06675:p0_1", "qasper:2002.06675:p5_1", "qasper:2002.06675:p2_0", "qasper:2002.06675:p7_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 044
- Case ID: `case-044`
- Question: How much is proposed model better in perplexity and BLEU score than typical UMT models?
- Reference forms for adjudication: ["Perplexity of the best model is 65.58 compared to best baseline 105.79.\nBleu of the best model is 6.57 compared to best baseline 5.50."]
### Arm A
- Answer: INSUFFICIENT_EVIDENCE
- Selected passage IDs: ["qasper:1701.06538:p11_1", "qasper:1811.11365:p10_1", "qasper:1905.01715:p8_2", "qasper:1701.06538:p10_2", "qasper:1701.06538:p18_6"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: INSUFFICIENT_EVIDENCE
- Selected passage IDs: ["qasper:1701.06538:p10_2", "qasper:1811.11365:p10_1", "qasper:1701.06538:p18_6", "qasper:1701.06538:p11_1", "qasper:1909.00279:p10_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 045
- Case ID: `case-045`
- Question: How do their models decide how much improtance to give to the output words?
- Reference forms for adjudication: ["Given the contribution matrix, we can obtain the word importance of each input word to the entire output sentence.", "They compute the gradient of the output at each time step with respect to the input words to decide the importance."]
### Arm A
- Answer: They aggregate the contribution values of each input word to all output words by summing them and then normalize the sums using the Softmax function.
- Selected passage IDs: ["qasper:1909.00326:p6_4", "qasper:1901.03859:p7_19", "qasper:1909.00326:p5_0", "qasper:1907.05664:p10_1", "qasper:1909.00326:p8_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: The model uses a gating mechanism to decide how much attention to pay to each pair of form and context embeddings.
- Selected passage IDs: ["qasper:1912.00667:p6_5", "qasper:1901.03859:p7_19", "qasper:1910.07181:p2_1", "qasper:1811.12889:p0_1", "qasper:1611.08661:p6_3"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 046
- Case ID: `case-046`
- Question: What semi-supervised learning is applied?
- Reference forms for adjudication: ["This silver data is then simply added to our training set, after which the model is retrained", "first a model is trained on the training set and then this model is used to predict the labels of the silver data"]
### Arm A
- Answer: Cross-View Training
- Selected passage IDs: ["qasper:1908.01294:p0_6", "qasper:1701.09123:p4_7", "qasper:1805.10824:p6_0", "qasper:1809.00530:p1_2", "qasper:1710.01492:p7_4"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: Cross-View Training
- Selected passage IDs: ["qasper:1805.10824:p6_2", "qasper:1908.01294:p0_6", "qasper:1701.09123:p8_0", "qasper:1911.00069:p20_3", "qasper:1809.00530:p0_4"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 047
- Case ID: `case-047`
- Question: What are the three datasets used?
- Reference forms for adjudication: ["DSTC2", "M2M-sim-M", "M2M-sim-R"]
### Arm A
- Answer: IWSLT14 German-English, Turkish-English, and WMT14 English-German
- Selected passage IDs: ["qasper:1806.00722:p6_0", "qasper:1706.02027:p8_0", "qasper:1611.02988:p2_0", "qasper:1711.05345:p5_0", "qasper:1709.10367:p4_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: IWSLT14 German-English, Turkish-English, and WMT14 English-German
- Selected passage IDs: ["qasper:1806.00722:p6_0", "qasper:1709.02271:p4_0", "qasper:1909.05246:p8_0", "qasper:1706.02027:p8_0", "qasper:1603.01417:p7_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 048
- Case ID: `case-048`
- Question: What details are given about the Twitter dataset?
- Reference forms for adjudication: ["Those tweets are mostly much noisier and shorter compared to the reviews in the movie corpus. In total, there are 1,716 tweets. 973 of them are negative and 743 of them are positive.", "one of the Twitter datasets is about Turkish mobile network operators, there are positive, neutral and negative labels and provide the total amount plus the distribution of labels"]
### Arm A
- Answer: The Twitter dataset used for SemEval 2018 Task 3 is manually annotated with binary labels for irony detection, includes a manually annotated dataset for sarcasm, and a user self-annotated corpus of tweets with the #sarcasm hashtag. Another Twitter dataset consists of a random sample of roughly 1
- Selected passage IDs: ["qasper:1809.09795:p3_1", "qasper:1803.09745:p1_3", "qasper:1907.01468:p3_11", "qasper:1701.04653:p2_2", "qasper:2001.01269:p8_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: The Twitter dataset consists of a random sample of roughly 10% of all tweets posted between 9 September 2008 and 22 October 2017, comprising more than 106 billion messages from about 750 million unique accounts. It includes verb forms of interest,
- Selected passage IDs: ["qasper:1803.09745:p1_3", "qasper:1910.06748:p4_0", "qasper:1809.09795:p3_1", "qasper:1909.10012:p3_3", "qasper:1910.06748:p0_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 049
- Case ID: `case-049`
- Question: How faster is training and decoding compared to former models?
- Reference forms for adjudication: ["Proposed vs best baselin<redacted-path> 8541 vs 8532 tokens/sec\nTraining: 8h vs 8h"]
### Arm A
- Answer: 3x faster to decode and 1.5x faster to train compared to former models.
- Selected passage IDs: ["qasper:1705.01991:p11_3", "qasper:1705.01991:p0_2", "qasper:1911.08673:p13_0", "qasper:1703.07090:p0_6", "qasper:1705.01991:p9_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: faster
- Selected passage IDs: ["qasper:1809.02731:p5_0", "qasper:1705.01991:p11_3", "qasper:1911.08673:p13_0", "qasper:1911.08673:p16_0", "qasper:1705.01991:p0_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 050
- Case ID: `case-050`
- Question: What provisional explanation do the authors give for the impact of document context?
- Reference forms for adjudication: ["adding context causes speakers to focus on broader semantic and pragmatic issues of discourse coherence"]
### Arm A
- Answer: Adding context causes speakers to focus on broader semantic and pragmatic issues of discourse coherence, rather than simply judging syntactic well formedness.
- Selected passage IDs: ["qasper:1809.01060:p2_4", "qasper:1909.02776:p1_7", "qasper:1809.01060:p2_3", "qasper:2004.03034:p2_10", "qasper:2004.03034:p3_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: Document context can improve or reduce human ratings depending on the sentence's acceptability, and models relying on discourse-based indicators may be misled in cross-document settings.
- Selected passage IDs: ["qasper:1811.00266:p5_0", "qasper:1809.01060:p2_3", "qasper:1906.03338:p5_8", "qasper:1909.02776:p1_7", "qasper:2004.03034:p13_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 051
- Case ID: `case-051`
- Question: How big are improvements with multilingual ASR training vs single language training?
- Reference forms for adjudication: ["relative WER improvement of 10%."]
### Arm A
- Answer: Improvements range from 4.6% to 10.3% relative WER reduction.
- Selected passage IDs: ["qasper:2002.06675:p20_5", "qasper:1909.06522:p12_0", "qasper:1809.01541:p5_4", "qasper:1909.06522:p14_0", "qasper:1909.06522:p1_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: 7.96% higher on average
- Selected passage IDs: ["qasper:1809.01541:p5_4", "qasper:1909.06522:p14_0", "qasper:1909.06522:p12_3", "qasper:1909.06522:p12_0", "qasper:1909.06522:p13_3"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 052
- Case ID: `case-052`
- Question: How is the correct number of experts to use decided?
- Reference forms for adjudication: ["varied the number of experts between models"]
### Arm A
- Answer: The correct number is determined by the needs of the user, not by the collection.
- Selected passage IDs: ["qasper:1701.06538:p18_7", "qasper:1907.01468:p13_2", "qasper:1701.06538:p5_3", "qasper:1701.06538:p19_4", "qasper:1701.06538:p3_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: The correct number of experts is decided based on performance improvement, as increasing the number of experts continued to improve test perplexity up to 2048.
- Selected passage IDs: ["qasper:1701.06538:p18_7", "qasper:1811.01299:p9_6", "qasper:1701.06538:p16_1", "qasper:1701.06538:p18_2", "qasper:1907.01468:p13_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 053
- Case ID: `case-053`
- Question: How much better was results of CamemBERT than previous results on these tasks?
- Reference forms for adjudication: ["2.36 point increase in the F1 score with respect to the best SEM architecture", "For POS tagging, we observe error reductions of respectively 0.71% for GSD, 0.81% for Sequoia, 0.7% for Spoken and 0.28% for ParTUT", "For parsing, we observe error reductions in LAS of 2.96% for GSD, 3.33% for Sequoia, 1.70% for Spoken and 1.65% for ParTUT", "lags behind models trained on the original English training set in the TRANSLATE-TEST setting, 81.2 vs. 82.91 for RoBERTa", "on the TRANSLATE-TRAIN setting (81.2 vs. 80.2 for XLM)"]
### Arm A
- Answer: CamemBERT improved state-of-the-art performance for most tasks over previous monolingual and multilingual approaches.
- Selected passage IDs: ["qasper:1911.03894:p22_0", "qasper:1911.03894:p19_0", "qasper:1911.03894:p21_0", "qasper:1911.03894:p20_0", "qasper:1911.03894:p0_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: CamemBERT improved performance by 0.71% to 3.5 points and 2.36 points F1 score increase over previous models on specific tasks.
- Selected passage IDs: ["qasper:1911.03894:p22_0", "qasper:1911.03894:p19_1", "qasper:1911.03894:p21_0", "qasper:1911.03894:p0_2", "qasper:1911.03894:p20_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 054
- Case ID: `case-054`
- Question: What contextual features are used?
- Reference forms for adjudication: ["The words that can indicate the characteristics of the neighbor words as contextual keywords and generate it from the automatically extracted contextual keywords."]
### Arm A
- Answer: contextual keywords, window size, token frequency, corpus frequency
- Selected passage IDs: ["qasper:2003.08553:p7_3", "qasper:1907.02636:p7_2", "qasper:1907.02636:p7_5", "qasper:1907.02636:p12_0", "qasper:1602.06291:p0_8"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: topics based on different segments of text, contextual keywords, co-occurring words indicating characteristics of neighbor words
- Selected passage IDs: ["qasper:2003.08553:p7_0", "qasper:1602.06291:p0_8", "qasper:1912.03234:p2_0", "qasper:2003.08553:p7_3", "qasper:1907.02636:p7_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 055
- Case ID: `case-055`
- Question: What model is used to encode the images?
- Reference forms for adjudication: ["LSTM", "a Convolutional Neural Network (CNN)"]
### Arm A
- Answer: ResNet-152
- Selected passage IDs: ["qasper:1809.01341:p5_4", "qasper:1806.00738:p3_0", "qasper:1906.00378:p3_2", "qasper:1711.06288:p7_2", "qasper:1811.11365:p10_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: VGG-16
- Selected passage IDs: ["qasper:1711.06288:p7_2", "qasper:1906.00378:p3_2", "qasper:1811.11365:p10_0", "qasper:1909.00279:p11_3", "qasper:1812.10479:p13_5"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 056
- Case ID: `case-056`
- Question: Which two news domains are country-independent?
- Reference forms for adjudication: ["mainstream and disinformation news", "mainstream news and disinformation"]
### Arm A
- Answer: mainstream and disinformation news
- Selected passage IDs: ["qasper:2002.12612:p10_5", "qasper:2002.12612:p12_2", "qasper:2002.12612:p1_2", "qasper:2002.12612:p0_7", "qasper:2002.12612:p3_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: political news and disinformation news
- Selected passage IDs: ["qasper:2002.12612:p10_5", "qasper:2002.12612:p12_2", "qasper:2002.12612:p0_7", "qasper:1910.09295:p12_0", "qasper:2002.12612:p10_7"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 057
- Case ID: `case-057`
- Question: Which social media platform is explored?
- Reference forms for adjudication: ["twitter"]
### Arm A
- Answer: Twitter
- Selected passage IDs: ["qasper:1908.00153:p3_2", "qasper:1710.01492:p3_2", "qasper:1608.04917:p1_0", "qasper:2001.08845:p0_1", "qasper:1908.04531:p3_12"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: Sina Weibo
- Selected passage IDs: ["qasper:2001.08845:p0_1", "qasper:1710.01492:p3_2", "qasper:1908.00153:p3_2", "qasper:1908.04531:p3_12", "qasper:1610.04377:p0_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 058
- Case ID: `case-058`
- Question: Does the use of out-of-domain data improve the performance of the method?
- Reference forms for adjudication: ["no"]
### Arm A
- Answer: Yes
- Selected passage IDs: ["qasper:1705.01265:p4_5", "qasper:1810.05334:p7_1", "qasper:1904.04019:p17_3", "qasper:1701.03214:p10_3", "qasper:1909.02764:p15_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: No
- Selected passage IDs: ["qasper:1804.07789:p14_0", "qasper:1612.06897:p6_1", "qasper:1701.03214:p5_4", "qasper:1701.09123:p1_5", "qasper:1705.01265:p4_5"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 059
- Case ID: `case-059`
- Question: What were the baselines?
- Reference forms for adjudication: ["Recursive Neural Networks (BIBREF9,BIBREF10), Convolutional Neural Networks (BIBREF11) and most recently transfer learning-based architectures like Bidirectional Encoder Representation from Transformers (BERT) (BIBREF12)", "varied from Maximum Entropy Classifiers (BIBREF4) to Support Vector Machines (BIBREF5,BIBREF6,BIBREF7,BIBREF8)"]
### Arm A
- Answer: The baselines were models chosen for comparison in the experiments.
- Selected passage IDs: ["qasper:1908.09246:p10_8", "qasper:1811.00625:p8_0", "qasper:1811.04604:p9_0", "qasper:1901.01010:p9_0", "qasper:1909.09484:p14_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: The baselines were models used for comparison in the task.
- Selected passage IDs: ["qasper:1811.04604:p9_0", "qasper:1811.00625:p8_0", "qasper:1703.10344:p5_19", "qasper:2001.07786:p5_0", "qasper:1909.09524:p6_12"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 060
- Case ID: `case-060`
- Question: What baseline is used for the experimental setup?
- Reference forms for adjudication: ["CNN-avgmax", "CNN-mean"]
### Arm A
- Answer: CHAR and word
- Selected passage IDs: ["qasper:1909.07593:p8_0", "qasper:2002.01207:p7_1", "qasper:2002.01207:p11_1", "qasper:1811.04604:p9_0", "qasper:1809.00530:p9_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: word surface forms only
- Selected passage IDs: ["qasper:2002.01207:p11_1", "qasper:1909.07593:p8_0", "qasper:2002.01207:p7_1", "qasper:1804.07789:p5_0", "qasper:1908.09246:p9_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 061
- Case ID: `case-061`
- Question: How do they evaluate their method?
- Reference forms for adjudication: ["Calinski-Harabasz score", "UMAP", "t-SNE"]
### Arm A
- Answer: They evaluate their method through a human evaluation.
- Selected passage IDs: ["qasper:1911.03350:p0_7", "qasper:1808.03738:p6_0", "qasper:1601.02543:p0_1", "qasper:1911.03350:p5_0", "qasper:2003.04642:p0_3"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: They compute automatic metrics on their test set.
- Selected passage IDs: ["qasper:1910.10781:p1_0", "qasper:1708.01464:p5_2", "qasper:1911.09845:p10_0", "qasper:2004.01694:p2_2", "qasper:1601.02543:p0_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 062
- Case ID: `case-062`
- Question: What datasets are used to evaluate the model?
- Reference forms for adjudication: ["WN18 and FB15k"]
### Arm A
- Answer: QuasarT, TriviaQA, SearchQA, NYT, XSum, CNN/DailyMail, DUC-2003, DUC-2004, PTB, E2E, PKU, MSR, AS, CITYU
- Selected passage IDs: ["qasper:1901.03866:p8_0", "qasper:1912.11602:p9_0", "qasper:1911.05343:p3_0", "qasper:1910.14537:p5_0", "qasper:2002.06424:p7_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: QuasarT, TriviaQA, SearchQA, Penn Treebank, E2E, NYT, XSum, CNN/DailyMail, DUC-2003, DUC-2004, DSTC2, M2M-sim-M, M2M-sim-R
- Selected passage IDs: ["qasper:1901.03866:p8_0", "qasper:1911.05343:p3_0", "qasper:1811.09353:p6_0", "qasper:1912.11602:p9_0", "qasper:1909.05246:p8_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 063
- Case ID: `case-063`
- Question: How much does HAS-QA improve over baselines?
- Reference forms for adjudication: ["For example, in QuasarT, it improves 16.8% in EM score and 20.4% in F1 score.", "For example, in QuasarT, it improves 4.6% in EM score and 3.5% in F1 score."]
### Arm A
- Answer: 4.6% in EM score and 3.5% in F1 score
- Selected passage IDs: ["qasper:1901.03866:p10_2", "qasper:1901.03866:p10_1", "qasper:1901.03866:p0_12", "qasper:1901.03866:p11_2", "qasper:1901.03866:p12_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: 4.6% in EM score and 3.5% in F1 score
- Selected passage IDs: ["qasper:1901.03866:p10_2", "qasper:1901.03866:p10_1", "qasper:1901.03866:p12_0", "qasper:1901.03866:p0_12", "qasper:1901.03866:p11_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 064
- Case ID: `case-064`
- Question: What translationese effects are seen in the analysis?
- Reference forms for adjudication: ["explicitation effect", "potentially indicating a shining through effect"]
### Arm A
- Answer: simplification, explicitation, normalization, interference
- Selected passage IDs: ["qasper:2004.04721:p4_0", "qasper:2004.01694:p17_2", "qasper:1911.01188:p2_2", "qasper:2004.01694:p17_5", "qasper:2004.01694:p17_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: simpler, more explicit, and more normalised
- Selected passage IDs: ["qasper:1911.01188:p2_2", "qasper:1903.03467:p5_0", "qasper:2004.01694:p17_5", "qasper:2004.01694:p17_1", "qasper:2004.01694:p17_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 065
- Case ID: `case-065`
- Question: Which inter-annotator metric do they use?
- Reference forms for adjudication: ["Kappa value", "agreement rates"]
### Arm A
- Answer: average pairwise inter-annotator agreement (APIAA)
- Selected passage IDs: ["qasper:2003.04866:p8_17", "qasper:2003.04642:p10_1", "qasper:1809.05752:p4_0", "qasper:1808.06834:p4_1", "qasper:1912.03184:p27_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: Fleiss's Kappa
- Selected passage IDs: ["qasper:1809.05752:p4_0", "qasper:1912.00819:p5_2", "qasper:1912.08320:p35_73", "qasper:1601.02403:p7_8", "qasper:1912.08320:p18_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 066
- Case ID: `case-066`
- Question: Which real-world datasets did they use?
- Reference forms for adjudication: ["SST-2", "SST-2 (Stanford Sentiment Treebank, version 2)", "Snips"]
### Arm A
- Answer: TRAC 2018 Dataset, Kaggle Dataset
- Selected passage IDs: ["qasper:1703.03097:p7_0", "qasper:2001.05493:p9_0", "qasper:1912.00667:p0_7", "qasper:2004.01670:p1_8", "qasper:1705.04153:p8_5"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: TRAC 2018 Dataset, Kaggle Dataset, four real-world human trafficking datasets, large real-world LBSN dataset
- Selected passage IDs: ["qasper:1804.09301:p0_4", "qasper:1703.03097:p7_0", "qasper:2001.05493:p9_0", "qasper:1912.00667:p0_7", "qasper:1909.09551:p6_7"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 067
- Case ID: `case-067`
- Question: What datasets are used?
- Reference forms for adjudication: ["The Wikipedia Toxic Comments dataset"]
### Arm A
- Answer: DSTC2, M2M-sim-M, M2M-sim-R, TRAC 2018 Dataset, Kaggle Dataset, IMDB, Yelp2014, Cell Phone, Baby, SQuAD v1.1, SQuAD v2.0, Quoref
- Selected passage IDs: ["qasper:1909.05246:p8_0", "qasper:2001.05493:p9_0", "qasper:1809.00530:p6_5", "qasper:1911.02855:p17_0", "qasper:1904.04019:p0_21"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: TRAC 2018 Dataset, Kaggle Dataset, DSTC2, M2M-sim-M, M2M-sim-R, 16 review corpora, SQuAD v1.1, SQuAD v2.0, Quoref
- Selected passage IDs: ["qasper:2001.05493:p9_0", "qasper:1909.05246:p8_0", "qasper:1804.08139:p7_1", "qasper:1910.10781:p5_0", "qasper:1911.02855:p17_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 068
- Case ID: `case-068`
- Question: Which corpus do they use?
- Reference forms for adjudication: ["The dataset was created by using translations provided by Tatoeba and OpenSubtitles BIBREF16."]
### Arm A
- Answer: BioScope Corpus
- Selected passage IDs: ["qasper:2001.02885:p0_4", "qasper:1904.04019:p14_3", "qasper:1909.01247:p25_1", "qasper:1911.07228:p3_0", "qasper:1908.10322:p8_3"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: BioScope Abstracts, BioScope Full Papers, and SFU Review Corpus
- Selected passage IDs: ["qasper:2001.02885:p0_4", "qasper:1911.07228:p3_0", "qasper:1710.06536:p14_16", "qasper:1904.04019:p14_3", "qasper:1710.00341:p8_3"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 069
- Case ID: `case-069`
- Question: Is is known whether Sina Weibo posts are censored by humans or some automatic classifier?
- Reference forms for adjudication: ["no"]
### Arm A
- Answer: Yes
- Selected passage IDs: ["qasper:2001.08845:p5_0", "qasper:2001.08845:p0_1", "qasper:2001.08845:p0_2", "qasper:2001.08845:p0_5", "qasper:2001.08845:p18_18"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: Yes
- Selected passage IDs: ["qasper:2001.08845:p0_2", "qasper:2001.08845:p0_1", "qasper:2001.08845:p18_18", "qasper:2001.08845:p0_5", "qasper:2001.08845:p1_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 070
- Case ID: `case-070`
- Question: How many layers are there in their model?
- Reference forms for adjudication: ["two LSTM layers"]
### Arm A
- Answer: Four
- Selected passage IDs: ["qasper:1712.03556:p1_1", "qasper:1812.10479:p13_1", "qasper:1809.03449:p7_0", "qasper:1902.00756:p12_0", "qasper:1909.00100:p13_5"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: six
- Selected passage IDs: ["qasper:1810.00663:p4_2", "qasper:1902.00756:p12_0", "qasper:1712.03556:p1_1", "qasper:1902.00756:p5_5", "qasper:1812.10479:p13_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 071
- Case ID: `case-071`
- Question: what dataset statistics are provided?
- Reference forms for adjudication: ["Distribution of category labels, number of answerable-not answerable questions, number of text-based and script-based questions, average text, question, and answer length, number of questions per text", "More than 2,100 texts were paired with 15 questions each, resulting in a total number of approx. 32,000 annotated questions. 13% of the questions are not answerable.  Out of the answerable questions, 10,160 could be answered from the text directly (text-based) and 3,914 questions required the use of commonsense knowledge (script-based).  The final dataset comprises 13,939 questions, 3,827 of which require commonsense knowledge (i.e. 27.4%)."]
### Arm A
- Answer: number of dialogues, turns, and words; statistics on sources of different content; dataset statistics
- Selected passage IDs: ["qasper:1812.07023:p2_1", "qasper:1908.04042:p2_0", "qasper:1906.03538:p12_0", "qasper:1811.09353:p13_0", "qasper:1907.08937:p10_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: Dataset statistics are provided in Table TABREF34 and Table TABREF46.
- Selected passage IDs: ["qasper:1811.09353:p13_0", "qasper:2004.04315:p0_3", "qasper:1908.04042:p2_0", "qasper:2001.06888:p5_0", "qasper:1906.03538:p12_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 072
- Case ID: `case-072`
- Question: How are content clusters used to improve the prediction of incident severity?
- Reference forms for adjudication: ["they are used as additional features in a supervised classification task"]
### Arm A
- Answer: Content clusters improve prediction by serving as additional features in supervised classification, enhancing performance when combined with text embeddings.
- Selected passage IDs: ["qasper:1909.00183:p0_5", "qasper:1909.00183:p15_3", "qasper:1909.00183:p9_0", "qasper:1909.00183:p11_3", "qasper:1909.00183:p13_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: Content clusters improve the prediction of incident severity by serving as additional features in a supervised classifier, enhancing performance when combined with free-text descriptions and other hand-coded features.
- Selected passage IDs: ["qasper:1909.00183:p13_0", "qasper:1909.00183:p15_3", "qasper:1909.00183:p9_0", "qasper:1909.00183:p15_0", "qasper:1909.00183:p0_5"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 073
- Case ID: `case-073`
- Question: what evaluation metrics were used?
- Reference forms for adjudication: ["weighted F1-score"]
### Arm A
- Answer: tokenized BLEU, METEOR, and two new metrics for over and under-translation
- Selected passage IDs: ["qasper:1708.01464:p8_0", "qasper:1805.08241:p4_3", "qasper:1805.11535:p18_1", "qasper:2002.12612:p7_1", "qasper:1912.05066:p10_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, ROUGE-L
- Selected passage IDs: ["qasper:1805.11535:p18_1", "qasper:1708.01464:p8_0", "qasper:1910.04601:p1_3", "qasper:1910.06036:p10_1", "qasper:1912.05066:p10_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 074
- Case ID: `case-074`
- Question: What news dataset was used?
- Reference forms for adjudication: ["collection of headlines published by HuffPost BIBREF12 between 2012 and 2018"]
### Arm A
- Answer: News Category Dataset
- Selected passage IDs: ["qasper:1909.07512:p2_0", "qasper:1707.06806:p9_0", "qasper:1909.00088:p13_0", "qasper:1908.09246:p10_6", "qasper:1910.09295:p12_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: The NowThisNews dataset
- Selected passage IDs: ["qasper:1707.06806:p9_0", "qasper:1909.07512:p2_0", "qasper:2004.01980:p15_0", "qasper:1910.01863:p0_5", "qasper:2003.03612:p2_5"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 075
- Case ID: `case-075`
- Question: What other factors affect the performance?
- Reference forms for adjudication: ["architecture of the classifier", "input domain", "sentence length"]
### Arm A
- Answer: sentence length, perplexity, probability of the good sentence
- Selected passage IDs: ["qasper:1906.10519:p17_4", "qasper:1912.00582:p17_0", "qasper:1807.09671:p8_10", "qasper:1908.08717:p16_0", "qasper:1807.09671:p8_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: sentence length, perplexity, probability of the good sentence
- Selected passage IDs: ["qasper:1906.10519:p17_4", "qasper:1912.00582:p17_0", "qasper:1807.09671:p9_3", "qasper:1807.09671:p8_10", "qasper:1807.09671:p8_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 076
- Case ID: `case-076`
- Question: Do the authors test their annotation projection techniques on tasks other than AMR?
- Reference forms for adjudication: ["no"]
### Arm A
- Answer: Yes
- Selected passage IDs: ["qasper:1704.04539:p9_0", "qasper:1704.04539:p0_2", "qasper:1704.04539:p2_0", "qasper:1704.04539:p8_0", "qasper:1704.04539:p5_4"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: Yes
- Selected passage IDs: ["qasper:1704.04539:p9_0", "qasper:1704.04539:p2_0", "qasper:1704.04539:p5_0", "qasper:1704.04539:p5_4", "qasper:1704.04539:p0_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 077
- Case ID: `case-077`
- Question: Is it a neural model? How is it trained?
- Reference forms for adjudication: ["No, it is a probabilistic model trained by finding feature weights through gradient ascent"]
### Arm A
- Answer: Yes, it is a neural model. It is trained to maximize the log probability of user utterances or by minimizing the negative log likelihood using stochastic gradient descent.
- Selected passage IDs: ["qasper:1909.09270:p18_2", "qasper:1805.06966:p7_0", "qasper:1912.00582:p1_0", "qasper:1707.00110:p1_6", "qasper:1809.04960:p6_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: Yes, it is a neural model; trained to maximize log probability or negative log likelihood using stochastic gradient descent.
- Selected passage IDs: ["qasper:1805.06966:p7_0", "qasper:1809.04960:p6_0", "qasper:1707.00110:p1_6", "qasper:1711.11221:p5_0", "qasper:1808.07625:p10_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 078
- Case ID: `case-078`
- Question: What experiments do the authors present to validate their system?
- Reference forms for adjudication: ["evaluations are done by managed judges who understands the knowledge base and then judge user queries relevance to the QA pairs", "we measure our system's performance for datasets across various domains"]
### Arm A
- Answer: experiments on the two scientific paper datasets (arXiv and PubMed)
- Selected passage IDs: ["qasper:1910.06748:p3_1", "qasper:1909.08089:p10_0", "qasper:1804.08782:p8_0", "qasper:1909.07593:p0_8", "qasper:1710.07960:p3_3"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: experiments on the two scientific paper datasets (arXiv and PubMed)
- Selected passage IDs: ["qasper:1910.11204:p0_7", "qasper:1909.08089:p10_0", "qasper:1906.10551:p11_4", "qasper:1804.08782:p8_0", "qasper:1909.07593:p0_8"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 079
- Case ID: `case-079`
- Question: what datasets were used?
- Reference forms for adjudication: ["WikiNews", "WikiNews test set BIBREF31", "a large collection of fully diacritized classical texts", "diacritized corpus that was used to train the RDI BIBREF7 diacritizer and the Farasa diacritizer BIBREF31", "large collection of fully diacritized classical texts (2.7M tokens) from a book publisher", "the diacritized corpus that was used to train the RDI BIBREF7 diacritizer"]
### Arm A
- Answer: DSTC2, M2M-sim-M, M2M-sim-R, SQuADv1.1, NewsQA, 16 review corpora datasets
- Selected passage IDs: ["qasper:1909.05246:p8_0", "qasper:1909.02560:p10_0", "qasper:1910.10781:p5_0", "qasper:1905.07471:p5_0", "qasper:1804.08139:p7_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: DSTC2, M2M-sim-M, M2M-sim-R, IMDB, Yelp2014, Cell Phone, Baby, Affective Text, Fairy Tales, ISEAR, 16 review corpora datasets, two CQA datasets
- Selected passage IDs: ["qasper:1909.05246:p8_0", "qasper:1809.00530:p6_5", "qasper:1611.02988:p2_0", "qasper:1804.08139:p7_1", "qasper:1912.00864:p9_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 080
- Case ID: `case-080`
- Question: What is their baseline model?
- Reference forms for adjudication: ["the model proposed in BIBREF3"]
### Arm A
- Answer: The baseline model uses Naive Bayes, SVM, and Maximum Entropy classifiers with feature vectors composed of Unigrams, Bigrams, Unigrams + Bigrams, or Unigrams + POS tags.
- Selected passage IDs: ["qasper:1701.03051:p7_0", "qasper:1905.06512:p1_1", "qasper:1805.00460:p2_16", "qasper:1811.00625:p8_0", "qasper:1909.08824:p7_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: Naive Bayes, SVM, and Maximum Entropy classifiers with feature vectors of Unigrams, Bigrams, Unigrams + Bigrams, or Unigrams + POS tags
- Selected passage IDs: ["qasper:1701.03051:p7_0", "qasper:1701.03051:p8_0", "qasper:1905.06512:p1_1", "qasper:1912.10435:p9_5", "qasper:1809.04960:p10_5"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 081
- Case ID: `case-081`
- Question: What lexico-syntactic cues are used to retrieve sarcastic utterances?
- Reference forms for adjudication: ["adjective and adverb patterns", "verb, subject, and object arguments", "verbal patterns"]
### Arm A
- Answer: rhetorical questions, hyperbole, intensifiers, elongated words, quotations, false politeness, negative evaluations, emoticons, tag questions, exclamations, interjections, all caps, typographic markers, emojis, hashtags
- Selected passage IDs: ["qasper:1709.05404:p3_4", "qasper:1709.05404:p0_0", "qasper:1709.05404:p6_1", "qasper:1809.09795:p0_3", "qasper:1709.05404:p3_5"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: rhetorical questions, hyperbole, intensifiers, elongated words, quotations, false politeness, negative evaluations, emoticons, tag questions, exclamations, interjections, all caps, hashtag, emojis
- Selected passage IDs: ["qasper:1709.05404:p3_4", "qasper:1709.05404:p0_0", "qasper:1709.05404:p6_1", "qasper:1809.09795:p0_3", "qasper:1709.05404:p3_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 082
- Case ID: `case-082`
- Question: What data were they used to train the multilingual encoder?
- Reference forms for adjudication: ["WMT 2014 En-Fr parallel corpus"]
### Arm A
- Answer: MultiUN and OpenSubtitles corpora
- Selected passage IDs: ["qasper:2002.06675:p16_1", "qasper:1912.01214:p2_1", "qasper:1911.03597:p6_0", "qasper:1807.00868:p1_7", "qasper:1911.03597:p7_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: MultiUN and OpenSubtitles corpora
- Selected passage IDs: ["qasper:2002.06675:p16_1", "qasper:1911.03597:p6_0", "qasper:1807.00868:p1_7", "qasper:1708.01464:p2_1", "qasper:1809.04686:p5_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 083
- Case ID: `case-083`
- Question: How long are the datasets?
- Reference forms for adjudication: ["Travel dataset contains 4100 raw samples, 11291 clauses, Hotel dataset contains 3825 raw samples, 11264 clauses, and the Mobile dataset contains 3483 raw samples and 8118 clauses"]
### Arm A
- Answer: INSUFFICIENT_EVIDENCE
- Selected passage IDs: ["qasper:1909.08306:p2_0", "qasper:1908.01294:p7_4", "qasper:1901.03859:p8_2", "qasper:1912.07976:p17_1", "qasper:1705.04153:p8_5"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: INSUFFICIENT_EVIDENCE
- Selected passage IDs: ["qasper:1909.08306:p2_0", "qasper:1908.01294:p7_4", "qasper:1901.03859:p8_2", "qasper:1811.09353:p13_0", "qasper:1705.04153:p8_5"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 084
- Case ID: `case-084`
- Question: How much is performance improved on NLI?
- Reference forms for adjudication: ["The average score improved by 1.4 points over the previous best result.", "improvement on the RTE dataset is significant, i.e., 4% absolute gain over the BERTBase"]
### Arm A
- Answer: 1-2 points
- Selected passage IDs: ["qasper:1908.10084:p6_2", "qasper:1804.11346:p0_2", "qasper:1907.04380:p0_3", "qasper:1804.11346:p1_0", "qasper:1810.09774:p5_6"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: 1-2 points
- Selected passage IDs: ["qasper:1908.10084:p6_2", "qasper:1901.02222:p0_4", "qasper:1810.09774:p5_6", "qasper:1911.03894:p14_1", "qasper:1809.02279:p8_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 085
- Case ID: `case-085`
- Question: What method was used to generate the OpenIE extractions?
- Reference forms for adjudication: ["for each multiple-choice question $(q,A) \\in Q_\\mathit {tr}$ and each choice $a \\in A$ , we use all non-stopword tokens in $q$ and $a$ as an ElasticSearch query against S", "take the top 200 hits, run Open IE v4, and aggregate the resulting tuples over all $a \\in A$ and over all questions in $Q_\\mathit {tr}$"]
### Arm A
- Answer: sequence to sequence model with attention
- Selected passage IDs: ["qasper:1905.07471:p9_0", "qasper:1905.13413:p0_0", "qasper:1905.07471:p0_0", "qasper:1905.13413:p0_2", "qasper:1905.07471:p14_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: sequence to sequence model with attention
- Selected passage IDs: ["qasper:1905.07471:p9_0", "qasper:1802.05574:p2_0", "qasper:1910.06036:p3_0", "qasper:1905.13413:p0_2", "qasper:1802.05574:p2_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 086
- Case ID: `case-086`
- Question: What is the TREC-CAR dataset?
- Reference forms for adjudication: ["in this dataset, the input query is the concatenation of a Wikipedia article title with the title of one of its section. The relevant passages are the paragraphs within that section"]
### Arm A
- Answer: A dataset where input queries are Wikipedia article titles concatenated with section titles, and relevant passages are paragraphs within those sections.
- Selected passage IDs: ["qasper:1901.04085:p3_0", "qasper:1901.04085:p3_2", "qasper:1901.04085:p3_1", "qasper:1901.04085:p1_0", "qasper:1812.06705:p7_5"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: A dataset where input queries are Wikipedia article titles concatenated with section titles, and relevant passages are paragraphs within those sections.
- Selected passage IDs: ["qasper:1901.04085:p3_0", "qasper:1901.04085:p1_0", "qasper:1901.04085:p3_1", "qasper:1901.04085:p3_2", "qasper:1705.04153:p8_8"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 087
- Case ID: `case-087`
- Question: What previously proposed methods is this method compared against?
- Reference forms for adjudication: ["Maximum Entropy classifiers", "Naive Bayes", "SVM"]
### Arm A
- Answer: 1TON and 1TON+
- Selected passage IDs: ["qasper:1706.00139:p12_0", "qasper:1709.06671:p0_10", "qasper:1709.06671:p8_3", "qasper:1808.03738:p7_2", "qasper:1909.08824:p7_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: fine tuning and multi domain methods
- Selected passage IDs: ["qasper:1909.07575:p13_0", "qasper:1709.06671:p0_10", "qasper:1810.05334:p4_5", "qasper:1908.11860:p18_0", "qasper:1701.03214:p11_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 088
- Case ID: `case-088`
- Question: What is the criteria for a good metric?
- Reference forms for adjudication: ["The first one is that it should be highly correlated with human judgement of similarity. The second one is that it should be able to distinguish sentences which are in logical contradiction, logically unrelated or in logical agreement. The third one is that a robust evaluator should also be able to identify unintelligible sentences. The last criteria is that a good evaluation metric should not give high scores to semantically distant sentences and low scores to semantically related sentences."]
### Arm A
- Answer: Highly correlated with human judgement, distinguishes logical relations, identifies unintelligible sentences, and avoids scoring semantically distant sentences highly or related ones low.
- Selected passage IDs: ["qasper:1909.09268:p10_0", "qasper:2003.04866:p5_2", "qasper:1906.01910:p8_0", "qasper:2003.08529:p8_1", "qasper:2001.07820:p0_4"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: Highly correlated with human judgement, distinguishes logical relations, identifies unintelligible sentences, and avoids rewarding semantically distant sentences.
- Selected passage IDs: ["qasper:1909.09268:p10_0", "qasper:1708.09609:p4_0", "qasper:2002.05058:p7_1", "qasper:2003.04866:p5_2", "qasper:2001.07820:p0_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 089
- Case ID: `case-089`
- Question: Which languages are evaluated?
- Reference forms for adjudication: ["Amharic", "Arabic", "Bengali", "English, German, Spanish, Dutch", "Hindi", "Somali"]
### Arm A
- Answer: English, French, German, Russian, Chinese, Swahili, Urdu, Arabic, Hindi, Vietnamese, Basque, Hungarian, Italian, Navajo, Turkish, Quechua, and others including romanized Hindi and traditional Chinese.
- Selected passage IDs: ["qasper:1911.02116:p4_1", "qasper:2002.07306:p8_0", "qasper:1604.05372:p0_4", "qasper:1911.10049:p5_0", "qasper:1910.05456:p9_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: French, Russian, Arabic, Chinese, Hindi, Vietnamese, English, German, Swahili, Urdu, Spanish, Dutch, Korean, Japanese
- Selected passage IDs: ["qasper:2002.07306:p8_0", "qasper:1911.02116:p4_1", "qasper:1911.10049:p5_0", "qasper:1910.07481:p4_2", "qasper:1910.11768:p4_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 090
- Case ID: `case-090`
- Question: What training data did they use?
- Reference forms for adjudication: ["NLM-180", "Training-22"]
### Arm A
- Answer: A dataset that maps pictures with questions, and an open-domain conversation dataset
- Selected passage IDs: ["qasper:2002.11268:p7_0", "qasper:1910.11949:p5_0", "qasper:1903.11283:p1_6", "qasper:1710.06536:p14_16", "qasper:2004.01670:p2_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: A dataset that maps pictures with questions, and an open-domain conversation dataset
- Selected passage IDs: ["qasper:2002.11268:p7_0", "qasper:1910.11949:p5_0", "qasper:1710.06536:p14_16", "qasper:1810.09774:p3_2", "qasper:1710.06536:p14_13"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 091
- Case ID: `case-091`
- Question: Did they experimnet in other languages?
- Reference forms for adjudication: ["yes"]
### Arm A
- Answer: Yes
- Selected passage IDs: ["qasper:1708.09157:p6_0", "qasper:1805.03122:p4_0", "qasper:1906.00378:p10_0", "qasper:1909.00326:p7_0", "qasper:1708.01464:p9_2"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: Yes
- Selected passage IDs: ["qasper:1708.09157:p6_0", "qasper:1906.00378:p10_0", "qasper:1805.03122:p4_0", "qasper:1906.10519:p12_0", "qasper:1909.00326:p7_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 092
- Case ID: `case-092`
- Question: what standard speech transcription pipeline was used?
- Reference forms for adjudication: ["pipeline that is used at Microsoft for production data"]
### Arm A
- Answer: two-pass transcription with error correction
- Selected passage IDs: ["qasper:1908.08717:p6_0", "qasper:1708.08615:p1_1", "qasper:1708.08615:p0_1", "qasper:1912.01772:p5_0", "qasper:2003.01472:p1_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: two-pass transcription
- Selected passage IDs: ["qasper:1708.08615:p1_1", "qasper:1708.08615:p0_1", "qasper:2003.01472:p1_0", "qasper:1908.08717:p6_0", "qasper:1912.01772:p5_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 093
- Case ID: `case-093`
- Question: What type and size of word embeddings were used?
- Reference forms for adjudication: ["200 as the dimension of the obtained word vectors", "word2vec"]
### Arm A
- Answer: 300-dimensional GloVe word embeddings
- Selected passage IDs: ["qasper:1901.03866:p4_1", "qasper:1611.03599:p8_0", "qasper:1704.02686:p9_3", "qasper:1611.04361:p5_0", "qasper:1912.02481:p11_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: 300-dimensional GloVe embeddings
- Selected passage IDs: ["qasper:1901.03866:p4_1", "qasper:1908.05434:p14_0", "qasper:1710.06536:p14_24", "qasper:1911.00547:p6_2", "qasper:1912.02481:p11_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 094
- Case ID: `case-094`
- Question: How do they match words before reordering them?
- Reference forms for adjudication: ["CFILT-preorder system"]
### Arm A
- Answer: They match words based on their relations in the dependency parse tree, focusing on head, sibling, and child words.
- Selected passage IDs: ["qasper:1608.01084:p0_0", "qasper:1608.01084:p2_3", "qasper:1811.00383:p8_2", "qasper:1808.03738:p2_2", "qasper:1903.12495:p6_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: They match words based on their relationships in the dependency parse tree, specifically head, sibling, and child words.
- Selected passage IDs: ["qasper:1608.01084:p2_3", "qasper:1811.00383:p8_2", "qasper:1808.03738:p2_2", "qasper:1608.01084:p8_5", "qasper:1803.02839:p3_21"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 095
- Case ID: `case-095`
- Question: Why is this work different from text-only UNMT?
- Reference forms for adjudication: ["the image can play the role of a pivot “language\" to bridge the two languages without paralleled corpus"]
### Arm A
- Answer: It uses images as a pivot language to align languages, unlike text-only UNMT which is fundamentally ill-posed due to ambiguous associations.
- Selected passage IDs: ["qasper:1811.11365:p0_4", "qasper:1909.01383:p19_2", "qasper:1811.11365:p10_4", "qasper:1811.11365:p12_5", "qasper:1909.00279:p1_1"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: It uses multi-modal attention and images as additional information to improve translation, unlike text-only UNMT which relies solely on text data.
- Selected passage IDs: ["qasper:1811.11365:p0_4", "qasper:1811.11365:p1_1", "qasper:1811.11365:p10_4", "qasper:1811.11365:p13_0", "qasper:1811.11365:p12_5"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 096
- Case ID: `case-096`
- Question: Do they test their approach on large-resource tasks?
- Reference forms for adjudication: ["yes"]
### Arm A
- Answer: Yes
- Selected passage IDs: ["qasper:1805.07513:p6_2", "qasper:1601.01705:p6_0", "qasper:1812.10860:p2_0", "qasper:1909.06434:p6_3", "qasper:1708.01464:p12_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: Yes
- Selected passage IDs: ["qasper:1708.01464:p5_2", "qasper:1909.04493:p0_9", "qasper:1708.01464:p12_0", "qasper:1809.04686:p0_1", "qasper:1912.13337:p0_4"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 097
- Case ID: `case-097`
- Question: What context do they use?
- Reference forms for adjudication: ["screen name of the user", "title of the news article"]
### Arm A
- Answer: natural context
- Selected passage IDs: ["qasper:1809.01060:p1_9", "qasper:2001.07786:p15_0", "qasper:1809.01060:p1_7", "qasper:1902.09393:p9_4", "qasper:1912.09713:p26_8"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: natural context
- Selected passage IDs: ["qasper:2001.07786:p15_0", "qasper:1906.07662:p8_0", "qasper:1809.01060:p1_9", "qasper:1809.01060:p1_7", "qasper:1912.09152:p3_9"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 098
- Case ID: `case-098`
- Question: What datasets do they use?
- Reference forms for adjudication: ["AI2 BIBREF2", "CC BIBREF19", "IL BIBREF4", "MAWPS BIBREF20"]
### Arm A
- Answer: TRAC 2018 Dataset and Kaggle Dataset
- Selected passage IDs: ["qasper:2001.05493:p9_0", "qasper:1710.06536:p14_13", "qasper:1910.10781:p5_0", "qasper:1705.04153:p8_5", "qasper:1609.01962:p3_26"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: TRAC 2018 Dataset and Kaggle Dataset
- Selected passage IDs: ["qasper:2001.05493:p9_0", "qasper:1912.08320:p35_81", "qasper:1909.02560:p10_0", "qasper:1609.01962:p3_26", "qasper:1910.10781:p5_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 099
- Case ID: `case-099`
- Question: What dataset was used in this work?
- Reference forms for adjudication: ["How2"]
### Arm A
- Answer: Persian movie reviews from 2014 to 2016
- Selected passage IDs: ["qasper:1808.05077:p2_0", "qasper:1905.07471:p5_0", "qasper:2004.04315:p0_3", "qasper:2001.05493:p9_0", "qasper:1802.05574:p3_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: Persian movie reviews from 2014 to 2016
- Selected passage IDs: ["qasper:1808.05077:p2_0", "qasper:2001.05493:p9_0", "qasper:1912.08320:p35_81", "qasper:1909.02560:p10_0", "qasper:1910.10781:p8_0"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:

## Case 100
- Case ID: `case-100`
- Question: What clustering algorithm is used on top of the VerbNet-specialized representations?
- Reference forms for adjudication: ["MNCut spectral clustering algorithm BIBREF58"]
### Arm A
- Answer: MNCut spectral clustering
- Selected passage IDs: ["qasper:1707.06945:p2_0", "qasper:1701.00185:p8_0", "qasper:1701.09123:p1_4", "qasper:1907.04152:p6_0", "qasper:2003.06651:p7_10"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
### Arm B
- Answer: MNCut spectral clustering
- Selected passage IDs: ["qasper:1707.06945:p2_0", "qasper:1701.09123:p8_16", "qasper:1701.09123:p8_2", "qasper:1908.10422:p4_2", "qasper:1701.09123:p8_19"]
- Input passages are identified by the canonical IDs above; policy identity is intentionally withheld.
- Reviewer fields: supported? partially supported? unsupported? formatting/alias issue? abstention justified? notes:
