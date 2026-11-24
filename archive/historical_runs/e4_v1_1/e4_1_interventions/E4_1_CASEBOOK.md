# E4-1 Casebook

All answer labels below are deterministic reference proxies, not independently verified factual judgments.

## multihop:0001
Question: <benchmark text omitted>

Reference (offline only): Donald Trump

```json
{
  "A_JINA_TOP5": {
    "answer": "Donald Trump",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "Donald Trump",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "Donald Trump",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "Donald Trump",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0015
Question: <benchmark text omitted>

Reference (offline only): European Commission

```json
{
  "A_JINA_TOP5": {
    "answer": "European Union",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "B_PPLX_TOP5": {
    "answer": "European Union",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "European Union",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "European Union",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "European Union",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0017
Question: <benchmark text omitted>

Reference (offline only): no

```json
{
  "A_JINA_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "No.",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0041
Question: <benchmark text omitted>

Reference (offline only): no

```json
{
  "A_JINA_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0056
Question: <benchmark text omitted>

Reference (offline only): Tyreek Hill

```json
{
  "A_JINA_TOP5": {
    "answer": "Tyreek Hill",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "Justin Jefferson",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "Rashee Rice",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "Rashee Rice",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0058
Question: <benchmark text omitted>

Reference (offline only): Yes

```json
{
  "A_JINA_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "B_PPLX_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0075
Question: <benchmark text omitted>

Reference (offline only): Federal Reserve

```json
{
  "A_JINA_TOP5": {
    "answer": "The Reserve Bank of Australia",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "Federal Reserve",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "Reserve Bank",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "Reserve Bank",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "Reserve Bank",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0086
Question: <benchmark text omitted>

Reference (offline only): no

```json
{
  "A_JINA_TOP5": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "B_PPLX_TOP5": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0094
Question: <benchmark text omitted>

Reference (offline only): Yes

```json
{
  "A_JINA_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "No.",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "No",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0119
Question: <benchmark text omitted>

Reference (offline only): no

```json
{
  "A_JINA_TOP5": {
    "answer": "Yes",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "B_PPLX_TOP5": {
    "answer": "Yes",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "Yes",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0130
Question: <benchmark text omitted>

Reference (offline only): Agree

```json
{
  "A_JINA_TOP5": {
    "answer": "The TechCrunch coverage agrees that Sam Bankman-Fried is facing seven charges: two counts of wire fraud and five conspiracy charges.",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "B_PPLX_TOP5": {
    "answer": "The TechCrunch coverage agrees on the number of fraud and conspiracy charges, stating seven counts in multiple articles.",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "The TechCrunch coverage agrees that Sam Bankman-Fried is facing seven counts of fraud and conspiracy.",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "There is no discrepancy; all articles agree he faces seven charges of fraud and conspiracy.",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "The TechCrunch coverage agrees that Sam Bankman-Fried is facing seven charges of fraud and conspiracy.",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0138
Question: <benchmark text omitted>

Reference (offline only): The Federal Reserve

```json
{
  "A_JINA_TOP5": {
    "answer": "Reserve Bank of Australia",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "Reserve Bank",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "Reserve Bank of Australia",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "Reserve Bank of Australia",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "Reserve Bank",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0146
Question: <benchmark text omitted>

Reference (offline only): no

```json
{
  "A_JINA_TOP5": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "B_PPLX_TOP5": {
    "answer": "No.",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0148
Question: <benchmark text omitted>

Reference (offline only): no

```json
{
  "A_JINA_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0161
Question: <benchmark text omitted>

Reference (offline only): Yes

```json
{
  "A_JINA_TOP5": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0162
Question: <benchmark text omitted>

Reference (offline only): no

```json
{
  "A_JINA_TOP5": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0168
Question: <benchmark text omitted>

Reference (offline only): no

```json
{
  "A_JINA_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "B_PPLX_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0203
Question: <benchmark text omitted>

Reference (offline only): Google

```json
{
  "A_JINA_TOP5": {
    "answer": "Google",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "Google",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "Google",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "Amazon",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "Google",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "Google",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0214
Question: <benchmark text omitted>

Reference (offline only): Both

```json
{
  "A_JINA_TOP5": {
    "answer": "Sport Grill",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "Sport Grill",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "Sport Grill",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "Sport Grill",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "Sport Grill",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0221
Question: <benchmark text omitted>

Reference (offline only): Yes

```json
{
  "A_JINA_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "B_PPLX_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0223
Question: <benchmark text omitted>

Reference (offline only): Sam Bankman-Fried

```json
{
  "A_JINA_TOP5": {
    "answer": "Sam Bankman-Fried",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "B_PPLX_TOP5": {
    "answer": "Sam Bankman-Fried",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "Sam Bankman-Fried",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "Sam Bankman-Fried",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "Sam Bankman-Fried",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "Sam Bankman-Fried",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0241
Question: <benchmark text omitted>

Reference (offline only): Yes

```json
{
  "A_JINA_TOP5": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0244
Question: <benchmark text omitted>

Reference (offline only): no

```json
{
  "A_JINA_TOP5": {
    "answer": "Yes",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "Yes",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0248
Question: <benchmark text omitted>

Reference (offline only): Golden State Warriors

```json
{
  "A_JINA_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0251
Question: <benchmark text omitted>

Reference (offline only): OpenAI

```json
{
  "A_JINA_TOP5": {
    "answer": "OpenAI",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "B_PPLX_TOP5": {
    "answer": "OpenAI",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "OpenAI",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "OpenAI",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "OpenAI",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0256
Question: <benchmark text omitted>

Reference (offline only): OpenAI

```json
{
  "A_JINA_TOP5": {
    "answer": "OpenAI",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "OpenAI",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "OpenAI",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "OpenAI",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "OpenAI",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "OpenAI",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0260
Question: <benchmark text omitted>

Reference (offline only): no

```json
{
  "A_JINA_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0265
Question: <benchmark text omitted>

Reference (offline only): Sam Bankman-Fried

```json
{
  "A_JINA_TOP5": {
    "answer": "Sam Bankman-Fried",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "B_PPLX_TOP5": {
    "answer": "Sam Bankman-Fried",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "Sam Bankman-Fried",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "Sam Bankman-Fried",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "Sam Bankman-Fried",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0268
Question: <benchmark text omitted>

Reference (offline only): Google

```json
{
  "A_JINA_TOP5": {
    "answer": "Google",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "B_PPLX_TOP5": {
    "answer": "Google",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "Google",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "Google",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "Google",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0272
Question: <benchmark text omitted>

Reference (offline only): Yes

```json
{
  "A_JINA_TOP5": {
    "answer": "No",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "B_PPLX_TOP5": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0290
Question: <benchmark text omitted>

Reference (offline only): Sam Bankman-Fried

```json
{
  "A_JINA_TOP5": {
    "answer": "Sam Bankman-Fried",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "B_PPLX_TOP5": {
    "answer": "Sam Bankman-Fried",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "Sam Bankman-Fried",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "Sam Bankman-Fried",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "Sam Bankman-Fried",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "Sam Bankman-Fried",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0314
Question: <benchmark text omitted>

Reference (offline only): no

```json
{
  "A_JINA_TOP5": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0320
Question: <benchmark text omitted>

Reference (offline only): Yes

```json
{
  "A_JINA_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0334
Question: <benchmark text omitted>

Reference (offline only): Yes

```json
{
  "A_JINA_TOP5": {
    "answer": "No",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "No",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "No",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "No",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0340
Question: <benchmark text omitted>

Reference (offline only): Yes

```json
{
  "A_JINA_TOP5": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "No",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0342
Question: <benchmark text omitted>

Reference (offline only): Yes

```json
{
  "A_JINA_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "B_PPLX_TOP5": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0345
Question: <benchmark text omitted>

Reference (offline only): Yes

```json
{
  "A_JINA_TOP5": {
    "answer": "No",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "No",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "No",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "No",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0346
Question: <benchmark text omitted>

Reference (offline only): Yes

```json
{
  "A_JINA_TOP5": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "B_PPLX_TOP5": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "Yes",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "Yes, all three articles address different legal issues.",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "Yes, all three articles address different legal issues.",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0347
Question: <benchmark text omitted>

Reference (offline only): no

```json
{
  "A_JINA_TOP5": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "B_PPLX_TOP5": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "No",
    "label": "REFERENCE_MATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```

## multihop:0366
Question: <benchmark text omitted>

Reference (offline only): True

```json
{
  "A_JINA_TOP5": {
    "answer": "Yes",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "B_PPLX_TOP5": {
    "answer": "Yes",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "C_ANNOTATED_ORACLE": {
    "answer": "Yes",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "D_SUPPORT_REMOVAL": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  },
  "D_CONTROL_NEUTRAL_SWAP": {
    "answer": "Yes",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": true
  },
  "E_NO_EVIDENCE": {
    "answer": "INSUFFICIENT_EVIDENCE",
    "label": "REFERENCE_MISMATCH_PROXY",
    "support_complete": false
  }
}
```
