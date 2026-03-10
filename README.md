# NovaCred | Credit Application Governance Analysis

**DEGO 2606 | Data Ecosystems and Governance in Organizations**  
MSc Business Analytics  Nova SBE 2026

**Team Members**
Team 3: Catarina Palma; Mariana Almeida; León Fischer; Maria Silva
 
---

## Executive Summary

NovaCred is a fintech startup that uses machine learning to make credit decisions. Following a regulatory inquiry into potential discrimination in its lending practices, our team acted as a **Data Governance Task Force** to audit the raw credit application dataset across three dimensions: data quality, algorithmic bias, and privacy compliance.

**The audit confirms the regulatory concern — and goes further.**

| Finding | Verdict |
|---|---|
| Data quality | **15 issues** identified across all 4 dimensions; all remediated |
| Gender bias | **Direct discrimination confirmed**: AIR = 0.767, p < 0.001 |
| Age bias | **Indirect discrimination confirmed**: AIR = 0.616 via credit history proxy |
| Intersectional bias | **Severe**: Young female applicants AIR = 0.464, p = 0.0004 |
| GDPR compliance | **12 gaps** : 3 CRITICAL, 7 HIGH, 2 MEDIUM |
| EU AI Act status | **HIGH-RISK**: 8/10 mandatory requirements not evidenced |
| Deployment readiness |  **System cannot be lawfully deployed in the EU in its current state** |

**Immediate actions required before any deployment:** conduct a DPIA, remove gender from the model, implement rejection explanations, and pseudonymise all direct identifiers.

---

## Repository Structure

```
project-team3/
├── README.md                          # This file — executive summary & findings
├── data/
│   ├── raw/
│   │   └── raw_credit_applications.json   # Original dataset (502 records)
│   └── processed/
│       ├── applications_clean.csv         # Cleaned dataset (500 rows × 20 cols)
│       └── applications_privacy_safe.csv  # Privacy-safe analytics output
├── notebooks/
│   ├── 01-data-quality.ipynb             # Data Engineer: cleaning pipeline
│   ├── 02-bias-analysis.ipynb            # Data Scientist: fairness analysis
│   └── 03-privacy-demo.ipynb             # Governance Officer: GDPR & AI Act
└── presentation/ 
```

---

## Team

| Name | Student No. | Role | Responsibilities |
|---|---|---|---|
| Catarina Palma | Data Engineer | Data loading, cleaning pipeline, repository structure |
| Maria Silva | Data Scientist | Bias analysis, fairness metrics, statistical testing |
| León Fischer | Governance Officer | GDPR mapping, EU AI Act, policy recommendations |
| Mariana Almeida | Product Lead | Presentation, coordination, README |

---

## 1. Data Quality Analysis

**Notebook:** `notebooks/01-data-quality.ipynb`  
**Role:** Catarina Palma (Data Engineer)

### 1.1 Dataset Overview

The raw dataset contains **502 records × 21 columns** in a nested JSON format. After cleaning, the final dataset is **500 rows × 20 columns**. All expected schema fields were present at load time.

### 1.2 Issues Identified: 15 Total

We assessed quality across four dimensions: **Accuracy, Completeness, Consistency, and Validity**.

#### Accuracy (2 issues)

| Issue | Count | % of Records |
|---|---|---|
| Duplicate `_id` values | 4 rows (2 distinct IDs) | 0.80% |
| Duplicate SSN values | 6 rows (3 distinct SSNs) | 1.21% of non-missing |

The SSN duplicates are particularly concerning: some appear across records with **different applicant names**, indicating either data entry errors or potential identity fraud. Both cases represent silent compliance exposure under GDPR.

#### Completeness (4 issues)

| Field | Missing Count | Missing % | Notes |
|---|---|---|---|
| `notes` | 500 / 502 | 99.60% | Only populated for duplicate records — dropped |
| `loan_purpose` | 452 / 502 | 90.04% | Retained — analytically valuable |
| `processing_timestamp` | 440 / 502 | 87.65% | Retained — audit trail when present |
| Blank strings (email, DOB, gender, ZIP) | 14 total | — | Converted to `NaN` |

The near-total absence of `loan_purpose` and `processing_timestamp` makes **purpose-limitation compliance** (GDPR Art. 5(1)(b)) and **audit trails** impossible to enforce.

#### Consistency (4 issues)

| Issue | Count | % | Remediation |
|---|---|---|---|
| `annual_income` stored as string | 8 records | 1.61% | Cast to numeric |
| `applicant_info.gender` shorthand (`M`/`F`) | 111 records | 22.11% | Mapped to `Male`/`Female` |
| `date_of_birth` non-ISO format (slash variants) | 157 records | 31.27% | Standardised to `YYYY-MM-DD` |
| `annual_salary` duplicate column | 5 records | — | Column dropped |
| Category labels — mixed formats | 0 | 0% | No issue found |

Date-of-birth inconsistency was the most structurally significant: 20% of records used `DD/MM/YYYY` or `MM/DD/YYYY` slash notation, of which **39 records (7.77%) were ambiguous** when the day value was ≤ 12.

#### Validity (5 issues)

| Issue | Count | % | Remediation |
|---|---|---|---|
| `credit_history_months` < 0 | 2 | 0.40% | Set to `NaN` |
| `debt_to_income` > 1 | 1 | 0.20% | Set to `NaN` |
| `savings_balance` < 0 | 1 | 0.20% | Set to `NaN` |
| Future timestamps | 2 | 0.40% | Set to `NaN` |
| Invalid email format | 4 | 0.80% | Set to `NaN` |

`interest_rate` values above 1.0 were initially flagged but confirmed to be stored in **percentage points** (range: 2.5–6.5%) — not a validity error.

### 1.3 Remediation Philosophy

Our approach was deliberate and conservative:

- **No automatic imputation.** Missing values were preserved as `NaN`. Filling them would introduce artificial data not supported by the original records, potentially inflating apparent data quality.
- **No placeholder categories.** We did not introduce `"Unknown"` for missing fields like `loan_purpose` or `gender`. Such placeholders mix true missingness with a fabricated category, making downstream analysis less interpretable.
- **Flag, don't delete, duplicate SSNs.** Rows with conflicting SSNs were retained; the SSN field was set to `NaN` and a `ssn_is_duplicate = True` flag added. This preserves the application-level data while making the identifier unreliable explicit.

### 1.4 Clean Dataset Summary

| Metric | Value |
|---|---|
| Final shape | 500 rows × 20 columns |
| Duplicate `_id` resolved | 0 remaining |
| Duplicate SSN flagged | 4 rows flagged (`ssn_is_duplicate = True`) |
| Blank strings remaining | 0 |
| Unparseable DOB remaining | 0 |
| Invalid numeric values remaining | 0 |

---

## 2. Bias Detection & Fairness

**Notebook:** `notebooks/02-bias-analysis.ipynb`  
**Role:** Maria Silva (Data Scientist)

We followed a five-stage framework: **Adverse Impact Ratio (AIR) → Chi-Square testing → Logistic Regression → Proxy Analysis → Intersectional Analysis**.

### 2.1 Gender: Direct Discrimination Confirmed

| Metric | Value | Threshold | Result |
|---|---|---|---|
| Male approval rate | 66.0% (163/247) | — | Reference group |
| Female approval rate | 50.6% (127/251) | — | — |
| Adverse Impact Ratio (AIR) | **0.767** | < 0.80 = adverse impact | ADVERSE IMPACT |
| Chi-Square p-value | **0.0007** | < 0.05 = significant | Statistically significant |
| Regression coef. (gender) | **−0.715** | — | Female applicants penalised |
| Regression p-value | **< 0.001** | — | Significant after controls |

The logistic regression controls for annual income, credit history, savings balance, and debt-to-income ratio. **The gender coefficient remains large and significant even after controlling for all financial variables.** Male and female applicants have virtually identical financial profiles (income: p = 0.326; savings: p = 0.943; DTI: p = 0.296). There is no legitimate business justification. This is **direct discrimination**.

### 2.2 Age (18–30 Group): Indirect Discrimination via Proxy

| Metric | Value | Threshold | Result |
|---|---|---|---|
| 18–30 approval rate | 41.1% | — | Lowest of all groups |
| 60+ approval rate | 66.7% | — | Reference group |
| AIR (18–30) | **0.616** | < 0.80 | ADVERSE IMPACT |
| Chi-Square p-value | **0.0021** | < 0.05 | Statistically significant |
| Age coefficient in regression | −0.0059, p = 0.621 | — | Not directly significant |
| `credit_history_months` ↔ age | r = **0.651**, p < 0.001 | — | Confirmed proxy |

Age itself is not a statistically significant predictor in the regression (p = 0.621). However, `credit_history_months` — which accumulates naturally with age — is strongly correlated with age (r = 0.651) and is the **primary driver of rejections** (80.95% of rejections cite `algorithm_risk_score`, which incorporates credit history length). Using credit history as a scoring variable **structurally penalises younger applicants**. This is **indirect discrimination**.

### 2.3 Intersectional Analysis: Most Severe Pattern

Young female applicants belong simultaneously to both disadvantaged groups. The combined effect is worse than either attribute in isolation:

| Group | Approval Rate | AIR | Result |
|---|---|---|---|
| Male & 31–45 (reference) | 70.3% | 1.000 | ✅ |
| Female & 60+ | 65.0% | 0.924 | ✅ |
| Male & 46–60 | 67.9% | 0.966 | ✅ |
| Male & 60+ | 68.4% | 0.973 | ✅ |
| Female & 31–45 | 54.0% | 0.768 | ❌ |
| Female & 46–60 | 53.7% | 0.764 | ❌ |
| Male & 18–30 | 50.0% | 0.711 | ❌ |
| **Female & 18–30** | **32.7%** | **0.464** | ❌ **SEVERE** |

Chi-Square test: χ² = 26.765, df = 7, **p = 0.0004**. The compounded disadvantage is not attributable to chance.

### 2.4 Bias Conclusion

Direct gender discrimination and indirect age discrimination operate **simultaneously**. Both dimensions must be addressed in parallel — neither can be remediated in isolation.

---

## 3. Privacy & Governance

**Notebook:** `notebooks/03-privacy-demo.ipynb`  
**Role:** León Fischer — Governance Officer

### 3.1 PII Audit

The dataset contains **19 PII fields** out of 20 total columns. The only non-PII field is `_id`, which itself qualifies as a pseudonym.

| Risk Level | Fields | Count |
|---|---|---|
| CRITICAL | `applicant_info.ssn` | 1 |
| HIGH | `full_name`, `email`, `ip_address`, `date_of_birth`, `zip_code`, `gender`, spending categories | 7 |
| MEDIUM | `annual_income`, `credit_history_months`, `debt_to_income`, `savings_balance`, `loan_approved`, `rejection_reason`, `approved_amount` | 7 |
| LOW | `_id`, `interest_rate`, `loan_purpose`, `processing_timestamp` | 4 |

**Total live PII data points across key fields: 3,470.** Direct identifiers (name, email, SSN, IP) are nearly 100% populated. In a GDPR-compliant system, most of these fields would not be present in an operational analytics database at all.

### 3.2 Pseudonymisation Demonstrated

We applied the following technical measures to the analytics layer:

| Field | Technique | GDPR Basis | Re-identifiable? |
|---|---|---|---|
| SSN | **HMAC-SHA256** (keyed pseudonym) | Art. 4(5), Recital 26 | Yes — with key |
| Email | **SHA-256** one-way hash | Art. 25 | No |
| IP Address | **Last-octet truncation** (`/24`) | Recital 30 | No |
| Full Name | **Suppression** (dropped) | Art. 5(1)(c) | N/A |
| Date of Birth | **5-year age band** generalisation | Art. 25 | No |

Output: `data/processed/applications_privacy_safe.csv` — 19 columns, 500 rows, with all raw PII replaced by the above techniques.

### 3.3 GDPR Compliance Gap Analysis: 12 Gaps

| Severity | Article | Gap |
|---|---|---|
| CRITICAL | Art. 35 | **No DPIA conducted** — mandatory for large-scale automated credit scoring |
| CRITICAL | Art. 9 | **Special-category data risk** — `Alcohol`, `Gambling`, `Adult Entertainment` spending categories can act as proxies for health/addiction conditions |
| CRITICAL | Art. 22 | **Opaque automated decisions** — `algorithm_risk_score` drives 80.95% of rejections with no explanation provided |
| HIGH | Art. 5(1)(c) | **Data minimisation failure** — `full_name`, `email`, `ip_address`, `date_of_birth` stored in analytics layer without necessity |
| HIGH | Art. 5(1)(e) | **No retention policy** — no deletion timestamps, no archival flags; records stored indefinitely |
| HIGH | Art. 6 | **Lawful basis not documented** — no evidence of consent mechanisms or LIA for processing |
| HIGH | Art. 13/14 | **No privacy notice** — data subjects not informed at application stage |
| HIGH | Art. 17 | **Right to Erasure not technically feasible** — no mechanism to locate and delete all records for a given data subject across systems |
| HIGH | Art. 30 | **No Record of Processing Activities (RoPA)** — mandatory for large-scale processing |
| HIGH | Art. 32 | **No security measures documented** — no evidence of encryption at rest or access controls |
| MEDIUM | Art. 5(1)(a) | **Fairness not demonstrated** — system produces discriminatory outcomes (see Bias section) |
| MEDIUM | Art. 25 | **No Privacy by Design** — PII stored in operational analytics without technical controls |

### 3.4 EU AI Act Classification

Under **Regulation (EU) 2024/1689 (EU AI Act)**, credit scoring systems are **explicitly classified as HIGH-RISK** under **Annex III, Section 5(b)**.

| Requirement | Status |
|---|---|
| Conformity assessment | Not evidenced |
| Technical documentation | Not evidenced |
| Logging & audit trail | Incomplete (87.65% of timestamps missing) |
| Transparency to users | Not evidenced |
| Human oversight mechanism | Not evidenced |
| Accuracy & robustness testing | Not evidenced |
| Bias testing & monitoring | Performed (this project) |
| Data governance measures | Not evidenced |
| Registration in EU database | Not evidenced |
| Post-market monitoring plan | Not evidenced |

**8 of 10 mandatory requirements are NOT evidenced. This system cannot be lawfully deployed in the EU without significant remediation.**

---

## 4. Governance Recommendations

Based on the full audit, we propose 12 concrete controls organised by priority tier.

### P1: Critical, Before Any Deployment

| # | Control | GDPR / Legal Basis | Owner |
|---|---|---|---|
| 1 | **Conduct a mandatory DPIA** covering automated credit scoring, bias risk, and data flows | Art. 35 / EU AI Act Art. 9 | DPO |
| 2 | **Replace `algorithm_risk_score` with meaningful rejection explanations** — specific, individualisable reasons for each denial | Art. 22 | Engineering |
| 3 | **Remove `applicant_info.gender` from all model inputs and feature sets** — AIR = 0.767, coef = −0.715, p < 0.001 confirms direct discrimination | Equality law + Art. 5(1)(a) | Data Science |
| 4 | **Establish a human review pathway** for borderline and high-value decisions | EU AI Act Art. 14 | Operations |

### P2: High,  Within 30 Days

| # | Control | GDPR / Legal Basis | Owner |
|---|---|---|---|
| 5 | **Pseudonymise all direct identifiers** in the analytics layer using the techniques demonstrated in Notebook 03 | Art. 25 | Engineering |
| 6 | **Implement a data retention policy** with automatic purge after the legally required period; add deletion timestamps to all records | Art. 5(1)(e) | Data Engineering |
| 7 | **Document lawful basis in a RoPA** for each processing activity (SSN, IP, spending behaviour, credit decisions) | Art. 30 | Legal / DPO |
| 8 | **Publish a privacy notice** at the application stage disclosing the automated decision-making logic and data subject rights | Art. 13 | Legal |

### P3: Medium, Within 90 Days

| # | Control | GDPR / Legal Basis | Owner |
|---|---|---|---|
| 9 | **Address age discrimination** — audit whether `credit_history_months` can be replaced with relative measures that do not structurally disadvantage 18–30 applicants | Equality law | Data Science |
| 10 | **Monitor the credit history proxy** — implement ongoing correlation monitoring between age and model features; r = 0.651 requires active management | EU AI Act Art. 9(7) | MLOps |
| 11 | **Register the system in the EU AI Act database** and complete the conformity assessment documentation | EU AI Act Art. 48–49 | Compliance |
| 12 | **Establish continuous fairness monitoring** — monthly Disparate Impact ratio reporting across gender and age groups; alert threshold at AIR < 0.85 | Art. 5(1)(a) + EU AI Act | Data Science |

---

## 5. Key Metrics Reference

| Category | Metric | Value |
|---|---|---|
| **Dataset** | Raw records | 502 |
| | Clean records | 500 |
| | Columns | 20 |
| **Data Quality** | Total issues identified | 16 |
| | Duplicate `_id` | 4 (0.80%) |
| | Duplicate SSN | 6 (1.21%) |
| | Missing `loan_purpose` | 452 (90.04%) |
| | Missing `processing_timestamp` | 440 (87.65%) |
| | DOB format inconsistencies | 157 (31.27%) |
| | Gender coding inconsistencies | 111 (22.11%) |
| **Gender Bias** | Male approval rate | 66.0% |
| | Female approval rate | 50.6% |
| | AIR (female) | **0.767** |
| | Chi-Square p-value | 0.0007 |
| | Regression coefficient | −0.715 (p < 0.001) |
| **Age Bias** | 18–30 approval rate | 41.1% |
| | AIR (18–30) | **0.616** |
| | Age proxy (credit history) | r = 0.651, p < 0.001 |
| **Intersectional** | Female & 18–30 approval rate | 32.7% |
| | AIR (Female & 18–30) | **0.464** |
| | Chi-Square p-value | 0.0004 |
| **Privacy** | PII fields | 19 / 20 |
| | CRITICAL risk fields | 1 (SSN) |
| | HIGH risk fields | 7 |
| | Live PII data points | 3,470 |
| **GDPR** | Total compliance gaps | 12 |
| | CRITICAL gaps | 3 |
| | HIGH gaps | 7 |
| | MEDIUM gaps | 2 |
| **EU AI Act** | Classification | HIGH-RISK (Annex III §5b) |
| | Requirements not evidenced | 8 / 10 |

---

## 6. How to Run

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels
```

### Execution Order

Run the notebooks in sequence — each notebook saves an output file used by the next:

```
01-data-quality.ipynb
    → saves: data/processed/applications_clean.csv

02-bias-analysis.ipynb
    → reads:  data/processed/applications_clean.csv

03-privacy-demo.ipynb
    → reads:  data/processed/applications_clean.csv
    → saves:  data/processed/applications_privacy_safe.csv
```

All notebooks are self-contained and runnable from the `notebooks/` directory. The path resolution logic handles both cases (running from repo root or from within `notebooks/`).

## 7. Link for the presentation video
https://youtu.be/96akD78Yahk

---

## 7. References

- GDPR (EU) 2016/679 — https://gdpr.eu/
- EU AI Act (EU) 2024/1689 — https://artificialintelligenceact.eu/
- EEOC Uniform Guidelines on Employee Selection Procedures (4/5ths Rule) — 29 C.F.R. § 1607
- Sweeney, L. (2000). *Simple Demographics Often Identify People Uniquely.* Carnegie Mellon University
- EDPB Guidelines 01/2022 on Data Subject Rights — Article 17
- NIST SP 800-122 — Guide to Protecting the Confidentiality of PII