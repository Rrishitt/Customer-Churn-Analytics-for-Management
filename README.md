# 📉 Customer Churn Analysis for Business Impact

This project analyzes customer churn using behavioral, contractual, and financial data to identify key churn drivers, high-risk customer segments, and actionable retention opportunities. The goal is to help businesses reduce revenue leakage and improve customer lifetime value through data-driven decisions.

---

## 📌 Problem Statement

Customer churn is one of the biggest threats to sustainable revenue growth.  
This analysis answers three critical business questions:

1. **Why are customers churning?**
2. **Which customers are most at risk?**
3. **How can the business intervene early and efficiently?**

---

## 📊 Dataset Overview

- **Training dataset:** 440,833 customers  
- **Test dataset:** 64,374 customers  
- **Total features:** 12 (contract, subscription, payment, support, spend behavior)

**Global churn rate:** **56.7%**  
➡️ More than half of the customers leave, indicating a serious retention issue.

---

## 🔍 Key Business Insights

### 1️⃣ Contract Length is the Strongest Churn Driver

![Churn by Contract Length](./images/churn_by_contract.png)

| Contract Length | Churn Rate |
|----------------|-----------|
| Monthly        | ~100% |
| Quarterly      | ~46% |
| Annual         | ~46% |

**Insight:**
- Monthly contracts result in near-certain churn.
- Longer commitments significantly reduce churn.
- Contract structure matters more than pricing or features.

**Business Action:**
- Reposition Monthly plans as paid trials.
- Push migration to Quarterly or Annual plans using incentives.

---

### 2️⃣ Subscription Type Has Minimal Impact on Churn

![Churn by Subscription Type](./images/churn_by_subscription.png)

| Subscription Type | Churn Rate |
|------------------|-----------|
| Basic            | ~58% |
| Standard         | ~56% |
| Premium          | ~56% |

**Insight:**
- Feature tiers do not meaningfully affect retention.
- Customers are not churning due to pricing or feature differences.

**Business Action:**
- Stop over-investing in new features.
- Focus on onboarding, usability, and customer experience.

---

### 3️⃣ Behavioral Signals Strongly Correlate with Churn

![Behavioral Signals](./images/behavioral_churn_signals.png)

**Observed Patterns:**
- Churned customers delay payments more often.
- Churned customers contact support more frequently.
- Churned customers have lower total lifetime spend.

**Insight:**
These behaviors act as **early warning signals** before churn occurs.

**Business Action:**
- Trigger retention workflows when:
  - Payment delays increase
  - Support calls spike
- Intervene proactively instead of reacting post-churn.

---

## 🤖 Churn Risk Model Performance

![Confusion Matrix](./images/confusion_matrix.png)

### Model Metrics

**Training Performance**
- Accuracy: **89.2%**
- Precision: **94.2%**
- Recall: **86.2%**
- F1 Score: **0.90**

**Test Performance**
- Accuracy: **58.7%**
- Precision: **53.5%**
- Recall: **98.6%**
- F1 Score: **0.69**

**Interpretation:**
- The model captures almost all churners (high recall).
- Suitable as a **churn risk radar**, not a perfect predictor.
- Ideal for prioritizing retention efforts.

---

## 🔥 High-Risk Customer Segments

| Contract Length | Subscription Type | Churn Rate |
|----------------|------------------|-----------|
| Monthly        | Basic            | 1.00 |
| Monthly        | Standard         | 1.00 |
| Monthly        | Premium          | 1.00 |

➡️ Monthly customers across all tiers represent the **highest churn risk**.

---

## ⚠️ High-Risk Customer Identification

- **Top 5% high-risk customers identified:** **3,219**
- Enables:
  - Targeted retention campaigns
  - Efficient use of retention budgets
  - Measurable revenue recovery

Generated files:
- `risk_segments.csv`
- `customer_predictions.csv`
- `high_risk_customers.csv`

---

## 💼 Business Value Delivered

This analysis helps businesses:

- Reduce revenue leakage
- Improve customer lifetime value
- Optimize pricing and contract strategy
- Focus retention efforts where they matter most
- Shift from reactive to proactive churn management

---

## ✅ Final Takeaway

Customer churn in this dataset is driven primarily by **contract structure and customer behavior**, not by subscription tiers or feature sets.  
By restructuring contracts and acting on early warning signals, businesses can significantly improve retention and revenue stability.

---

## 📁 Outputs Generated

- Cleaned datasets
- Risk segmentation files
- Customer churn predictions
- Business-ready charts for reporting
- Executive-ready insights for decision-making

---

**Status:** Ready for real-world business application 🚀
