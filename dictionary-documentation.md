# Dictionary Documentation

## Overview
A keyword-based filter dictionary for identifying reputational risk disclosures in SEC 10-K filings.

- 14 categories, 358 keywords total
- Stored in: `keyword_categories.json`

## Categories
1. Cybersecurity & Data Privacy
2. Financial Performance & Fraud
3. Product & Service Quality
4. Production Failure
5. Trademark/Brand Erosion
6. Human Resources
7. Health & Safety
8. Governance Risk
9. Labor & Human Rights
10. Data/Privacy Breaches
11. Environmental Misconduct
12. Supply Chain & Third-Party Risk
13. Legal & Litigation
14. Credit, Liquidity & Market Risk

## Sources
Keywords were derived from the following literature:
- Tonello (2007): risk category taxonomy and stakeholder groups
- Zhu et al. (2022): reputational risk drivers from 10-K text disclosures
- Rajagopal et al. (2022): STAR method risk classification
- Perera et al. (2022): cyberattack reputational damage factors
- Hanay et al. (2024): fuzzy modeling reputational risk factors in finance

## Approach

### Step 1: Literature Extraction
Risk categories and associated keywords were extracted manually from each of the five source papers.

### Step 2: Consolidation
The five keyword sets were merged into 14 coherent top-level categories by:
- Collapsing overlapping categories from different papers into a single parent category (e.g. Tonello's "Health & Safety" and Zhu's "Risk of workplace culture" merged into one)
- Removing redundant or overly generic terms that would match irrelevant paragraphs (e.g. bare words like "risk" or "failure")
- Deduplicating keywords that appeared across multiple source categories
- Dropping case-study sub-categories from Tonello (BP, Nike, Mattel, Sony, J&J, Martha Stewart) whose keywords were absorbed into the relevant parent categories

**AI Prompt used:**
> I have extracted all the keywords and categories from the relevant papers. Further clean it to make a final keyword_categories.json by:
> - Merge overlapping sources, consolidate into coherent top-level categories
> - Remove redundant/too-generic terms
> - Deduplicate across categories
> - Enrich each category with multi-word phrases that actually appear in 10-K filings
> - Drop case-study sub-categories

### Step 3: Enrichment via LLM and 10-K Review
Keywords were extended by prompting an LLM to suggest additional phrases companies use when disclosing each risk type in annual filings, cross-referenced against actual SEC 10-K Item 1A language. This step added filing-specific terminology such as FCPA, going concern, denial-of-service attack, material weakness, and Superfund liability.

**AI Prompt used:**
> Below is a JSON dictionary where each key is a reputational risk category and each value is a list of keyword phrases used to identify that risk in annual report disclosures.
> Your task is to extend the keyword lists only — do not add, remove, or rename any categories.
> For each category, add additional multi-word phrases that:
> - Actually appear in SEC 10-K Item 1A risk factor disclosures
> - Are specific enough to identify the risk type (no generic single words like "risk" or "failure" alone)
> - Cover natural paraphrasing variations companies use across different industries and filing years
> - Do not duplicate any existing keywords
>
> Return the full updated JSON only, with no explanation or markdown formatting.

## Output
14 categories, 358 keywords stored in `keyword_categories.json`. Each keyword is a multi-word phrase chosen to be specific enough to identify the risk type while covering natural paraphrasing across companies and filing years.


---

## Raw Extracted Keywords (Per Source)

### Tonello (2007)
```json
{
  "Lack of coherent business strategy": ["strategy failure", "strategic change", "business direction"],
  "Financial performance": ["earnings decline", "revenue loss", "financial underperformance"],
  "Product/service quality": ["product recall", "product defect", "quality failure"],
  "Production failure": ["production shutdown", "capacity failure", "operational disruption"],
  "Trademark/Brand erosion": ["brand damage", "brand erosion", "trademark infringement"],
  "Human resources": ["employee misconduct", "talent retention", "workforce reduction"],
  "Health & safety": ["safety incident", "workplace injury", "environmental accident"],
  "Governance risk": ["compliance failure", "ethics violation", "regulatory breach"],
  "Harassment/discrimination": ["workplace harassment", "discriminatory practices"],
  "Whistleblower retaliation": ["whistleblower", "retaliation"],
  "Bribery/corruption": ["bribery", "corruption", "FCPA"],
  "Financial fraud": ["accounting fraud", "financial misstatement"],
  "Labor/human rights": ["child labor", "human rights", "labor standards"],
  "Data/privacy breaches": ["data breach", "privacy violation", "cybersecurity"],
  "Environmental misconduct": ["environmental violation", "emissions standards"],
  "Investors/Banks": ["credit rating", "investor confidence", "capital access"],
  "Customers": ["customer satisfaction", "product liability", "brand loyalty"],
  "Regulators": ["regulatory scrutiny", "compliance requirement", "government investigation"],
  "Activists/Communities": ["ESG", "activist pressure", "community opposition", "boycott"],
  "Employees": ["talent acquisition", "key personnel", "workforce risk"]
}
```

### Zhu et al. (2022)
```json
{
  "Inadequate information safeguards": ["Leakage and loss", "Internal or customers' data", "Confidential or proprietary information"],
  "System interruptions": ["Disruption of business", "System failures", "Hardware and software failures", "Telecommunication problems", "Security breach"],
  "Litigation risk": ["Making or defending a claim in court", "Significant financial losses"],
  "Compliance risk": ["Penalties", "Failure to comply with regulations", "Regulatory investigations"],
  "Human error": ["Employees' errors or misconduct", "Failed transaction processing", "Miscommunication", "Data entry, maintenance or loading error", "Missing deadline or responsibility", "Legal liability and regulatory scrutiny"],
  "Partners' performance": ["Business cooperation with partners", "Dependence on third-party platforms", "Uncontrollability of franchise operation", "Contractual obligations"],
  "Conflicts of interest": ["Conflict of interest between stakeholders", "Clients withdraw funds"],
  "Investment risk": ["Uncertainties or losses", "Future operations and financial activities", "New businesses", "Strategic investments or acquisitions"],
  "Product and service problems": ["Defective or unsatisfactory", "Additional cost", "Loss of profit opportunities"],
  "Fraud": ["Defraud", "Misappropriate property", "Circumvent regulations, the law, or company policy", "Mortgage fraud"],
  "Loss of professionals": ["Brain drain", "Professional and managerial personnel", "Top executives", "Senior professionals"],
  "Credit risk": ["Fail to meet financial obligations", "Default risk", "Bankruptcy", "Downgrade risk", "Settlement risk"],
  "Liquidity risk": ["Failure to pay down or refinance debts", "Satisfy cash obligations", "Fund capital withdraws", "Funding liquidity risk", "Trading liquidity risk"]
}
```

### Rajagopal et al. (2022)
```json
{
  "Financial risk": ["inability to meet the targeted financial performance", "poor likelihood of continuing profitability", "downsized future growth prospects", "non-performing financial assets"],
  "Quality risk": ["substandard or low-grade composition products", "risks to consumers' health and safety", "non-adherence to regulations of quality standards", "law-suits and product recalls"],
  "Risk of business disruption": ["business disruptions and system outage proliferation", "application crashes and network outages", "human errors", "lack of drug efficacy or adverse drug reactions"],
  "Risk of a security breach due to IT system failure": ["security breach due to IT system failure", "loss of customers' confidential data and records", "malware and cyber-attacks", "data loss from failed backup or restore"],
  "Risk of workplace culture": ["lack of concern for the health and well-being of employees", "unsafe workplace and unfair discrimination", "wages below standard", "lack of motivation and empowerment", "union disputes and job dissatisfaction", "labor strike and factory lockout"],
  "Risk of unethical governance": ["inconsistency in reaching a trade-off between economic objectives and social obligations", "poor leadership and unethical practices", "bribery, coercion, and insider trading", "executives' or employees' misconduct", "forging records", "non-sterile manufacturing facility"],
  "Internal coordination risk during crisis": ["lack of proper coordination within the firm in the event of a crisis", "unclear ownership of reputation risk and responsibilities", "weak internal coordination"],
  "Crisis communication risk": ["unaccountability of the crisis to the stakeholders", "lack of proper justifications for actions and decisions", "poor content and style of communication to the media or victims", "lack of structured communication strategy"],
  "Risk of non-implementation of corporate social responsibility": ["failure to go beyond the compliance of rules and regulation", "lack of innovative policies addressing social and environmental welfare", "deforestation"],
  "Spill-over effect due to the risk of upstream association": ["transfer of damaged reputation of the upstream third parties", "suppliers' or contractors' unethical practices", "partner bankruptcy and bad reputation", "poor supply chain visibility"],
  "Demand risk": ["spill-over risks associated with downstream operations", "decline in demand or product stockout", "inaccurate forecast of demand and customer preferences", "spill-over effect from competitors' unethical practices", "bullwhip effect"],
  "Transportation risk": ["product theft or damages during transportation and handling", "delay in product delivery", "changes in transportation modes", "border crossings, port shutdown and re-routing", "cargo theft at warehouse"]
}
```

### Perera et al. (2022)
```json
{
  "Strategy": ["Customer trust and confidentiality", "Customer perception", "Public dissemination of incident", "Customer relationship management", "Executive leadership", "Promise fulfilment", "Community management", "Company visibility", "Corporate branding", "Customer satisfaction", "Competitor effectiveness", "Decline in revenue", "Stock market price"],
  "Structure": ["Security effectiveness", "Management and leadership", "Regulatory risks", "Corporate policies and guidelines", "Business transparency", "Corporate code of conduct", "Financial performance", "Innovation", "Price-to-quality ratio", "Product recall"],
  "Process": ["Cyberattacks and incidents", "Internal coordination and controls", "Stakeholder response speed", "Effectiveness of communication", "Digital interactivity", "Fake news reporting", "Customer reviews/ratings", "Design of website", "Online advertisement and publicity"],
  "Rewards": ["Emotional connections and responses", "Employee satisfaction", "Payment systems", "Emotional appeal", "Employee benefits"],
  "People": ["Training and awareness programmes of organisations", "Sustained credibility", "Psychological factors", "Stakeholder type", "Loss of customers", "Desirable employer"]
}
```

### Hanay et al. (2024)
```json
{
  "Environmental": ["Region", "Institution Type", "Growth opportunities", "Perceptions of stakeholders"],
  "Financial": ["Size (Asset)", "Leverage", "Assets' opacity", "Earnings Volatility", "Stock Price Volatility", "Firm Performance", "Revenue", "Return on Asset (ROA)", "Shareholder Value", "Return on Equity (ROE)", "Capital Efficiency", "Cash Flow Volatility", "Capital Cost", "Frequency of Dividends", "Market Value", "Loan Commitments"],
  "Organizational": ["Reputation awareness", "Risk culture (CRO, Risk Committee awareness)", "Age/Year", "Industrial Diversification", "Institutional Ownership", "Social Responsibility Support", "Assessment of Big auditors (PwC, EY, KPMG, Deloitte, etc.)", "Assessment of Big rating agency (S&P, Fitch, Moody's, etc.)", "Number of Fraud Issues"]
}
```