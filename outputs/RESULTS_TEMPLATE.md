# Capstone Results — Sahibpreet Singh

**Scenario:** Banking 
**Model/API provider used:** OpenRouter `nvidia/nemotron-3-super-120b-a12b:free` (if down then used `liquid/lfm-2.5-2.6b:free`)
**Embedding model:** all-MiniLM-L6-v2 (default)
**Date completed:** 2026-08-13
**Total time spent:** ~ 4 hours 0 minutes

---

## 1. Pipeline overview

I used sentence-aware chunking. Applied Cross Encoder Reranking while retrieval.
Created Agentic flow with Langgraph encasulating human approval gate within it.
Evaluated the responses using LLM as judge Scoring.

Built my own script and built a RAG pipeline and a ChatBot (RAG_AND_CHATBOT.ipynb).
The script implements ingestion, chunking (robust sentence aware chunking), embedding,bm25Index - regex based tokenization, hybrid search , RRF, HyDE, Prompting, Maintaining Chat history, Summarizing chat history (into consolidated meaningful piece of text, while preserving user PII, intent and user preferneces) after every 4 messages to reduce context size being sent to LLM (prevents `lost in the middle` issues and saves cost).
---

## 2. Sample Q&A with citations

### Example 1
**Q:** How much will I be charged if I close my account after 2 months?
**A:** If you close your account after 2 months (which falls between 14 days and 12 months of opening), the closure charge is INR 500.
**Sources cited:** account_closure.txt

### Example 2
**Q:** Can I postpone my EMI payment if I'm going through financial hardship?
**A:** Yes, you can request an EMI deferral if you are experiencing genuine financial hardship. The bank allows deferral of up to two EMIs per loan t
enure, subject to approval, with interest continuing to accrue and the loan tenure being extended accordingly. Requests must be submitted at leas
t five working days before the due EMI date.
**Sources cited:** faq_emi.txt

### Example 3 (edge case — abstention or emergency escalation test)
**Q:** What is capital of France?
**A:** I don't have enough information to answer that precisely — please contact your care team directly.

**Sources cited:** None
**Did it correctly abstain/escalate?** Yes

---

## 3. Evaluation results

### RAGAS (if run)
| Metric | Score |
|---|---|
| Faithfulness | |
| Answer Relevancy | |

### LLM-as-Judge (average across samples)
| Dimension | Avg Score (1-5) |
|---|---|
| Relevance | 3.67|
| Faithfulness |5 |
| Completeness |3.33 |

**Interpretation (required, 3-5 sentences):** 
These numbers tell that answers from llm are totally based on the context provided to it. So llm wnd prompt are fine in this context.

The generated answers and retrieved context are relevant to the query for banking related questions that LLM is expected to answer. The Relevance drops in case of out of scope questions (as desired, as we have prompted the LLM to not answer any question for which it does not have any context provided)

For non relevant queries the answer is score lowest completeness as LLM won't go beyond given context to answer (as prompted). It is fine as per our designed pipeline.

Yes, some information is missing in the retrieved context in one of the cases, that is why for one of the answers we got completeness score of 4. As faithfulness is 5, so may need to check retrieval part or the add in more information in our knowledge base.

---

## 4. Documented failure case (required)

The the llm used to hallucinate and answer incompletely.

**What happened:**
The retrieved context was not complete ending abruptly. Also the Prompts needed structure. 

**Why it happened (your hypothesis):**
Chunking was not proper. Also the direct query was no that meaningful. Prompts needed to be improved and structured.

**What you'd change to fix it:**
...
There was chunking problem, then I created sentence aware chunks and regex based tokenization for bm25, the retrieval improved. Also in chatBot I added query drafter to generate better query considering chat history and current user query. Gave structure to prompts and mentioned clear instructions
---

## 5. Human-in-the-loop test

Query: Approve my loan of 2000000

I first rejected it , it did not give out any response and abruptly ended the program, saying "Action not approved by human reviewer. No further action taken."

Next time I approved it, I got this response: """The loan amount you’re requesting—INR\u202f2,000,000—falls within t
he permissible personal‑loan range of INR\u202f50,000 to INR\u202f2,500,000 stated in the loan policy. Approval, however, also depends on your credit score (standard approval for scores\u202f≥\u202f700, secondary review for 650‑699) and whether the amount does not exceed 20\u202ftimes your average monthly net income. Without those details I cannot confirm approval; you’ll need to submit a full application for the bank’s assessment.
Sources: loan_policy.txt'"""
---

## 6. Production-readiness note (5-10 lines)

real content-safety API instead of the regex PII filter, semantic caching, monitoring dashboards, retry/
backoff on API failures, rate limiting, proper secrets management, CI/CD regression
tests on prompts.

---

## 7. Stretch goals attempted

- [ Yes] Cross-encoder reranking
- [ Yes] Real LangGraph StateGraph (not the simplified loop)
- [ Yes] Second LLM-as-judge rubric dimension
- [ Yes] HyDE query rewriting
- [ No] Alternate embedding model comparison
- [ Yes] Other: ChatBot, Regex Based tokenization.
