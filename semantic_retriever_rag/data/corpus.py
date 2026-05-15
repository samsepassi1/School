"""Synthetic employee-handbook corpus and queries used by the retrievers.

The corpus mimics the kind of internal documents an employee-facing assistant
might be asked to search over. The ``RELEVANT`` mapping is the ground truth
used by ``evaluator.py`` to compute Recall@k, Precision@k and MRR.
"""

from __future__ import annotations

from typing import Dict, List


DOCUMENTS: List[str] = [
    # 0
    "Employees are entitled to twenty paid vacation days per calendar year. "
    "Unused days do not roll over after December 31.",
    # 1
    "Sick leave is granted separately from vacation and does not require "
    "advance approval. A doctor's note is required after three consecutive days.",
    # 2
    "Parental leave provides twelve weeks of paid time off for new parents, "
    "available within the first twelve months following the birth or adoption.",
    # 3
    "The remote work policy allows employees to work from home up to three days "
    "per week with manager approval. Hybrid schedules must be filed quarterly.",
    # 4
    "Business travel must be booked through the corporate portal. Receipts and "
    "expense reports are due within fourteen days of returning to the office.",
    # 5
    "Health insurance coverage begins on the first day of the month following "
    "your hire date. Dependents may be added during open enrollment.",
    # 6
    "The 401(k) retirement plan offers a company match of 50 percent on the "
    "first six percent of salary that you contribute each pay period.",
    # 7
    "Performance reviews are conducted twice a year, in June and December. "
    "Promotion decisions are made during the December review cycle.",
    # 8
    "All employees must complete the annual cybersecurity training by March 31. "
    "Phishing simulations are run quarterly to reinforce awareness.",
    # 9
    "Reimbursement for tuition or professional certifications is capped at "
    "five thousand dollars per year and requires prior approval from your manager.",
    # 10
    "The office is closed on the following observed holidays: New Year's Day, "
    "Memorial Day, Independence Day, Labor Day, Thanksgiving, and Christmas.",
    # 11
    "Onboarding for new hires takes place every Monday. The first week includes "
    "IT setup, benefits enrollment, and an introduction to company values.",
    # 12
    "Use the IT helpdesk portal to request a new laptop, reset your password, "
    "or report hardware problems. Most tickets are resolved within one business day.",
    # 13
    "Workplace harassment of any kind is prohibited. Reports can be made "
    "confidentially to Human Resources or through the anonymous ethics hotline.",
    # 14
    "Stock options vest over four years with a one-year cliff. Vested shares "
    "may be exercised through the equity platform once they become available.",
    # 15
    "Conference attendance for professional development is encouraged. Employees "
    "should submit a request with the expected business impact for budget approval.",
]

QUERIES: List[str] = [
    "How many vacation days do I get each year?",
    "What is the company policy on working from home?",
    "How does the 401k match work?",
    "When do health benefits start after I am hired?",
    "How do I report harassment at work?",
    "Can I get reimbursed for taking an online course?",
    "What holidays does the office observe?",
    "How do I get a new laptop from IT?",
]

# Ground truth: query index -> list of relevant document indices.
RELEVANT: Dict[int, List[int]] = {
    0: [0],
    1: [3],
    2: [6],
    3: [5],
    4: [13],
    5: [9],
    6: [10],
    7: [12],
}
