"""Build the full educational project writeup as a DOCX (python-docx). Part 1: helpers + Ch 1-2."""
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(r"F:\credit-risk-project - Copy")
FIG_EDA = ROOT / "reports" / "figures" / "eda"
FIG = ROOT / "reports" / "figures"
TBL = ROOT / "reports" / "tables"
OUT = ROOT / "docs" / "PROJECT_WRITEUP.docx"

NAVY = RGBColor(0x1F, 0x38, 0x64)
TEAL = RGBColor(0x2E, 0x75, 0xB6)
GREEN = RGBColor(0x37, 0x5F, 0x2C)
GRAY = RGBColor(0x59, 0x59, 0x59)
AMBER = RGBColor(0x7F, 0x60, 0x00)

doc = Document()

style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.12

for lvl, size, color in [("Heading 1", 20, NAVY), ("Heading 2", 15, TEAL), ("Heading 3", 12.5, NAVY)]:
    h = doc.styles[lvl]
    h.font.name = "Calibri Light"
    h.font.size = Pt(size)
    h.font.color.rgb = color
    h.font.bold = True
    h.paragraph_format.space_before = Pt(16 if lvl == "Heading 1" else 12)
    h.paragraph_format.space_after = Pt(6)


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def para(text="", bold=False, italic=False, size=11, color=None, align=None, space_after=6):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(size)
    if color:
        r.font.color.rgb = color
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(space_after)
    return p


def bullet(text, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        r = p.add_run(bold_prefix)
        r.bold = True
        p.add_run(text)
    else:
        p.add_run(text)
    p.paragraph_format.space_after = Pt(3)
    return p


def box(title, body_lines, bg, border_hex, title_color):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = t.cell(0, 0)
    shade(cell, bg)
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "8")
        e.set(qn("w:color"), border_hex)
        borders.append(e)
    tcPr.append(borders)
    p = cell.paragraphs[0]
    r = p.add_run(title)
    r.bold = True
    r.font.size = Pt(10.5)
    r.font.color.rgb = title_color
    p.paragraph_format.space_after = Pt(3)
    for line in body_lines:
        p2 = cell.add_paragraph()
        r2 = p2.add_run(line)
        r2.font.size = Pt(10.5)
        p2.paragraph_format.space_after = Pt(3)
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(0)
    return t


def concept(title, lines):
    box(f"CONCEPT | {title}", lines, "DDEBF7", "2E75B6", TEAL)


def decision(title, lines):
    box(f"DECISION | {title}", lines, "E2EFDA", "548235", GREEN)


def warning(title, lines):
    box(f"COMMON TRAP | {title}", lines, "FFF2CC", "BF8F00", AMBER)


FIG_COUNTER = [0]


def figure(path, caption, width=6.2):
    FIG_COUNTER[0] += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(f"Figure {FIG_COUNTER[0]}. {caption}")
    r.italic = True
    r.font.size = Pt(9.5)
    r.font.color.rgb = GRAY
    cap.paragraph_format.space_after = Pt(10)


def make_table(headers, rows, widths=None, header_bg="1F3864", font_size=9.5):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(font_size)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(hdr[i], header_bg)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            r = p.add_run(str(val))
            r.font.size = Pt(font_size)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Inches(w)
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(2)
    return t


def code_block(text):
    t = doc.add_table(rows=1, cols=1)
    cell = t.cell(0, 0)
    shade(cell, "F2F2F2")
    p = cell.paragraphs[0]
    r = p.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(9)
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(2)


# ================================================================ TITLE PAGE
for _ in range(4):
    doc.add_paragraph()
para("FROM RAW LOAN DATA TO A DEPLOYABLE CREDIT-RISK MODEL", bold=True,
     size=26, color=NAVY, align="center", space_after=8)
para("A Complete, Beginner-Friendly Walkthrough of Every Step, Every Decision, and Every Concept",
     italic=True, size=14, color=TEAL, align="center", space_after=24)
para("MSc Dissertation Project: Machine Learning for Financial Credit Risk",
     size=12, align="center", space_after=4)
para("Dataset: Home Credit Default Risk | 307,511 loan applications | 2018-2020",
     size=11, color=GRAY, align="center", space_after=4)
para("September 2026", size=11, color=GRAY, align="center")
doc.add_page_break()

# ================================================================ HOW TO READ
doc.add_heading("How to Read This Document", level=1)
para("This writeup was written for someone starting machine learning with basic knowledge. "
     "It follows the project exactly in the order it was done: data analysis first, then cleaning "
     "decisions, feature engineering, modelling, evaluation, and explanation. Nothing is skipped, "
     "and every technical term is explained the first time it appears.")
bullet("Blue boxes marked CONCEPT explain a machine-learning idea in plain language before it is used.")
bullet("Green boxes marked DECISION state a choice the project made and the exact reason behind it.")
bullet("Yellow boxes marked COMMON TRAP point out mistakes that beginners (and professionals) make.")
bullet("Every figure is explained in the text around it: what to look at, and what it proves.")
para("The single most important idea in this whole project is data leakage: making sure the model "
     "is never tested on information it has secretly already seen. That idea appears in Chapter 2 "
     "and comes back in almost every chapter after it.", italic=True, color=GRAY)
doc.add_page_break()

# ================================================================ CH 1
doc.add_heading("Chapter 1: The Business Problem", level=1)
para("A bank lends money to people. Some of those people will not pay the money back. "
     "That event is called a default. Every loan that defaults is a direct financial loss, so banks "
     "want to predict, before approving a loan, how likely each applicant is to default.")
para("This is a perfect machine-learning problem because the bank has history: hundreds of "
     "thousands of past loans where we already know the outcome (repaid or defaulted). We can show "
     "those examples to an algorithm and let it learn the patterns that separate good borrowers "
     "from bad ones. Then we apply what it learned to new applicants.")

concept("Supervised learning and classification", [
    "Supervised learning = learning from examples where the answer is already known. Each past loan "
    "is an example; its outcome (repaid/defaulted) is the answer key.",
    "Classification = the task of predicting a category. Here there are exactly two categories: "
    "default (1) or no default (0). Two categories makes this a binary classification problem.",
    "The algorithm's job: find the combination of applicant characteristics that best separates "
    "the 1s from the 0s.",
])

para("The dataset comes from Home Credit, a consumer lender. Each row is one loan application:",
     bold=True)
make_table(
    ["Dataset fact", "Value", "What it means"],
    [
        ["Rows", "307,511", "One row per loan application"],
        ["Columns", "122 features + 1 target + ID", "Characteristics recorded at application time"],
        ["Target column", "TARGET", "1 = the borrower defaulted, 0 = repaid"],
        ["Time span", "2018-2020", "Each application has a date - this matters a lot later"],
        ["ID column", "SK_ID_CURR", "Anonymous applicant identifier"],
    ],
    widths=[1.6, 1.9, 3.2],
)

concept("What is a feature?", [
    "A feature (also called a variable or attribute) is one measurable property of an applicant. "
    "Examples here: annual income (AMT_INCOME_TOTAL), loan size (AMT_CREDIT), age (DAYS_BIRTH), "
    "whether they own a car (FLAG_OWN_CAR), and three scores from external credit bureaus "
    "(EXT_SOURCE_1/2/3).",
    "The model never sees a person - it sees a row of numbers. Our job in Chapters 2-5 is to turn "
    "the messy raw columns into honest, useful numbers.",
])

doc.add_heading("Why this problem is hard", level=2)
bullet("Only about 8% of borrowers default. The interesting cases are rare.", bold_prefix="Imbalance: ")
bullet("Income, employment and other fields contain errors and placeholder values.", bold_prefix="Dirty data: ")
bullet("Many columns repeat the same information in slightly different forms.", bold_prefix="Redundancy: ")
bullet("A loan approved in 2020 faces a different economy than one in 2018. The model must survive time.", bold_prefix="Time: ")
para("Each of these four problems drives a major part of this project.")
doc.add_page_break()

# ================================================================ CH 2
doc.add_heading("Chapter 2: Exploratory Data Analysis (EDA)", level=1)
para("EDA means looking at the data before modelling: distributions, missing values, strange "
     "values, relationships between columns. The goal is not charts for their own sake - every "
     "finding in this chapter forces a concrete decision in Chapter 3.")

doc.add_heading("2.1 Finding #1: the target is heavily imbalanced", level=2)
figure(FIG_EDA / "eda_target_balance.png",
       "Distribution of the target variable. Only ~8% of the 307,511 loans defaulted.", 5.2)
para("Out of 307,511 applications, roughly 282,000 were repaid and only 24,000 defaulted. "
     "This imbalance shapes everything that follows: which metric we trust, how we train, and how "
     "we judge success.")

concept("Class imbalance and the accuracy trap", [
    "Class imbalance = one outcome is much rarer than the other. Here defaults are ~8%.",
    "Imagine a lazy model that says 'nobody defaults' for every applicant. It would be right "
    "92% of the time - 92% accuracy! But it catches zero defaulters, which is exactly what the "
    "bank pays to avoid.",
    "Lesson: on imbalanced data, accuracy is a lie detector that always says 'fine'. This project "
    "therefore bans accuracy as a headline metric and uses metrics that focus on the rare class "
    "(Chapter 8).",
])

doc.add_heading("2.2 Finding #2: a physically impossible employment value", level=2)
figure(FIG_EDA / "eda_days_employed_sentinel.png",
       "DAYS_EMPLOYED contains a spike at exactly 365,243 days - one thousand years.", 6.0)
para("The column DAYS_EMPLOYED records how long the applicant has been employed, in negative days "
     "(-365 means one year). But 55,374 rows (18% of the data) contain exactly 365,243 days - "
     "one thousand years of employment. Nobody has worked for a millennium. This is a placeholder "
     "value the data creators used for pensioners and unemployed applicants.")

concept("Sentinel (placeholder) values", [
    "A sentinel value is a fake number used to mean 'not applicable' or 'missing' - for example "
    "-999, 999999, or here 365243. Databases do this when they have no proper missing-value slot.",
    "The danger: algorithms treat sentinels as real numbers. A model sees '1,000 years employed' "
    "and may conclude that extremely long employment predicts something - learning a rule built "
    "on a bookkeeping artifact instead of reality.",
])

doc.add_heading("2.3 Finding #3: extreme incomes stretch every scale", level=2)
figure(FIG_EDA / "eda_income_winsor.png",
       "Income before and after capping. 278 applicants earn above the 99.9th percentile cap of $900,000.", 6.4)
para("Annual income (AMT_INCOME_TOTAL) has a very long tail: a tiny number of applicants report "
     "incomes far beyond everyone else. Those few extreme values matter far more than their count "
     "suggests, because several engineered features are ratios that divide by income.")

doc.add_heading("2.4 Finding #4: credit-bureau scores are the strongest signal", level=2)
figure(FIG_EDA / "eda_ext_sources.png",
       "The three external bureau scores, split by outcome. Defaulters (red) sit clearly lower.", 6.4)
para("EXT_SOURCE_1/2/3 are scores bought from external credit bureaus - companies that track "
     "people's credit history across all their lenders. The histograms show a clean separation: "
     "borrowers who defaulted have visibly lower bureau scores. These three columns will dominate "
     "the model's decisions (confirmed by SHAP in Chapter 11).")

concept("Credit bureau", [
    "A credit bureau collects repayment history from many lenders and compresses it into a score. "
    "Banks buy these scores because they contain information the bank itself does not have: how "
    "the applicant behaved with other lenders.",
])

doc.add_heading("2.5 Finding #5: whole families of duplicated columns", level=2)
figure(FIG_EDA / "eda_building_corr.png",
       "Correlation matrix of building-information columns. Almost every off-diagonal square is black: |r| > 0.95.", 5.6)
para("Information about the applicant's building exists in triplets: APARTMENTS_AVG, "
     "APARTMENTS_MEDI and APARTMENTS_MODE (same for 13 other building properties). The correlation "
     "matrix shows these triplets are near-identical copies of each other (correlation above 0.97). "
     "Keeping all three adds nothing except noise and computation.")

concept("Correlation and multicollinearity", [
    "Correlation (r) measures how two columns move together, from -1 (perfect opposites) through "
    "0 (unrelated) to +1 (perfect copies). |r| > 0.95 means the columns carry almost the same "
    "information.",
    "Multicollinearity = feeding a model many near-copy columns. Tree models waste splits deciding "
    "between identical columns; linear models become unstable because credit can be assigned to "
    "either copy. Removing duplicates makes models simpler and more honest.",
])

doc.add_heading("2.6 Finding #6: a correlation that was actually a data artifact", level=2)
figure(FIG_EDA / "eda_flag_emp_phone.png",
       "Left: raw data. Right: the same correlation after fixing the sentinel. It collapses from -0.9998 to ~0.002.", 6.4)
para("EDA showed FLAG_EMP_PHONE (does the applicant have a work phone: 0/1) correlated with "
     "DAYS_EMPLOYED at r = -0.9998 - essentially a perfect relationship. That is implausible for "
     "real human behaviour. Investigation revealed the truth: almost every row with the fake "
     "365,243-day employment value also had FLAG_EMP_PHONE = 0. The 'correlation' was created "
     "entirely by the sentinel. After replacing the sentinel with a proper missing value, the "
     "correlation collapsed to 0.002 - no relationship at all.")

warning("Correlation can be manufactured by bad data", [
    "A strong correlation does not prove a real relationship. Always ask: could a data artifact "
    "(sentinel, duplicate, bug) be generating it? Here, deleting one bad value destroyed a "
    "'perfect' correlation. If we had dropped FLAG_EMP_PHONE because of it, we would have removed "
    "a genuinely useful, independent feature based on a mirage.",
])

doc.add_heading("2.7 Finding #7: the data has a time dimension", level=2)
figure(FIG_EDA / "eda_temporal_split.png",
       "Applications by year. The project trains on 2018-2019 and tests on 2020.", 5.6)
para("Every application has a date. This is a financial dataset, and financial behaviour drifts "
     "over time: the economy changes, lending policy changes, the mix of applicants changes. "
     "A model must work on future applicants, not just on the past.")

concept("Data leakage - the most important idea in this project", [
    "Data leakage = letting information from the future (or from the test set) influence training. "
    "A leaked model looks excellent in experiments and fails in production, because in production "
    "the future has not happened yet.",
    "Example of leakage: shuffling all years randomly and splitting 80/20. The model would train on "
    "2020 applications and be 'tested' on 2019 ones - it would literally be tested on the past it "
    "already studied.",
    "The fix used everywhere in this project: time-ordered splits. Train only on 2018-2019, test "
    "only on 2020, and never compute any statistic (income cap, scaling, resampling) using test data.",
])

decision("Temporal split: train 2018-2019, test 2020", [
    "Train window: 205,007 applications (8.12% default). Test window: 102,504 applications "
    "(7.98% default).",
    "Why: this mimics production. The bank deploys a model trained on history and scores future "
    "applications. Testing on 2020 - a full year the model never touched - measures exactly that.",
    "This style of evaluation is called out-of-time (OOT) validation, and it is the industry "
    "standard for credit-risk models.",
])
doc.add_page_break()

# ================================================================ CH 3
doc.add_heading("Chapter 3: Cleaning Decisions - Every Finding Gets a Fix", level=1)
para("This chapter converts each EDA finding into an exact cleaning action, with the reasoning "
     "spelled out. The table below is the decision register; the sections after it unpack the two "
     "hardest ones in full.")
make_table(
    ["EDA finding", "Decision", "Why"],
    [
        ["365,243-day employment sentinel (18% of rows)",
         "Replace with missing (NaN) + add a binary flag DAYS_EMPLOYED_ANOM",
         "The value is fake; but being a pensioner/unemployed is real information, so we keep it as a flag"],
        ["Extreme income tail (278 rows)",
         "Winsorize (cap) income at the 99.9th percentile = $900,000",
         "Stops a handful of outliers from distorting ratios and scales; cap computed on training years only"],
        ["58 raw ORGANIZATION_TYPE categories",
         "Group into 11 business sectors",
         "Many categories had only a few dozen rows; grouped sectors are learnable and interpretable"],
        ["'XNA' tokens in CODE_GENDER / ORGANIZATION_TYPE",
         "Convert to missing",
         "XNA means 'not answered'; treating it as a real category would invent a fake group"],
        ["28 building _MEDI/_MODE columns + 2 social-circle 60-day columns",
         "Drop all 30",
         "Near-duplicates of kept columns (|r| > 0.97); TOTALAREA_MODE kept because it has no _AVG twin"],
        ["FLAG_EMP_PHONE looked collinear (r = -0.9998)",
         "KEEP it",
         "The correlation was a sentinel artifact; after the fix it is 0.002 - an independent, useful feature"],
    ],
    widths=[2.2, 2.2, 2.6],
)

doc.add_heading("3.1 Deep dive: why winsorize income, and why does it matter so much?", level=2)
para("This is one of those lines that looks dense, so let us unpack it word by word:")
box("THE SENTENCE, DECODED", [
    "'Even though RobustScaler scales by IQR, extreme un-winsorized numbers distort derived ratios "
    "(like CREDIT_INCOME_PERCENT and INCOME_PER_PERSON) and can create single-applicant outlier "
    "splits in tree models.'",
], "FCE4D6", "C55A11", RGBColor(0x84, 0x3C, 0x0C))

bullet("RobustScaler scales by IQR: our scaler (Chapter 5) divides numbers by the interquartile "
       "range - the spread of the middle 50% of people. It is already resistant to outliers, so why "
       "bother capping at all?", bold_prefix="Part 1 - ")
bullet("The scaler only protects the raw column. But we also create ratio features by dividing "
       "one column by income (for example debt-to-income = loan size / income). If one applicant "
       "earns $10 million, their debt-to-income becomes almost zero, and INCOME_PER_PERSON blows up "
       "for their household. The outlier has now been copied into several new columns where the "
       "scaler cannot trace it back.", bold_prefix="Part 2 - ")
bullet("Tree models split on single values. One $10-million income can become its own split rule "
       "('if income > $9,999,999 then ...'), a rule that applies to exactly one person in history "
       "and will never fire again. That is memorizing noise, not learning.", bold_prefix="Part 3 - ")

para("The fix: winsorizing (capping) income at the 99.9th percentile ($900,000) is the recognized "
     "financial and econometric standard.", bold=True)

concept("Winsorizing and percentiles", [
    "A percentile answers: 'below what value do X% of people fall?' The 99.9th percentile of income "
    "is the value below which 99.9% of applicants sit. Here that value is $900,000.",
    "Winsorizing = capping everything above that value down to it. The 278 people earning more than "
    "$900,000 are recorded as earning exactly $900,000. Nobody is deleted; the tail is just blunted.",
    "Why 99.9% and not 95%? In finance, the top 1% of incomes can be genuine signal (wealthy "
    "borrowers behave differently), but the top 0.1% is almost always data-entry noise. Cutting at "
    "99.9% removes the noise while keeping nearly all real variation.",
])

decision("Leakage control on the cap", [
    "The $900,000 cap was computed using ONLY the 2018-2019 training applications, then applied "
    "to everyone including the 2020 test set.",
    "Why this matters: if we had computed the cap on all three years, the test year's extreme "
    "incomes would have influenced a number used during training. That is a subtle leak. The rule "
    "of this project: every statistic is fitted on training data and merely applied to test data.",
])

doc.add_heading("3.2 Deep dive: the sentinel fix keeps the information", level=2)
para("A naive fix would delete the 55,374 sentinel rows or just blank the value. Both lose "
     "something. Deleting rows throws away 18% of the dataset. Blank-only loses the fact that "
     "these people are pensioners/unemployed - which genuinely predicts lower default risk.")
para("So the fix does two things at once: the fake number becomes missing (so no model can read "
     "'1,000 years'), and a new column DAYS_EMPLOYED_ANOM = 1 marks these applicants. The "
     "pensioner information survives as an honest binary flag.")

concept("Missing-value imputation (preview)", [
    "Most algorithms cannot consume blank cells. After we create missing values, a later step "
    "(Chapter 5) fills them with sensible substitutes - here, the median of the column, computed on "
    "training data only. The flag column makes sure the fact 'this value was missing on purpose' "
    "is not lost when the blank gets filled.",
])
doc.add_page_break()

# ================================================================ CH 4
doc.add_heading("Chapter 4: Feature Engineering - Turning Columns into Knowledge", level=1)
para("Raw columns are facts the applicant reported. Engineered features are ratios and "
     "combinations that express financial meaning. A bank underwriter does not look at 'loan size' "
     "and 'income' separately - they look at how big the loan is relative to the income. We build "
     "that judgement into numbers.")

doc.add_heading("4.1 Financial burden ratios", level=2)
make_table(
    ["New feature", "Formula", "What an underwriter reads from it"],
    [
        ["CREDIT_INCOME_PERCENT", "loan size / income", "Debt-to-income: how many years of income the loan equals. Higher = heavier burden"],
        ["ANNUITY_INCOME_PERCENT", "monthly payment / income", "Debt service ratio: what slice of income the monthly instalment eats"],
        ["PAYMENT_RATE", "monthly payment / loan size", "How aggressive the repayment schedule is relative to the borrowed amount"],
        ["GOODS_CREDIT_RATIO", "goods price / loan size", "Below 1 means the loan exceeds the item price - extra cash borrowed on top"],
        ["CREDIT_GOODS_DIFF", "loan size - goods price", "The absolute amount of that extra cash"],
        ["INCOME_PER_PERSON", "income / family members", "Household income pressure: same salary supports different family sizes"],
    ],
    widths=[2.0, 1.9, 3.1],
)

concept("Why ratios beat raw amounts", [
    "A $20,000 loan is trivial for someone earning $200,000 and crushing for someone earning "
    "$15,000. Raw loan size cannot express that; the ratio can.",
    "This is domain knowledge encoded as arithmetic - one of the highest-value steps in any "
    "applied ML project. Note the tiny epsilon (1e-6) added to every denominator: it prevents "
    "division by zero when income is missing or zero.",
])

doc.add_heading("4.2 Time-of-life ratios", level=2)
para("Dates in this dataset are stored as negative day counts (days before the application). We "
     "convert them into human-readable quantities:")
bullet("AGE_YEARS = -DAYS_BIRTH / 365.25 (age in years).", bold_prefix="")
bullet("EMPLOYED_YEARS = -DAYS_EMPLOYED / 365.25 (tenure in years).", bold_prefix="")
bullet("DAYS_EMPLOYED_PERCENT = employment days / age days: what fraction of adult life spent in "
       "the current job - a stability signal.", bold_prefix="")
bullet("PHONE_TO_BIRTH_RATIO, REGISTRATION_TO_BIRTH_RATIO, ID_PUBLISH_TO_BIRTH_RATIO: how old the "
       "phone number, address registration and ID document are, relative to age. Old, stable "
       "documents are a trust signal; brand-new ones can indicate fraud.", bold_prefix="")
bullet("CAR_TO_AGE_RATIO = car age / applicant age (a 30-year-old driving a 25-year-old car is a "
       "different profile than a 60-year-old doing the same).", bold_prefix="")

doc.add_heading("4.3 Bureau-score aggregates and interactions", level=2)
para("Since the three bureau scores are the strongest raw signal (Finding #4), we give the model "
     "several views of them:")
bullet("EXT_SOURCES_MEAN / MIN / MAX / STD: the average, worst, best and spread of the three "
       "scores. The minimum matters because lenders often reject on the worst score.", bold_prefix="")
bullet("Pairwise products (EXT_SOURCE_1_2_PROD, 2_3_PROD, 1_3_PROD) and the triple product: "
       "interaction features. If two bureaus both rate someone poorly, the product amplifies that "
       "joint bad news far more than either score alone.", bold_prefix="")
bullet("All aggregates are NaN-safe: applicants missing one or two bureau scores still get a "
       "mean computed from whatever scores they have, instead of being excluded.", bold_prefix="")

concept("Interaction features", [
    "An interaction feature combines two columns (often by multiplication) so the model can see "
    "their joint effect as one number. Multiplying two bureau scores makes 'both bad' much more "
    "extreme than 'one bad'. Linear models especially need these, because on their own they can "
    "only add effects, not multiply them.",
])

doc.add_heading("4.4 Other engineering", level=2)
bullet("ORGANIZATION_TYPE: 58 raw employer categories grouped into 11 sectors (Trade, Industry, "
       "Government/Public, Education/Healthcare, Financial/RealEstate, etc.) so each group has "
       "enough rows to learn from.", bold_prefix="")
bullet("DOCUMENT_COUNT: the sum of all FLAG_DOCUMENT_* columns - how many identity documents the "
       "applicant provided. More paperwork often correlates with verified, lower-risk applicants.", bold_prefix="")
para("After cleaning and engineering, the dataset has 97 numeric and 16 categorical columns ready "
     "for modelling.")
doc.add_page_break()

# ================================================================ CH 5
doc.add_heading("Chapter 5: Preprocessing - Getting Numbers Model-Ready", level=1)
para("Algorithms need a clean numeric matrix: no blanks, no text categories, comparable scales. "
     "This chapter builds that transformation - and builds it in a way that cannot leak.")

doc.add_heading("5.1 Numeric columns: impute, then scale", level=2)
concept("Missing-value imputation", [
    "Imputation = filling blank cells with a substitute so algorithms can process them. We use the "
    "median (the middle value) rather than the mean, because the median is not dragged around by "
    "the extreme values we saw in EDA.",
    "Critical rule: the median is computed on the training data only, then used to fill blanks in "
    "both training and test data. Using the test set's own median to fill its blanks would leak "
    "test-set information into the pipeline.",
])

concept("Feature scaling, StandardScaler vs RobustScaler", [
    "Scaling puts every numeric column onto a comparable range. Without it, income (tens of "
    "thousands) would dominate age (tens) simply because its numbers are bigger.",
    "The classic StandardScaler subtracts the mean and divides by the standard deviation. Problem: "
    "the mean and standard deviation are themselves sensitive to outliers - exactly what this "
    "dataset has.",
    "RobustScaler instead subtracts the median and divides by the IQR (interquartile range - the "
    "width of the middle 50% of values). Both statistics ignore the extremes, so the scaling stays "
    "stable even with residual outliers. That is why this project chose RobustScaler.",
    "Which models care? Distance- and regularization-based models (logistic regression) need "
    "scaling badly. Tree models (random forest, XGBoost) are scale-invariant, but we scale "
    "everything anyway so all nine experiment cells get identical inputs - a fair comparison.",
])

doc.add_heading("5.2 Categorical columns: encode into 0/1 columns", level=2)
concept("One-hot encoding", [
    "Models cannot do arithmetic on text like 'Trade' or 'Industry'. One-hot encoding replaces a "
    "categorical column with one binary (0/1) column per category: ORGANIZATION_TYPE_Trade = 1 "
    "means the applicant works in trade.",
    "Two safety settings matter here. First, unknown categories in the test set are ignored "
    "(handle_unknown='ignore') instead of crashing - a 2020 applicant might have an employer type "
    "never seen in 2018-19. Second, missing categories are filled with an explicit 'Missing' "
    "category, so 'did not answer' becomes learnable information instead of a blank.",
    "After one-hot encoding, the 16 categorical columns expand, giving roughly 196 final model "
    "features (the exact count varies slightly per fold).",
])

doc.add_heading("5.3 The anti-leakage architecture", level=2)
para("The whole transformation is wrapped in a scikit-learn pipeline, and the project enforces one "
     "rule in two places:", bold=True)
bullet("Out-of-time track: the pipeline is fitted exactly once, on the full 2018-2019 window, and "
       "then applied untouched to 2020.", bold_prefix="OOT track: ")
bullet("Cross-validation track: the pipeline is re-fitted from scratch inside every single fold, "
       "on that fold's training portion only, and applied to that fold's validation portion. "
       "Fifteen folds means fifteen completely independent fittings.", bold_prefix="CV track: ")

warning("The most common leakage mistake in ML projects", [
    "Fitting the scaler/imputer on the whole dataset before splitting. It feels harmless - 'I'm "
    "just preprocessing!' - but the scaler has now memorized the test set's mean, median and "
    "missingness, and smuggled it into training. Every number downstream is slightly contaminated.",
    "The discipline: preprocessing objects are trained on training data and frozen afterwards. "
    "This project re-fits inside every fold, which is the strictest version of that discipline.",
])
doc.add_page_break()

# ================================================================ CH 6
doc.add_heading("Chapter 6: The Three Models - and Why These Three", level=1)
para("The experiment compares three model families that cover the major schools of supervised "
     "learning. Choosing one from each school means the results teach us about model types, not "
     "just about one algorithm.")

doc.add_heading("6.1 Logistic Regression - the interpretable baseline", level=2)
concept("Logistic regression", [
    "Despite the name, this is a classifier. It computes a weighted sum of the features, then "
    "squeezes the result through the logistic (sigmoid) function into a probability between 0 and "
    "1: P(default) = 1 / (1 + e^-(weighted sum)).",
    "Strengths: fast, stable, and transparent - each feature has one weight you can read and "
    "explain to a regulator. In credit risk that transparency has real legal value (GDPR Article "
    "22 gives people the right to an explanation of automated decisions).",
    "Weakness: on its own it can only draw straight decision boundaries, so we hand it interaction "
    "features (Chapter 4) to compensate.",
    "Settings used: regularization C = 1.0 (L2 penalty, the default - it shrinks weights slightly "
    "to fight overfitting), solver lbfgs, max 2000 iterations.",
])

doc.add_heading("6.2 Random Forest - many imperfect trees voting together", level=2)
concept("Decision trees and random forests", [
    "A decision tree is a flowchart learned from data: 'if bureau score < 0.4, go left; else if "
    "payment rate > 0.08, go right...' Each split searches for the question that best separates "
    "defaulters from repayers. Trees capture non-linear effects and interactions automatically.",
    "One tree overfits (memorizes noise). A random forest grows 200 trees, each on a random "
    "subset of rows and features, and averages their votes. Randomness decorrelates the trees, so "
    "their individual mistakes cancel out - this is called ensemble learning.",
    "Settings used: 200 trees, maximum depth 12, minimum 20 samples per leaf. These limits are "
    "deliberately conservative: they cap how much any tree can memorize, and they fit on an "
    "8 GB laptop.",
])

doc.add_heading("6.3 XGBoost - trees that correct each other", level=2)
concept("Gradient boosting and XGBoost", [
    "Boosting builds trees sequentially instead of in parallel. The first tree makes predictions; "
    "the second tree is trained specifically to fix the first tree's mistakes; the third fixes "
    "what remains; and so on. Each tree is small and only a fraction of its vote counts "
    "(learning rate 0.05), so the ensemble improves in careful increments.",
    "'Gradient' refers to gradient descent: each new tree is fitted to the direction that most "
    "reduces the current error. XGBoost is an engineered, regularized implementation of this idea "
    "and a perennial winner on tabular (spreadsheet-style) data.",
    "Settings used: 200 trees, learning rate 0.05, depth 6, subsample 0.8 (each tree sees 80% of "
    "rows), colsample_bytree 0.8 (each tree sees 80% of features), histogram tree method for speed.",
])

doc.add_heading("6.4 Frozen hyperparameters: the heart of a fair experiment", level=2)
concept("Hyperparameters vs parameters", [
    "Parameters are what the model learns from data (tree splits, regression weights). "
    "Hyperparameters are the settings we choose before training (number of trees, depth, learning "
    "rate). Tuning hyperparameters on the test set is another form of leakage.",
])
decision("Freeze hyperparameters across all nine cells", [
    "Each model family uses exactly one hyperparameter configuration for all three imbalance "
    "strategies (baseline, cost-sensitive, SMOTE).",
    "Why: the experiment's question is 'which imbalance strategy helps?', not 'which tuning "
    "helps?'. If XGBoost-with-SMOTE also got different tuning, we could never tell whether a score "
    "change came from SMOTE or from the tuning. Freezing everything else isolates the one variable "
    "under study - the same logic as a controlled laboratory experiment.",
])
doc.add_page_break()

# ================================================================ CH 7
doc.add_heading("Chapter 7: Handling Class Imbalance - Three Strategies", level=1)
para("With only 8% defaulters, a model can achieve high accuracy by ignoring the minority class "
     "entirely (Chapter 2). This project tests three ways to fight that, arranged in a 3x3 grid: "
     "three model families x three imbalance strategies = nine experiment cells.")

doc.add_heading("7.1 Baseline: do nothing special", level=2)
para("The control group. The model trains on the data exactly as it is: 92% repaid, 8% defaulted. "
     "Every experiment needs a control, otherwise we cannot measure whether an intervention "
     "actually helped. As we will see, the baselines learn to almost never predict default - which "
     "is precisely the failure mode the other two strategies attack.")

doc.add_heading("7.2 Cost-sensitive learning: make mistakes expensive", level=2)
concept("Cost-sensitive learning and class weights", [
    "Algorithms learn by minimizing their mistakes. Normally every mistake costs the same. "
    "Cost-sensitive learning changes the price list: in this project, missing a real defaulter "
    "(a false negative) is charged 10x more than a false alarm (a false positive).",
    "Implementation: logistic regression and random forest get class_weight='balanced' (weights "
    "inversely proportional to class frequency); XGBoost gets scale_pos_weight = number of "
    "non-defaulters / number of defaulters, computed on the training window only (about 11.3).",
    "Why 10x? In consumer lending, a missed default costs the bank the entire loan amount, while a "
    "false alarm costs one lost customer and one manual review. The 10:1 ratio is a standard "
    "conservative encoding of that asymmetry.",
])
para("Expected effect: the model becomes far more willing to flag applicants as risky. Recall "
     "(share of defaulters caught) should jump, at the price of more false alarms.")

doc.add_heading("7.3 SMOTE: manufacture synthetic defaulters", level=2)
concept("SMOTE (Synthetic Minority Over-sampling Technique)", [
    "Instead of re-weighting mistakes, SMOTE creates new training examples. For each minority "
    "(defaulter) example it finds its 5 nearest minority neighbours, picks one at random, and "
    "places a synthetic point on the line between them. The training set ends up roughly balanced.",
    "The intuition: give the model so many defaulter examples that it can no longer afford to "
    "ignore the class.",
    "The risk (confirmed by our results): in 196 dimensions with only ~16,000 real defaulters, "
    "'neighbours' can be far apart in any meaningful sense. The synthetic points may fill space "
    "with plausible-looking noise, teaching the model patterns that do not exist.",
])

warning("SMOTE must never touch the validation/test data", [
    "If synthetic defaulters were created from the full dataset before splitting, information "
    "about test-set applicants would leak into training (each synthetic point is built from real "
    "points, possibly including test ones).",
    "This project applies SMOTE strictly inside each training fold, after preprocessing, and the "
    "validation fold always keeps its natural 92/8 distribution - because production data will "
    "have the natural distribution too.",
])

decision("SMOTE computed once per fold, shared by all three SMOTE cells", [
    "Inside each CV fold, SMOTE runs once; the logistic-regression, random-forest and XGBoost "
    "SMOTE cells all train on that identical resampled set.",
    "Why: if each cell drew its own synthetic sample, differences between them could come from "
    "resampling luck rather than from the model family. Sharing one resample keeps the comparison "
    "clean.",
])
doc.add_page_break()

# ================================================================ CH 8
doc.add_heading("Chapter 8: Metrics - How Success Is Measured", level=1)
para("Choosing metrics is a modelling decision, not an afterthought. On this dataset, the wrong "
     "metric rewards the wrong behaviour.")

doc.add_heading("8.1 The confusion matrix: four outcomes", level=2)
concept("Confusion matrix vocabulary", [
    "Every prediction falls into one of four boxes:",
    "True Positive (TP): predicted default, actually defaulted. The catch we want.",
    "False Negative (FN): predicted safe, actually defaulted. The expensive miss - the bank loses "
    "the loan.",
    "False Positive (FP): predicted default, actually repaid. A lost good customer.",
    "True Negative (TN): predicted safe, actually repaid. Business as usual.",
    "All metrics below are just different ways of summarizing these four counts.",
])
make_table(
    ["Metric", "Formula idea", "What it answers", "Role in this project"],
    [
        ["Accuracy", "(TP+TN) / all", "How often right overall?", "BANNED as headline - the 92% trap"],
        ["Recall (sensitivity)", "TP / (TP+FN)", "What share of real defaulters did we catch?", "Core business question"],
        ["Precision", "TP / (TP+FP)", "Of our flags, how many were real defaulters?", "Cost of false alarms"],
        ["F1 score", "Harmonic mean of precision & recall", "Balance of the two", "Reported for completeness"],
        ["F2 score", "Recall-weighted F-beta (beta=2)", "Balance that values recall 4x more", "Matches the 10:1 cost logic"],
        ["ROC-AUC", "Ranking quality across all thresholds", "Can the model rank risk correctly?", "Secondary headline"],
        ["PR-AUC", "Same idea, precision-recall space", "Ranking quality focused on the rare class", "PRIMARY metric"],
        ["Cost/applicant", "(10 x FN + 1 x FP) / N", "Expected monetary loss per applicant", "Business bottom line"],
    ],
    widths=[1.4, 1.9, 2.2, 1.7],
)

concept("AUC: area under a curve, explained", [
    "A classifier does not have to output yes/no; it outputs a risk probability (0 to 1). "
    "Sweeping the decision threshold from 0 to 1 traces out a curve, and the area under that "
    "curve (AUC) summarizes performance at every threshold at once.",
    "ROC-AUC has a beautiful interpretation: pick one random defaulter and one random repayer - "
    "ROC-AUC is the probability the model assigns higher risk to the defaulter. 0.5 = coin flip, "
    "1.0 = perfect ranking.",
    "PR-AUC (precision-recall AUC) is the same idea but plots precision against recall. On "
    "imbalanced data it is far stricter: a model that flags almost nobody gets punished, because "
    "its recall collapses. That is why PR-AUC is the primary metric here.",
])

concept("Thresholds and the fixed 0.5 rule", [
    "The threshold is the probability above which we call someone a defaulter. This project uses "
    "0.5 for all cells so comparisons are like-for-like, and also reports the best achievable F1 "
    "over all thresholds as a reference.",
    "In production the threshold is a business dial: lower it to catch more defaulters (more false "
    "alarms), raise it to reduce false alarms (more misses). The cost/applicant metric quantifies "
    "that trade-off in money.",
])
doc.add_page_break()

# ================================================================ CH 9
doc.add_heading("Chapter 9: Validation Design - Two Tracks, Zero Leakage", level=1)
para("A single train/test split tells you one number and gives no sense of stability. This project "
     "runs two complementary evaluation tracks.")

doc.add_heading("9.1 Track 1: repeated stratified cross-validation (in-time)", level=2)
concept("K-fold cross-validation", [
    "Split the training data into K equal parts (folds). Train on K-1 folds, test on the held-out "
    "fold; rotate until every fold has been the test fold once. You get K performance estimates "
    "from one dataset, and their spread tells you how stable the model is.",
    "Stratified means every fold preserves the 8% default rate - otherwise a fold could randomly "
    "end up with almost no defaulters, making metrics meaningless.",
    "Repeated means the whole 5-fold procedure is run 3 times with different shuffles (seeds 42, "
    "1042, 2042). That gives 15 observations per configuration - enough for real statistics "
    "(Chapter 10).",
])
decision("CV runs on 2018-2019 only", [
    "The cross-validation track never touches 2020. Its purpose is model comparison and stability "
    "within the training era. The 2020 year is reserved as a completely untouched final exam for "
    "Track 2.",
    "Inside every one of the 15 folds: preprocessing re-fit on the train portion, SMOTE applied "
    "to the train portion only, cost weights computed from the train portion only. Nothing crosses "
    "fold boundaries.",
])

doc.add_heading("9.2 Track 2: out-of-time validation (the final exam)", level=2)
para("Everything is fitted once on the full 2018-2019 window (205,007 rows) and evaluated once on "
     "2020 (102,504 rows). This answers the only question that matters to the bank: if we deploy "
     "this model next year, how will it perform?")
concept("Why out-of-time beats random splitting in finance", [
    "Random splits mix years: the model trains on 2020 behaviour and is tested on 2019 - it "
    "effectively studies the future to predict the past. Scores inflate, then collapse at "
    "deployment.",
    "Temporal splits respect causality: the past predicts the future, never the reverse. Regulators "
    "and internal model-risk teams expect OOT evidence for exactly this reason.",
])
para("Both tracks use the same nine cells, the same metrics, and the same threshold - so results "
     "are directly comparable, and any gap between them measures temporal drift.")
doc.add_page_break()

# ================================================================ CH 10
doc.add_heading("Chapter 10: Results - What the Nine Cells Show", level=1)
import pandas as pd

cv = pd.read_csv(TBL / "cv_summary.csv")
oot = pd.read_csv(TBL / "oot_results.csv")

doc.add_heading("10.1 In-time results: 3x5 repeated cross-validation on 2018-2019", level=2)
para("Each number below is the mean over 15 folds (3 repeats x 5 folds). The table is sorted by "
     "PR-AUC, the primary metric:", bold=True)
cv_rows = []
for _, r in cv.iterrows():
    cv_rows.append([
        r["display_name"],
        f"{r['roc_auc_mean']:.4f} +/- {r['roc_auc_std']:.4f}",
        f"{r['pr_auc_mean']:.4f} +/- {r['pr_auc_std']:.4f}",
        f"{r['recall_mean']:.3f}",
        f"{r['f2_mean']:.3f}",
        f"{r['cost_mean']:.3f}",
    ])
make_table(["Configuration", "ROC-AUC (mean +/- SD)", "PR-AUC (mean +/- SD)",
            "Recall @0.5", "F2", "Cost/applicant"],
           cv_rows, widths=[1.9, 1.5, 1.5, 0.8, 0.7, 0.9], font_size=8.5)

doc.add_heading("10.2 Out-of-time results: train 2018-19, test on untouched 2020", level=2)
oot_rows = []
for _, r in oot.sort_values("roc_auc", ascending=False).iterrows():
    oot_rows.append([
        r["display_name"],
        f"{r['roc_auc']:.4f}",
        f"{r['pr_auc']:.4f}",
        f"{r['recall']:.3f}",
        f"{r['precision']:.3f}",
        f"{r['cost_per_applicant']:.3f}",
    ])
make_table(["Configuration", "ROC-AUC", "PR-AUC", "Recall @0.5", "Precision @0.5", "Cost/applicant"],
           oot_rows, widths=[1.9, 0.9, 0.9, 0.9, 1.0, 1.0], font_size=8.5)

doc.add_heading("10.3 Reading the ROC and PR curves", level=2)
figure(FIG / "roc_curves_oot.png",
       "ROC curves on the untouched 2020 test year for all nine configurations.", 6.3)
figure(FIG / "pr_curves_oot.png",
       "Precision-recall curves on 2020. The dashed grey line marks the 8% base rate - the 'no skill' level.", 6.3)
para("How to read these plots: every curve belongs to one configuration, traced as the decision "
     "threshold sweeps from 0 to 1. A curve that hugs the top-left corner is better. Notice three "
     "things:")
bullet("The three baseline curves (plain lines) sit highest overall - the raw ranking quality "
       "lives there.", bold_prefix="")
bullet("Cost-sensitive curves trade a little ranking quality for dramatically higher recall - "
       "their curves bend differently, staying usable at high-recall operating points.", bold_prefix="")
bullet("The SMOTE curves, especially Random Forest's, sink visibly - visual confirmation that "
       "SMOTE damaged the models.", bold_prefix="")

doc.add_heading("10.4 Confusion matrices: what actually happened to 102,504 applicants", level=2)
figure(FIG / "confusion_matrices_oot.png",
       "Confusion matrices for all nine cells on the 2020 test set.", 6.4)
para("The baselines produce an almost empty 'predicted default' column: XGBoost Baseline flags "
     "only 137 false positives but misses 7,998 of the 8,183 real defaulters (recall 2%). The "
     "cost-sensitive versions flip the picture: XGBoost Cost-Sensitive catches 5,374 defaulters "
     "(recall 65.7%) at the price of 25,088 false alarms. Neither extreme is 'wrong' - they are "
     "different business operating points, and the cost metric below adjudicates them.")

doc.add_heading("10.5 The money view: expected cost per applicant", level=2)
figure(FIG / "cost_curves_oot.png",
       "Expected cost per applicant (FN costs 10x an FP) as the decision threshold varies, on 2020.", 6.3)
para("With a false negative charged 10x a false positive, the baselines are the most expensive "
     "configurations (~0.79-0.81 cost units per applicant) because they miss almost every "
     "defaulter. Cost-sensitive learning cuts expected cost by roughly a third to ~0.52-0.55. "
     "SMOTE configurations stay expensive. The business conclusion and the statistical conclusion "
     "(next chapter) point the same way.")

decision("Champion: XGBoost (Baseline) - with a deployment caveat", [
    "XGBoost Baseline wins on both headline ranking metrics: best CV PR-AUC (0.2502) and best OOT "
    "ROC-AUC (0.7669).",
    "But the raw 0.5 threshold catches only 2% of defaulters. For deployment, the cost-sensitive "
    "XGBoost variant is the practical choice: identical ranking quality (ROC-AUC 0.7661), recall "
    "jumping to 66%, and the lowest expected cost of all nine cells (0.519).",
    "Same model family, same hyperparameters - only the mistake price list differs. That is the "
    "cleanest possible demonstration that cost-sensitive learning, not tuning, produced the gain.",
])
doc.add_page_break()

# ================================================================ CH 11
doc.add_heading("Chapter 11: Statistical Significance - Proving Differences Are Real", level=1)
para("XGBoost beat logistic regression by 0.015 ROC-AUC. Is that a real superiority or fold luck? "
     "With 15 paired observations per configuration, statistics can answer properly.")

concept("Paired tests and why pairing matters", [
    "Each configuration was measured on the exact same 15 folds. Fold 3 might be an 'easy' fold "
    "for everyone, fold 7 a 'hard' one. Comparing raw averages mixes model quality with fold "
    "difficulty.",
    "Paired tests remove fold difficulty from the picture by analysing the difference within each "
    "fold: (XGBoost score - LR score) on fold 1, on fold 2, ... on fold 15. If XGBoost is truly "
    "better, those 15 differences should be consistently positive.",
])
concept("p-values, t-tests and Wilcoxon", [
    "The p-value answers: if the two models were actually equal, how likely would we see a "
    "difference this big by chance? Small p-value = the observed gap is very unlikely under "
    "'no difference', so we conclude the difference is real. The conventional significance line "
    "is p < 0.05.",
    "The paired t-test assumes the differences look roughly bell-shaped. The Wilcoxon signed-rank "
    "test makes no such assumption - it only ranks the differences. Running both is insurance: "
    "when they agree, the conclusion is robust.",
    "Why 15 observations matter: with only 5 folds, Wilcoxon's smallest possible two-sided "
    "p-value is 0.0625 - it can NEVER reach significance however huge the gap. Repeating the CV "
    "3 times (n=15) was a deliberate design choice to make significance testable.",
])
concept("Effect size: Cohen's d", [
    "A p-value says 'is the difference real?'; an effect size says 'how big is it?'. Cohen's d "
    "divides the mean difference by its standard deviation: ~0.2 small, ~0.5 medium, ~0.8 large.",
    "With n=15, even tiny differences can become statistically significant, so effect sizes keep "
    "us honest about practical importance. Hedges' g is a small-sample-corrected cousin of d, "
    "reported alongside it.",
])

figure(FIG / "statistical_significance_heatmap.png",
       "Significance matrix: -log10(p) of the paired t-test for all 16 comparisons, with Cohen's d annotated.", 5.4)
para("15 of 16 comparisons are significant at the 5% level; 13 at the 1% level. The verdicts:",
     bold=True)
bullet("XGBoost beats both rivals decisively: vs logistic regression +0.0147 ROC-AUC "
       "(p = 1.0e-13, d = 7.3); vs random forest +0.0166 (p = 1.3e-17, d = 13.9).", bold_prefix="Model family: ")
bullet("Cost-sensitive recall gains are overwhelming for all three families (+0.62 to +0.67 "
       "recall, p < 1e-27). For XGBoost the ROC-AUC price is a significant but tiny -0.0022.", bold_prefix="Imbalance strategies: ")
bullet("SMOTE significantly hurts all three families (XGBoost -0.018, RF -0.032, LR -0.007 "
       "ROC-AUC; all p < 1e-11, all large effects).", bold_prefix="SMOTE verdict: ")
bullet("Logistic regression cost-sensitive vs baseline: p = 0.69, not significant - the weights "
       "barely moved LR's ranking, though recall still jumped.", bold_prefix="The one non-significant result: ")
para("The full numbers live in reports/tables/statistical_significance.csv (16 comparisons, each "
     "with t-statistic, both p-values, Cohen's d, Hedges' g and a 95% confidence interval).")
doc.add_page_break()

# ================================================================ CH 12
doc.add_heading("Chapter 12: Explaining the Model with SHAP", level=1)
para("A credit model that cannot be explained is a liability: regulators demand reasons, loan "
     "officers need to trust it, and rejected applicants deserve to know why. This chapter opens "
     "the black box.")

concept("SHAP values", [
    "SHAP (SHapley Additive exPlanations) borrows from game theory. Imagine each feature as a "
    "player in a game whose payout is the prediction. A feature's SHAP value is its fair-share "
    "contribution to that payout - computed by averaging over every possible coalition of features.",
    "For any single applicant: base prediction + sum of all feature SHAP values = the model's "
    "final output. The explanation is exact, not approximate.",
    "TreeSHAP computes these values exactly and fast for tree ensembles (used on the XGBoost "
    "champion). LinearSHAP does the same for the logistic regression, giving a linear-vs-tree "
    "comparison.",
])

figure(FIG / "shap_importance.png",
       "Global feature importance: mean |SHAP value| across a stratified sample of 2,000 training-window applicants.", 6.2)
para("The mean absolute SHAP value measures each feature's average impact on the prediction. The "
     "ranking is unambiguous:", bold=True)
make_table(
    ["Rank", "Feature", "Mean |SHAP|", "Plain meaning"],
    [
        ["1", "EXT_SOURCES_MEAN", "0.438", "Average external bureau score - by far the dominant signal"],
        ["2", "PAYMENT_RATE", "0.128", "Monthly payment relative to loan size"],
        ["3", "GOODS_CREDIT_RATIO", "0.079", "Item price vs loan amount"],
        ["4", "EXT_SOURCE_2_3_PROD", "0.077", "Joint bureau signal (engineered interaction!)"],
        ["5", "EXT_SOURCES_MAX", "0.073", "Best of the three bureau scores"],
    ],
    widths=[0.6, 2.0, 1.1, 3.0],
)
para("Two lessons hide in this table. First, the bureau scores dominate - EDA Finding #4 was "
     "right, and quantitatively so. Second, rank 4 is an engineered interaction feature: the "
     "Chapter 4 feature engineering measurably contributed to the model's reasoning.")

figure(FIG / "shap_beeswarm.png",
       "Beeswarm plot: each dot is one applicant; horizontal position = SHAP impact; colour = feature value.", 6.3)
concept("How to read a beeswarm plot", [
    "Each row is a feature; each dot is an applicant. Dots to the right pushed the prediction "
    "toward default; dots to the left pushed it toward repayment. Colour encodes the feature "
    "value: red = high, blue = low.",
    "Example reading: on the EXT_SOURCES_MEAN row, red dots (high bureau scores) sit on the left "
    "(toward repayment) and blue dots (low scores) sit on the right (toward default). The model "
    "has learned exactly the relationship the EDA histograms showed.",
])

figure(FIG / "shap_dependence.png",
       "Dependence panels: SHAP impact vs feature value for the three strongest financial features.", 6.4)
para("Dependence plots show the shape of each relationship. EXT_SOURCES_MEAN shows a clean "
     "monotonic slide: lower bureau score, higher default push. PAYMENT_RATE and "
     "CREDIT_INCOME_PERCENT show the expected financial logic - heavier repayment burdens push "
     "risk up.")

figure(FIG / "shap_case_studies.png",
       "Three individual applicants explained: a true positive, a true negative and a false positive.", 6.5)
para("Global plots describe the model; case studies describe decisions. Three real applicants "
     "from the SHAP sample:", bold=True)
bullet("Correctly flagged defaulter (P(default) = 0.77): driven by a very low average bureau "
       "score (SHAP +1.31), an aggressive payment rate (+0.37) and a low Bureau Score 2 (+0.31). "
       "Every factor tells the same story.", bold_prefix="Case 1 - ")
bullet("Clearly safe borrower (P(default) = 0.004): high bureau scores pull strongly toward "
       "repayment (-0.74), reinforced by provided verification documents (-0.24) and higher "
       "education (-0.15).", bold_prefix="Case 2 - ")
bullet("Borderline false alarm (P(default) = 0.52, actually repaid): a low bureau score alone "
       "(+0.91) dragged an otherwise decent applicant over the line. This is exactly the kind of "
       "case a manual-review queue exists for.", bold_prefix="Case 3 - ")
para("These per-applicant factor lists are precisely what GDPR Article 22-style adverse-action "
     "explanations require: concrete, individual, understandable reasons.")

doc.add_heading("12.1 Tree vs linear explanations", level=2)
para("The top-30 features ranked by TreeSHAP (XGBoost) and LinearSHAP (logistic regression) "
     "overlap on 16 of 30 entries. Both models agree on the dominant bureau-score signal; the "
     "tree model additionally exploits interactions and non-linear splits the linear model cannot "
     "express. That 16/30 agreement is itself evidence the findings are about the data, not about "
     "one algorithm's quirks.")
doc.add_page_break()

# ================================================================ CH 13
doc.add_heading("Chapter 13: Conclusions and Machine-Learning Lessons", level=1)

doc.add_heading("13.1 What the experiment concluded", level=2)
bullet("XGBoost (Baseline) is the champion: best CV PR-AUC (0.2502) and best OOT ROC-AUC "
       "(0.7669), significantly ahead of both rivals.", bold_prefix="1. ")
bullet("Cost-sensitive learning is the deployment choice: recall rises from ~2% to ~66% for a "
       "negligible -0.002 ROC-AUC, and expected cost per applicant drops by a third. All of this "
       "with frozen hyperparameters - the gain comes purely from re-pricing mistakes.", bold_prefix="2. ")
bullet("SMOTE hurts every model family on this data (significant, large effects; worst for random "
       "forest at -0.032 ROC-AUC). In 196 dimensions with 8% minority mass, synthetic neighbours "
       "manufacture noise rather than signal. Oversampling is not a universal fix.", bold_prefix="3. ")
bullet("Logistic regression stays competitive (within 0.015 ROC-AUC of XGBoost) while being "
       "fully transparent - a legitimate interpretable baseline and a fallback for regulation-heavy "
       "environments.", bold_prefix="4. ")
bullet("No temporal degradation: OOT performance matches in-time CV (0.7669 vs 0.7622), "
       "supporting deployment on future cohorts.", bold_prefix="5. ")

doc.add_heading("13.2 The doctrine: decisions specific to financial credit risk", level=2)
para("Every major choice in this project came from the domain, not from textbook defaults:",
     bold=True)
make_table(
    ["Doctrine", "What we did", "Why finance demands it"],
    [
        ["Respect time", "Temporal split + out-of-time test", "Models score the future, not the past; random splits leak"],
        ["Price mistakes honestly", "10:1 FN/FP cost ratio, cost metric", "A missed default costs the loan; a false alarm costs a review"],
        ["Distrust raw fields", "Sentinel fix, winsorization on train only", "Banking data is full of placeholders and entry errors"],
        ["Encode underwriter logic", "Debt-to-income, payment rate, tenure ratios", "Ratios, not raw amounts, carry credit meaning"],
        ["Validate statistics, not vibes", "3x5 repeated CV, paired tests, effect sizes", "Model-risk review demands evidence that gaps are real"],
        ["Explain every decision", "SHAP global + local, adverse-action case studies", "GDPR Art. 22; regulator and customer trust"],
        ["Freeze everything except the question", "One hyperparameter set per family", "Otherwise you cannot attribute gains to the intervention"],
    ],
    widths=[1.6, 2.4, 2.8],
)

doc.add_heading("13.3 Reproducibility", level=2)
para("The entire experiment regenerates from two commands (about 2.2 hours on the project "
     "hardware):", bold=True)
code_block("py run_experiment.py    # 3x5 repeated CV + OOT track  (~124 min)\n"
           "py run_analysis.py    # statistics, plots, SHAP          (~2 min)")
make_table(
    ["Artifact", "Location"],
    [
        ["Per-fold CV results (135 rows)", "reports/tables/cv_fold_results.csv"],
        ["CV summary / OOT results", "reports/tables/cv_summary.csv, oot_results.csv"],
        ["Statistical tests (16 comparisons)", "reports/tables/statistical_significance.csv"],
        ["SHAP tables (importance, case studies)", "reports/tables/shap_*.csv"],
        ["OOT probabilities + fitted preprocessor", "models/oot_probas.parquet, oot_preprocessor.joblib"],
        ["All figures (300 DPI)", "reports/figures/"],
        ["Run logs", "logs/experiment.log, logs/analysis.log"],
    ],
    widths=[3.0, 3.8],
)
para("Random seed 42 everywhere, with repeat seeds 42 / 1042 / 2042 for the three CV repeats.",
     italic=True, color=GRAY)

doc.add_heading("13.4 If you take five lessons into your own projects", level=2)
bullet("Leakage first: fit every statistic (caps, scalers, imputers, resamplers) on training data "
       "only, and split by time when the data has time.", bold_prefix="")
bullet("Pick metrics before models: on imbalanced data, accuracy deceives; PR-AUC and cost-weighted "
       "metrics tell the truth.", bold_prefix="")
bullet("Clean for the domain: sentinels, duplicate columns and manufactured correlations are data "
       "questions, and wrong answers silently poison everything downstream.", bold_prefix="")
bullet("Freeze what you are not testing: a fair experiment changes one variable at a time.", bold_prefix="")
bullet("Demand evidence and explanations: paired statistics prove a gap is real; SHAP shows why the "
       "model decides. Both are what separates a science project from a demo.", bold_prefix="")

para("")
para("End of writeup. All figures, tables and code referenced here ship with the project.",
     italic=True, color=GRAY, align="center")

# ================================================================ SAVE
OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"SAVED: {OUT}")
print(f"Figures embedded: {FIG_COUNTER[0]}")
print(f"Paragraphs: {len(doc.paragraphs)}, Tables: {len(doc.tables)}")
