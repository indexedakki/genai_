# First install required packages
# pip install langchain langsmith huggingface_hub python-dotenv

import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langsmith import Client
from langchain.schema.runnable import RunnableSequence
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers.langchain import LangChainTracer

# Set up environment variables (replace with your credentials)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = ""  # Get from https://smith.langchain.com
os.environ["LANGSMITH_PROJECT"] = "pr-slight-switching-8"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""  # Get from https://huggingface.co/settings/tokens

# Initialize LangSmith client
client = Client()

# Create a LangChain tracer
tracer = LangChainTracer()

# Initialize Hugging Face LLM with tracing
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={"temperature": 0.5, "max_length": 256},
    callback_manager=CallbackManager([tracer]),
)

# Create a prompt template
prompt_template = """Answer the following question from context:
Context: Abstract
Introduction
This study aimed to investigate the efficacy and safety of sucroferric oxyhydroxide (SFOH) versus sevelamer carbonate in controlling serum phosphorus (sP) in adult Chinese dialysis patients with hyperphosphataemia (sP >1.78 mmol/L).

Methods
Open-label, randomised (1:1), active-controlled, parallel group, multicentre, phase III study of SFOH and sevelamer at starting doses corresponding to 1,500 mg iron/day and 2.4 g/day, respectively, with 8-week dose titration and 4-week maintenance (NCT03644264). Primary endpoint was non-inferiority analysis of change in sP from baseline to week 12. Secondary endpoints included sP over time and safety.

Results
415 patients were screened; 286 were enrolled and randomised (142 and 144 to SFOH and sevelamer, respectively). Mean (SD) baseline sP: 2.38 (0.57) and 2.38 (0.52) mmol/L, respectively. Mean (SD) change in sP from baseline to week 12: – 0.71 (0.60) versus −0.63 (0.52) mmol/L, respectively; difference (sevelamer minus SFOH) in least squares means (95% CI): 0.08 mmol/L (−0.02, 0.18) with the lower limit of 95% CI above the non-inferiority margin of −0.34 mmol/L. The SFOH group achieved target sP (1.13–1.78 mmol/L) earlier than the sevelamer group (56.5% vs. 32.8% at week 4) and with a lower pill burden (mean 3.7 vs. 9.1 tablets/day over 4 weeks of maintenance, respectively). Safety and tolerability of SFOH was consistent with previous studies, and no new safety signals were observed.

Conclusion
SFOH effectively reduced sP from baseline and was non-inferior to sevelamer after 12 weeks of treatment but had a lower pill burden in Chinese dialysis patients with hyperphosphataemia; SFOH benefit-risk profile is favourable in Chinese patients.

Keywords: Chronic kidney disease, Hyperphosphataemia, Phosphate binder, Sucroferric oxyhydroxide

Introduction
Hyperphosphataemia is a key abnormality of chronic kidney disease-mineral and bone disorder (CKD-MBD), a common complication of CKD, which may have serious adverse outcomes [1, 2]. Treatment of hyperphosphataemia is, therefore, one of the primary goals of CKD-MBD management. In addition to dietary restriction and adequate dialysis, both international and Chinese CKD-MBD guidelines recommend using phosphate binders (PBs) to lower elevated serum phosphorus (sP) levels towards the normal range [3, 4].

Sucroferric oxyhydroxide (SFOH) is an oral, non-calcium, iron-based PB developed to control sP levels [5]. Two pivotal efficacy and safety studies of SFOH in CKD dialysis patients (studies PA-CL-03A and PA-CL-05A/B) demonstrated a rapid reduction in sP levels at doses of SFOH corresponding to ≥ 1,000 mg iron/day which were maintained over 52 weeks in the extension study [6–8]. SFOH was well tolerated except for common gastrointestinal (GI) events such as diarrhoea, with an overall adverse event (AE) profile broadly similar to sevelamer [6–8]. Efficacy was largely unaffected by age, sex, race, or dialysis modality; however, the majority (96%) of patients were of White or Black race [8]. Additionally, four supportive phase III clinical studies have confirmed efficacy and tolerability of SFOH as both monotherapy and in combination with calcium carbonate in a Japanese population [9–12]. Based on these studies, SFOH was approved in the USA, EU, and Japan for the control of sP levels in patients with CKD on dialysis at a starting dose corresponding to 1,500 mg iron/day [13–15]. Following a paediatric study of SFOH [16], an extension to include patients ≥2 years of age, with CKD stages 4–5 or with CKD on dialysis, was approved in the EU [15]. Real-world studies have also indicated that SFOH is effective and well tolerated across different clinical settings [17].

The Chinese population differs from Western and Japanese populations in terms of race and ethnicity, including differences in dietary habits [18], medical practice patterns, and available medications [19]. An increasing number of patients in China are initiating dialysis [20], and there is a need to improve the control of sP levels [21]; Chinese dialysis patients were shown to have a higher mean sP level, with a lower proportion having target-range sP and a higher proportion having severe hyperphosphataemia compared with other world regions [22, 23]. The aim of this phase III study was, therefore, to investigate the efficacy and safety of SFOH in comparison with sevelamer carbonate (sevelamer) in controlling sP levels in adult Chinese dialysis patients with hyperphosphataemia and to provide a basis for registration of SFOH in China as required by the Chinese Center for Drug Evaluation.

Methods
Study Design and Patients
An open-label, randomised, active-controlled, parallel group, multicentre, phase III study was carried out to investigate the efficacy and safety of SFOH in adult Chinese patients with CKD and hyperphosphataemia (NCT03644264) in comparison with sevelamer. The 23-week study, carried out at 14 centres in China, consisted of screening (≤4 weeks), washout (≤3 weeks, for patients previously taking PBs), dose titration (8 weeks) (stage 1), 4-week maintenance (stage 2), and follow-up of 30 days (Fig. 1).

Fig. 1.
Fig. 1.

Open in a new tab
Study design. SFOH, sucroferric oxyhydroxide.

Key inclusion criteria were maintenance HD or peritoneal dialysis (PD) for ≥12 weeks prior to screening and hyperphosphataemia (sP >1.78 mmol/L [>5.5 mg/dL]) at screening or during washout. Key exclusion criteria were intact parathyroid hormone (iPTH) levels >800 ng/L; planned or expected parathyroidectomy within the next 6 months; serum total calcium >2.6 mmol/L or <1.9 mmol/L; history of haemochromatosis; serum ferritin >800 µg/L or transferrin saturation (TSAT) > 50%; alanine aminotransferase or aspartate transaminase ≥3 times the upper limit of normal range; and PD with a history of peritonitis in the last 3 months or ≥3 episodes in the last 12 months.

Patients were randomised (1:1) via interactive response technology to either SFOH or sevelamer at starting doses corresponding to 1,500 mg iron/day (3 tablets) or 2.4 g/day (3 tablets) and maximum doses corresponding to 3,000 mg iron/day (6 tablets) and 14.4 g/day (18 tablets), respectively. During dose titration, doses were increased or decreased as required to achieve target-range sP of 1.13–1.78 mmol/L (3.5–5.5 mg/dL) [24], provided the patient had received that dose for ≥2 weeks and for safety or tolerability reasons at any time. During maintenance, patients continued on the dose established during dose titration; however, dose changes for safety or tolerability reasons were allowed.

Study Objectives and Endpoints
The primary objective was to evaluate the efficacy of SFOH in comparison with sevelamer in lowering sP levels after 12 weeks of treatment. Secondary objectives were to evaluate the efficacy of SFOH and sevelamer over time, the efficacy at each timepoint in terms of the percentage of patients achieving target-range sP, the safety of SFOH, and the safety and tolerability of SFOH versus sevelamer.

The primary endpoint was the change in sP from baseline to week 12 with SFOH versus sevelamer. Secondary efficacy endpoints included assessment of sP levels at each time point and their change from baseline, and achievement of response, defined as the percentage of patients with sP within the target range of 1.13–1.78 mmol/L (3.5–5.5 mg/dL). Secondary safety endpoints included frequency of treatment-emergent AEs (TEAEs), withdrawals due to TEAEs, and biochemical/haematological laboratory evaluations including serum total calcium, serum iPTH, and blood iron parameters at each time point and their change from baseline.

Analysis Populations
The full analysis set (FAS) included all patients randomised to treatment receiving ≥1 dose investigational product and who had ≥1 baseline and 1 post-baseline assessment of sP level. The FAS was used to repeat the analysis for the primary endpoint, its sensitivity analyses, and secondary and other efficacy endpoints as secondary and supportive analyses.

The per-protocol set (PPS) included all patients who, in addition to the FAS criteria, had no major protocol deviations. The PPS population was the primary population for the analysis of efficacy parameters.

The safety analysis set (SAF) included all randomised patients who received ≥1 dose of the investigational product. The SAF was used for all safety analyses and for the analysis of stage 1 and overall drug exposure, as well as for prior and concomitant medications and procedures. The stage-2 SAF included all randomised patients who received ≥1 dose of the investigational product during stage 2. The stage-2 SAF was used for the analysis of stage 2 drug exposure.

Statistical Analyses
All statistical analyses were performed using SAS® Version 9.4 (SAS Institute Inc., SAS/STAT, Cary, NC). Descriptive statistics by the treatment group were provided for continuous and categorical variables. Two-sided confidence intervals (CIs) were calculated for the primary endpoint analyses and odds ratios.

The primary analysis consisted of an analysis of covariance (ANCOVA) using the last observation carried forward (LOCF) on the PPS. Change in sP level from baseline to week 12 was a dependent variable, baseline sP level a covariate, and treatment a fixed factor. The mean difference in change from baseline between SFOH and sevelamer and its 95% CI was estimated. This analysis considered sP levels from the local laboratory only if the central laboratory values were missing. Three sensitivity analyses of the primary endpoint were also performed (see online Suppl. methods for details; for all online Suppl. material, see https://doi.org/10.1159/000531869).

The secondary efficacy endpoint of percentage of patients with target-range sP consisted of logistic models to derive the odds ratios, using treatment and baseline phosphorus values as covariates, and calculation of Wald CIs. Time to achieving target-range sP in responders was analysed descriptively post hoc. The treatment groups were compared using a Cox proportional regression model adjusted for baseline sP level; the corresponding hazard ratio and p value were calculated.

Safety analyses were performed on the SAF and were descriptive. Pill burden was defined as the actual average daily number of tablets taken by the patient. Analysis of treatment compliance is described in online Supplementary methods.

Results
Study Populations, Patient Demographics, and Baseline Characteristics
Overall, 415 patients were screened, and 286 were enrolled and randomised: 142 to SFOH and 144 to sevelamer (Fig. 2). Patient numbers in different analysis populations were comparable between groups, with no major differences in treatment completion and reasons for early termination observed. COVID-19 had a limited impact on the study.

Fig. 2.
Fig. 2.

Open in a new tab
Patient disposition. SFOH, sucroferric oxyhydroxide.

Demographic and baseline characteristics were balanced between treatment groups. Overall, the median age of patients was 50.0 years, 60% of patients were male, and the primary reasons for CKD included glomerulonephritis, hypertension, and diabetes (Table 1). Mean (SD) baseline sP levels were 2.38 (0.58) mmol/L in the SFOH group and 2.38 (0.52) mmol/L in the sevelamer group.
Table 1.
Patient demographic and clinical characteristics (PPS)

Parameter	SFOH (n = 133)	Sevelamer (n = 136)	Total (N = 269)
Sex, n (%)
 Male	75 (56.4)	87 (64.0)	162 (60.2)
 Female	58 (43.6)	49 (36.0)	107 (39.8)
Age at informed consent, years
 Mean±SD	49.7±12.9	50.4±12.1	50.0±12.5
 Median [min., max.]	50.0 [20, 81]	50.0 [24, 77]	50.0 [20, 81]
 Chinese racea, n (%)	133 (100)	136 (100)	269 (100)
Baseline height, cm
 Mean±SD	165.5±8.0	166.0±7.6	165.7±7.8
 Median [min., max.]	165.0 [143, 186]	165.0 [148, 185]	165.0 [143, 186]
Baseline weight, kg
 Mean±SD	66.1±14.3	65.5±12.3	65.8±13.3
 Median [min., max.]	64.6 [39.0, 111.5]	64.2 [41.3, 104.0]	64.3 [39.0, 111.5]
Baseline BMI
 Mean±SD	24.0±4.2	23.7±4.0	23.9±4.1
 Median [min., max.]	23.5 [15.0, 35.6]	23.0 [16.1, 38.2]	23.3 [15.0, 38.2]
Dialysis type at baselineb, n (%)
 HD	113 (85.0)	107 (78.7)	220 (81.8)
 PD	20 (15.0)	29 (21.3)	49 (18.2)
Time from the first diagnosis of CKD to randomisation, yearsc, n [missing]	48 [85]	66 [70]	114 [155]
 Mean±SD	5.3±4.6	4.8±3.6	5.0±4.0
 Median [Q1, Q3]	3.8 [1.4, 7.5]	4.3 [1.5, 7.6]	4.2 [1.4, 7.6]
Time from the first dialysis to randomisation, yearsc, n [missing]	90 (43)	98 (38)	188 (81)
 Mean±SD	4.1±3.8	4.3±3.6	4.2±3.7
 Median [Q1, Q3]	2.8 [1.3, 6.4]	3.4 [1.7, 6.5]	3.0 [1.4, 6.5]
 [min., max.]	[0.3, 17.4]	[0.3, 17.7]	[0.3, 17.7]
Baseline sP, mmol/Ld
 Mean±SD	2.381±0.575	2.375±0.522	2.378±0.548
 Median [Q1, Q3]	2.270 [2.000, 2.800]	2.320 [1.985, 2.685]	2.280 [2.000, 2.760]
Baseline serum calcium, mmol/Ld
 Mean±SD	2.249±0.166	2.227±0.178	2.238±0.172
 Median [Q1, Q3]	2.270 [2.120, 2.370]	2.230 [2.120, 2.330]	2.250 [2.120, 2.350]
Primary reason for CKD, n (%), n [missing]	131 [2]	136 [0]	267 [2]
 Hypertension	25 (19.1)	19 (14.0)	44 (16.5)
 Glomerulonephritis	42 (32.1)	52 (38.2)	94 (35.2)
 Diabetic nephropathy	23 (17.6)	17 (12.5)	40 (15.0)
 Pyelonephritis	0	3 (2.2)	3 (1.1)
 Polycystic kidney disease	6 (4.6)	8 (5.9)	14 (5.2)
 Interstitial nephritis	3 (2.3)	1 (0.7)	4 (1.5)
 Hydronephrosis	1 (0.8)	1 (0.7)	2 (0.7)
 Congenital	0	0	0
 Other	31 (23.7)	35 (25.7)	66 (24.7)
Open in a new tab
BMI, body mass index; CKD, chronic kidney disease; HD, haemodialysis; PD, peritoneal dialysis; PPS, per-protocol set; SD, standard deviation; SFOH, sucroferric oxyhydroxide.

aParents are both Chinese.

bDialysis type at baseline uses information collected at informed consent if there is missing information at baseline.

cTime to randomisation was missing when the date specifying the time to randomisation was missing or partially missing.

dBoth central and local laboratory assessments were considered; however, local laboratory value was considered only if the central laboratory value at baseline was missing (note: central laboratory baseline data were available for all patients).

Drug Exposure and Compliance
In the study, overall, mean (SD) duration of drug exposure was 75.8 (22.5) days (median 85.0 days) and 78.5 (21.0) days (median 85.0 days) in the SFOH and sevelamer groups, respectively (online Suppl. Table 1). The actual average daily number of SFOH tablets taken was lower than for sevelamer: mean (SD) 3.2 (0.9) versus 6.3 (2.6) tablets during the overall study, 3.1 (0.8) and 5.2 (1.9) tablets during titration, and 3.7 (1.4) and 9.1 (3.9) tablets during maintenance for SFOH and sevelamer, respectively (online Suppl. Table 2). During the overall study in the SFOH and sevelamer groups, mean (SD) treatment compliance was 93.6% (9.2%) and 92.6% (11.8%), and dose adjustments for efficacy were done in 76.6% and 84.7%, with dose increases for efficacy in 58.9% and 84.7%, respectively (online Suppl. Table 1).

Efficacy
Mean (SD) change in sP level from baseline to week 12 for the LOCF endpoint in the PPS was comparable in the SFOH and sevelamer groups: −0.71 (0.60) mmol/L versus −0.63 (0.52) mmol/L (Fig. 3; Table 2a). The difference (sevelamer minus SFOH) in least squares (LS) means (standard error [SE]) using the ANCOVA-LOCF model without interaction was 0.08 mmol/L (0.05), 95% CI of −0.02, 0.18 (Table 2b). The lower limit of the 95% 2-sided CI for the difference was above the non-inferiority margin of −0.34 mmol/L; non-inferiority of SFOH versus sevelamer was therefore demonstrated. Moreover, in all three sensitivity analyses, the lower limit of the 95% CI for the difference was above the non-inferiority margin of −0.34 mmol/L (online Suppl. Table 3).

Fig. 3.
Fig. 3.

Open in a new tab
Primary outcome: sP change from baseline to week 12: LOCF endpoint (PPS). †Values from both central and local laboratories were considered; however, local laboratory assessment was considered only if the central laboratory assessment was missing. Change in sP levels from baseline to week 12 (stage 2) was computed as if the sP level at week 12 (stage 2) analysis visit was missing or was 2 days after the last day of study treatment administration or later, then LOCF approach was applied, and the LOCF was the last on-treatment laboratory value available excluding all data before the week 4 (stage 1) analysis visit but including values on the day after the last day of study treatment. The box is defined by two lines, Q1 (lower) and Q3 (upper), and the distance between both lines of the box is the interquartile range (IQR). Inside the box, the line represents the median (Q2), while the dot represents the mean. The whisker boundaries represent the minimum and maximum. LOCF, last observation carried forward; PPS, per-protocol set; SFOH, sucroferric oxyhydroxide.

Table 2.
Primary efficacy outcome: sP concentrations and change from baseline to week 12 (PPS)†

a Absolute values and change from baseline
sP, mmol/L	SFOH (n = 133)	Sevelamer (n = 136)
value	change from BL	value	change from BL
Visit
BL, n	125		128	
 Mean±SD	2.389±0.582	2.397±0.510
 Median [Q1, Q3]	2.270 [2.000, 2.820]	2.335 [2.005, 2.705]
Week 12 (stage 2), n	116	116	117	117
 Mean±SD	1.675±0.466	−0.693±0.600	1.731±0.374	−0.660±0.520
 Median [Q1, Q3]	1.645 [1.320, 1.950]	−0.695 [–1.090, −0.335]	1.710 [1.450, 1.970]	−0.650 [–0.920, −0.370]
LOCF endpoint‡, n	125	125	128	128
 Mean±SD	1.681±0.465	−0.708±0.604	1.763±0.422	−0.634±0.519
 Median [Q1, Q3]	1.640 [1.370, 1.940]	−0.700 [–1.150, −0.340]	1.725 [1.445, 2.030]	−0.640 [–0.915, −0.305]
b Change from baseline: ANCOVA-LOCF model without interaction§
SFOH (n = 133)	Sevelamer (n = 136)
Number of patients in the ANCOVA model	125	128
LS, means (SE) [95% CI]	−0.711 (0.037) [–0.784, −0.638]	−0.631 (0.0365) [–0.703, −0.559]
Difference in LS, means (SE) [95% CI]	0.080 (0.052) [−0.0221, 0.1824]
Open in a new tab
ANCOVA, analysis of covariance; BL, baseline; CI, confidence interval; LOCF, last observation carried forward; LS, least squares; PPS, per-protocol set; SD, standard deviation; SE, standard error; SFOH, sucroferric oxyhydroxide.

†Values from both central and local laboratories were considered; however, the local laboratory assessment was considered only if the central laboratory assessment was missing.

‡Change in sP levels from baseline to week 12 (stage 2) was computed as if sP level at week 12 (stage 2) analysis visit was missing or was 2 days after the last day of study treatment administration or later, then LOCF approach was applied, and the LOCF was the last on-treatment laboratory value available excluding all data before week 4 (stage 1) analysis visit but including values on the day after the last day of study treatment.

§The ANCOVA-LOCF model included change in sP levels from baseline to week 12 as a dependent variable, baseline sP as covariate and treatment as fixed factors. Endpoint results were calculated with week 12 (stage 2) analysis visit results or, if this value was missing, with the latest on-treatment laboratory measurement excluding all data before the week 4 (stage 1) analysis visit and including unscheduled visit data as well as values on the day after the last dose.

Absolute mean sP levels decreased from baseline in both treatment groups (Fig. 4a), with the decrease more pronounced in SFOH group at early timepoints (during stage 1): mean (SD) change from baseline with SFOH versus sevelamer −0.69 (0.55) mmol/L versus −0.37 (0.47) mmol/L at week 4, and –0.74 (0.56) versus −0.59 (0.50) mmol/L at week 8 (online Suppl. Table 4).

Fig. 4.
Fig. 4.

Open in a new tab
Secondary efficacy outcomes (PPS). a Absolute sP values over time (central and local laboratories). sP levels up to week 12 (stage 2) were computed as done for the primary endpoint, i.e., if sP level at week 12 (stage 2) analysis visit was missing or was 2 days after the last day of study treatment administration or later, then the LOCF approach was applied, and the LOCF was the last on-treatment laboratory value available excluding all data before the week 4 (stage 1) analysis visit but including values on the day after the last day of study treatment. Values from both central and local laboratories were considered; however, local laboratory assessment was considered only if the central laboratory assessment was missing. b Percentage of patients with sP within the target range. The target range was defined as sP 1.13–1.78 mmol/L (3.5–5.5 mg/dL). On-treatment values as well as values on the day after the last dose from both central and local laboratory assessments were considered; however, central laboratory values were used in preference to local laboratory values. LOCF, last observation carried forward; PPS, per-protocol set; SEM, standard error of the mean; SFOH, sucroferric oxyhydroxide; sP, serum phosphorus.

The proportion of patients with target-range sP increased earlier in the SFOH group: from 15.8% at baseline to 46.6% at week 1 versus 8.1% at baseline to 23.3% at week 1 in the sevelamer group and was comparable between groups at week 12 (Fig. 4b). There were statistically significant differences between the groups in the odds of achieving target-range sP during the first 4 weeks of treatment (online Suppl. Table 5). For responders, the median (Q1; Q3) time to achieving target-range sP with SFOH versus sevelamer was 1.9 (1.0, 4.0) versus 4.0 (1.0, 6.0) weeks, respectively (Table 3).

Table 3.
Post hoc analysis of time elapsed between the first dose and the first achievement of sP within the target range in responders (PPS)

SFOH (n = 99)	Sevelamer (n = 91)
Time, median (Q1, Q3), weeks	1.86 (1.00, 4.00)	4.00 (1.00, 6.00)
Treatment difference, HR (95% CI)	1.97 (1.46, 2.66)
p value	<0.001
Open in a new tab
The target range was defined as sP 1.13–1.78 mmol/L (3.5–5.5 mg/dL).

Responders were defined as patients achieving target-range sP at least once while on treatment excluding those with target-range sP at baseline. Baseline was defined as the last available (non-missing) assessment (scheduled or unscheduled) before or on the same day as the first dose of study treatment. Only on-treatment values as well as values on the day after the last dose from both central and local laboratory results were included. HR and p value for time taken to the first achievement of sP within target range come from a Cox proportional regression model adjusted for baseline sP value. On-treatment difference HR calculated for SFOH versus sevelamer.

CI, confidence interval; HR, hazard ratio; PPS, per-protocol set; SFOH, sucroferric oxyhydroxide; sP, serum phosphorus.

The proportion of patients with sP levels within the laboratory’s normal range also increased earlier in the SFOH group than the sevelamer group (35.1% vs. 10.5% at week 1) and was comparable between groups at week 12 (online Suppl. Fig. 1). There were significant differences between groups in the odds of achieving the laboratory’s normal range during the first 8 weeks of treatment (online Suppl. Table 5).

Safety
The proportion of patients in the SFOH versus sevelamer groups with any TEAEs was 83.0% versus 72.2% and with treatment-related TEAEs was 52.5% versus 27.8%, respectively (Table 4a). TEAEs leading to early study drug withdrawal were reported for 9 (6.4%) versus 5 (3.5%) patients in the SFOH versus sevelamer groups, respectively. Of the 74 patients who experienced ≥1 treatment-related TEAEs in the SFOH group, 62 patients experienced ≥1 treatment-related TEAEs within the system organ class of GI disorders, including faeces discolouration (n = 44) and diarrhoea (n = 17) (Table 4b). Of the TEAEs leading to study drug withdrawal, 10/27 events in 7 (5.0%) versus 6/10 events in 3 (2.1%) patients from the SFOH versus sevelamer groups, respectively, were related to study treatment (online Suppl. Table 6).

Table 4.
Adverse events (SAF)

a Treatment-emergent AEs until end of study
SFOH (n = 141) n (%) [E]	Sevelamer (n = 144) n (%) [E]
Any TEAE	117 (83.0) [405]	104 (72.2) [328]
Any treatment-related TEAE	74 (52.5) [130]	40 (27.8) [60]
Any serious TEAE	19 (13.5) [22]	13 (9.0) [17]
Any serious treatment-related TEAE	1 (0.7) [1]	2 (1.4) [2]
Any TEAE leading to study drug withdrawal	9 (6.4) [27]	5 (3.5) [10]
Any treatment-related TEAE leading to study drug withdrawal	7 (5.0) [10]	3 (2.1) [6]
Any serious TEAE leading to study drug withdrawal	1 (0.7) [1]	0
Any TEAE leading to death	1 (0.7) [1]	0
Any treatment-related TEAE leading to death	0	0
Any TEAE with mild intensity	108 (76.6) [312]	85 (59.0) [214]
Any TEAE with moderate intensity	24 (17.0) [46]	34 (23.6) [68]
Any TEAE with severe intensity	24 (17.0) [47]	22 (15.3) [46]
b Treatment-related treatment-emergent adverse events reported in ≥2% of patients in any group by system organ class and preferred term
SFOH (n = 141) n (%) [E]	Sevelamer (n = 144) n (%) [E]
Any treatment-related TEAE	74 (52.5) [130]	40 (27.8) [60]
 Gastrointestinal disorders	62 (44.0) [97]	30 (20.8) [36]
  Faeces discoloured	44 (31.2) [44]	0
  Diarrhoea	17 (12.1) [20]	4 (2.8) [4]
  Nausea	9 (6.4) [10]	4 (2.8) [4]
  Abdominal pain upper	7 (5.0) [7]	2 (1.4) [2]
  Abdominal distension	4 (2.8) [4]	0
  Vomiting	3 (2.1) [4]	1 (0.7) [1]
  Constipation	2 (1.4) [2]	11 (7.6) [13]
  Abdominal discomfort	0	6 (4.2) [6]
 Metabolism and nutrition disorders	8 (5.7) [9]	5 (3.5) [5]
  Decreased appetite	3 (2.1) [3]	4 (2.8) [4]
 Product issues	7 (5.0) [7]	5 (3.5) [5]
  Product taste abnormal	7 (5.0) [7]	5 (3.5) [5]
 Investigations	4 (2.8) [5]	3 (2.1) [4]
 General disorders and administration site conditions	3 (2.1) [4]	1 (0.7) [1]
 Skin and subcutaneous tissue disorders	2 (1.4) [2]	3 (2.1) [3]
  Pruritus	2 (1.4) [2]	3 (2.1) [3]
Open in a new tab
MedDRA Version 23.0 coding dictionary was applied. System organ classes were sorted in descending frequency as reported in the SFOH column; preferred terms were sorted in descending frequency within the system organ class. A TEAE was defined as any event that first occurs or worsens in intensity after the first dose of the investigational product. For AEs starting exactly on the first day of investigational product administration, the classification as treatment-emergent or pretreatment was based on the investigator’s judgement, as collected in the eCRF. If this information was missing, the AE was considered treatment-emergent. If a patient experienced more than one event in a given system organ class, that patient was counted once for the system organ class. If a patient experienced more than one event in a given preferred term, that patient was counted only once for that preferred term. A worst-case approach was followed in the event of missing causality data. If the causality of a TEAE was missing, the TEAE was considered related to the study drug.

E, total number of adverse events; eCRF, electronic case report form; MedDRA, Medical Dictionary for Regulatory Activities; n, number of patients: each patient counts only once for each adverse event; SAF, safety analysis set; TEAE, treatment-emergent adverse event; SFOH, sucroferric oxyhydroxide.

The proportions of patients with moderate and severe TEAEs were comparable between treatment groups (moderate, 17.0% vs. 23.6%, and severe, 17.0% vs. 15.3% in the SFOH group vs. the sevelamer group, respectively; Table 4a). Most of the patients (76.6%) in the SFOH group reported ≥1 TEAE considered to be mild in severity. One serious TEAE resulted in death; this event was considered unlikely to be related to SFOH (see online Supplementary results for details).

Anaemia was reported as a TEAE in 4 (2.8%) versus 11 (7.6%) patients in the SFOH versus sevelamer groups, respectively. Baseline mean serum ferritin levels were <200 µg/L. Mean levels of serum iron, serum ferritin, TSAT, and haemoglobin increased from baseline to week 12 in the SFOH group, but changes were less pronounced in the sevelamer group (online Suppl. Table 7). Transferrin levels decreased in the SFOH group but increased in the sevelamer group. The differences between SFOH and sevelamer groups in the changes from baseline were statistically significant (p < 0.05) for serum iron, ferritin, transferrin, TSAT, and haemoglobin. In both groups, there were minimal changes in levels of serum calcium, and iPTH, 25-hydroxyvitamin D, and 1,25-dihydroxyvitamin D levels decreased by a similar amount. Concomitant CKD-MBD and anti-anaemic medications are presented in online Suppl. Table 8.
Question: {question}
Answer: """

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["question"]
)

# Create a RunnableSequence (replaces LLMChain)
chain: RunnableSequence = prompt | llm

# Run the chain with tracing
response = chain.invoke(
    {"question": "how many patients were there in study"},
    config={"callbacks": [tracer]}
)

print("Response:", response)