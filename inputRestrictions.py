import csv
import pandas as pd

# Read the CSV file
df = pd.read_csv('assets/2025_02_20_13_05_32.csv', index_col='Enrolled Patient #')

# Count unique values for each column
unique_counts = {col: df[col].nunique() for col in df.columns}

# # Print the results in a formatted way
# print("\nUnique value counts for each column:")
# print("-" * 50)
# for col, count in unique_counts.items():
#     print(f"{col}: {count} unique values")

# Optional: Save to a dictionary for later use
column_unique_counts = unique_counts
# Data Fields and Explanations
data_fields = [
    "Enrolled Patient #",          # Unique patient identifier
    "Age at consult",                # Patient's age at initial consultation
    "Race",                          # Patient's racial/ethnic background
    "Sex",                           # Patient's biological sex
    "ECOG",                          # Eastern Cooperative Oncology Group Performance Status (0-5 scale of functional status)
    "Enroll Date",                   # Date patient was enrolled in study
    "Consult date",                  # Date of initial consultation
    "Tumor site",                    # Anatomical location of the tumor
    "Prior Surgery",                 # Whether patient had surgery before this treatment
    "Biopsy Histology",              # Histological findings from initial biopsy
    "Diabetes",                      # Presence of diabetes
    "Smoker",                        # Smoking status
    "Biopsy Grade",                  # Tumor grade from initial biopsy
    "Surgery Grade",                 # Tumor grade from surgical specimen
    "Imaging Tumor Size",            # Size of tumor on imaging studies
    "T Stage",                       # Tumor size and extent (TNM staging system)
    "N Stage",                       # Lymph node involvement (TNM staging system)
    "M Stage",                       # Distant metastasis (TNM staging system)
    "AJCC Stage",                    # American Joint Committee on Cancer staging
    "End of RT Date",                # Date radiation therapy was completed
    "Treatment Machine",             # Type of radiation machine used
    "RT Technique",                  # Radiation therapy technique used
    "RT Dose",                       # Total radiation dose administered
    "V12 Skin Total ",                # Volume of skin receiving 12Gy
    "D0.5cc",                        # Maximum dose to 0.5cc volume
    "Acute Dermatological Toxicity", # Skin-related side effects during treatment
    "Acute Fatigue Toxicity",        # Fatigue-related side effects during treatment
    "Acute Toxicity - Other",        # Other acute side effects
    "Toxicity - Other Description",  # Description of other toxicities
    "Surgery Date",                  # Date of surgical procedure
    "Extent of Resection",           # How much of the tumor was removed
    "Histology on Surgery",          # Histological findings from surgical specimen
    "Histology Category",            # Category of tumor histology
    "Tumor Surgery Size",            # Size of tumor at surgery
    "Necrosis Score",                # Percentage of tumor necrosis
    "Wound Toxicity Acute",          # Acute wound-related complications
    "Major Wound Complications",     # Significant wound complications
    "Major Wound Complications AK + FCE", # Major wound complications in above knee and foot/ankle
    "Wound complication type",       # Type of wound complication
    "Date of wound closure",         # Date when wound was closed
    "Time to wound closure (days)",  # Days until wound closure
    "Fibrosis Grade @ 2 Years",      # Fibrosis severity at 2-year follow-up
    "Joint Stiffness Grade",         # Severity of joint stiffness
    "Edema Grade @ 2 years",         # Severity of swelling at 2-year follow-up
    "Other late >=G2 tox @ 2 yr",    # Other late grade 2 or higher toxicities at 2 years
    "Adjuvant Chemo Given?",         # Whether adjuvant chemotherapy was administered
    "Last follow up Primary imaging Date", # Date of last primary site imaging
    "Last follow up Distant Imaging Date", # Date of last distant site imaging
    "Last follow up date",           # Date of last patient follow-up
    "Local Recurrence",              # Whether cancer recurred at primary site
    "Local Recurrence Date",         # Date of local recurrence
    "Distant Recurrence?",           # Whether cancer spread to distant sites
    "Distant Recurrence Date",       # Date of distant recurrence
    "Survival",                      # Overall survival time
    "Date of Death",                 # Date of patient death
    "Zipcode",                       # Patient's zip code
    "Did the patient receive preoperative systemic therapy?", # Whether patient received systemic therapy before surgery
    "Distal Extremity",              # Whether tumor was in distal extremity
    "Method of Closure",             # Surgical closure technique used
    "EBL (CC)",                      # Estimated blood loss during surgery in cubic centimeters
    "Drain Use",                     # Whether surgical drain was used
    "Drain Type",                    # Type of surgical drain used
    "Margin Status",                 # Whether surgical margins were clear of tumor
    "Surgical Margin Value (mm)",    # Distance of tumor from surgical margin in millimeters
    "Drain End Date",                # Date surgical drain was removed
    "Drain Duration",                # Duration of drain use in days
    "Surgical Wound Complication",   # Any complications related to surgical wound
    "Soft Tissue Complication Type", # Type of soft tissue complications
    "Hyperbaric Oxygen",            # Whether hyperbaric oxygen therapy was used
    "Date of First Mention of Dehiscence", # First documentation of wound separation
    "Time to Dehiscence",           # Time until wound separation occurred
    "Secondary Surgery",            # Whether additional surgery was needed
    "Secondary Surgery Date",       # Date of additional surgery
    "Time to 2nd Surgery",          # Time until additional surgery was needed
    "Secondary Surgery Type",       # Type of additional surgery performed
    "Other Surgery",                # Any other surgical procedures
    "Bony Complication",            # Any complications involving bone
    "Bony Complication Type",       # Type of bone complications
    "Local Recurrence.1",           # Additional local recurrence information
    "Time to Local Recurrence",     # Time until local recurrence occurred
    "Amputation",                   # Whether amputation was performed
    "Date of Amputation",           # Date of amputation
    "Time to Amputation",           # Time until amputation was needed
    "Death",                        # Whether patient died
    "Complete?"                     # Whether patient's data collection is complete
]

numerical_variables = [
    "Age at consult",
    "Imaging Tumor Size",
    "V12 Skin Total ",
    "D0.5cc",
    "RT Dose",
    "Tumor Surgery Size",
    "Necrosis Score",
    "Time to wound closure (days)",
    "EBL (CC)",
    "Surgical Margin Value (mm)",
    "Drain Duration",
    "Time to Dehiscence",
    "Time to 2nd Surgery",
    "Time to Local Recurrence",
    "Time to Amputation"
]

categorical_variables = [
    "Race",
    "Sex",
    "ECOG",
    "Tumor site",
    "Prior Surgery",
    "Diabetes",
    "Smoker",
    "Biopsy Grade",
    "Surgery Grade",
    "T Stage",
    "N Stage",
    "M Stage",
    "AJCC Stage",
    "Treatment Machine",
    "RT Technique",
    "Acute Dermatological Toxicity",
    "Acute Fatigue Toxicity",
    "Acute Toxicity - Other",
    "Extent of Resection",
    "Histology Category",
    "Wound Toxicity Acute",
    "Major Wound Complications",
    "Major Wound Complications AK + FCE",
    "Fibrosis Grade @ 2 Years",
    "Joint Stiffness Grade",
    "Edema Grade @ 2 years",
    "Other late >=G2 tox @ 2 yr",
    "Adjuvant Chemo Given?",
    "Local Recurrence",
    "Distant Recurrence?",
    "Survival",
    "Did the patient receive preoperative systemic therapy?",
    "Distal Extremity",
    "Drain Use",
    "Drain Type",
    "Margin Status",
    "Surgical Wound Complication",
    "Hyperbaric Oxygen",
    "Secondary Surgery",
    "Bony Complication",
    "Local Recurrence.1",
    "Amputation",
    "Death",
    "Complete?"
]

ordinal_variables = [
    "ECOG",                          # Eastern Cooperative Oncology Group Performance Status (0-5)
    "Biopsy Grade",                  # Tumor grade from initial biopsy
    "Surgery Grade",                 # Tumor grade from surgical specimen
    "T Stage",                       # Tumor size and extent (TNM staging system)
    "N Stage",                       # Lymph node involvement (TNM staging system)
    "M Stage",                       # Distant metastasis (TNM staging system)
    "AJCC Stage",                    # American Joint Committee on Cancer staging
    "Acute Dermatological Toxicity", # Skin-related side effects during treatment
    "Acute Fatigue Toxicity",        # Fatigue-related side effects during treatment
    "Wound Toxicity Acute",          # Acute wound-related complications
    "Fibrosis Grade @ 2 Years",      # Fibrosis severity at 2-year follow-up
    "Joint Stiffness Grade",         # Severity of joint stiffness
    "Edema Grade @ 2 years"          # Severity of swelling at 2-year follow-up
]

# Scatter Plot Inputs
    #X Axis
spx = numerical_variables
    #Y Axis
spy = numerical_variables
    #Color Axis
spc = categorical_variables
    #Size Axis
spsize = ordinal_variables

# Histogram Inputs
    #X Axis
hx = numerical_variables
    #Group Axis
hgroup = categorical_variables

# Box Plot Inputs
    #X Axis
bx = numerical_variables
    #Y Axis
by = numerical_variables
    #Group Axis
bgroup = categorical_variables


