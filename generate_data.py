import pandas as pd
import numpy as np
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Create a larger list of diverse locations and colleges
locations_list = [f"City_{i}" for i in range(1, 101)]  # 100 unique locations
colleges_list = [f"College_{i}" for i in range(1, 200)]  # 200 unique colleges

# Generate weights for locations and colleges to simulate realistic distributions
locations_weights = np.random.dirichlet(np.ones(len(locations_list)), size=1).flatten()
colleges_weights = np.random.dirichlet(np.ones(len(colleges_list)), size=1).flatten()

# Set number of rows
n_rows = random.randint(10000, 15000)

# Generate Random Data
lead_ids = [f"LD{str(i).zfill(4)}" for i in range(1, n_rows + 1)]
Location = random.choices(locations_list, weights=locations_weights, k=n_rows)
College = random.choices(colleges_list, weights=colleges_weights, k=n_rows)
years_of_study = np.random.choice(["1st", "2nd", "3rd", "4th"], n_rows)
program_interests = np.random.choice(
    ["Data Science", "Robotics", "AI", "Electric Vehicle", "Cyber Security", "ML", "statistics"],
    n_rows,
    p=[0.25, 0.09, 0.1, 0.07, 0.2, 0.15, 0.14],  # Adjusting probabilities for diversity
)
lead_sources = np.random.choice(
    ["Instagram", "LinkedIn", "College Collaboration", "Google Form", "Mass-Mailing", "Whatsapp" ,"X"],
    n_rows,
    p=[0.2, 0.15, 0.25, 0.15, 0.1, 0.1, 0.05],  # Adjusting probabilities for diversity
)

# Combine into a DataFrame
dataset = pd.DataFrame({
    "Lead ID": lead_ids,
    "Location": Location,
    "College": College,
    "Year of Study": years_of_study,
    "Program Interest": program_interests,
    "Lead Source": lead_sources,
})

# Introduce Unequal Missing Values
missing_rates = {
    "Location": 0.03,       # 3% missing
    "College": 0.05,        # 5% missing
    "Year of Study": 0.02,  # 2% missing
    "Program Interest": 0.07,  # 7% missing
    "Lead Source": 0.04,     # 4% missing
}

# Apply different missing rates to each column
for column, rate in missing_rates.items():
    dataset.loc[dataset.sample(frac=rate).index, column] = np.nan

# Add Outliers
outlier_count = int(n_rows * 0.01)  # 1% of rows as outliers
outlier_indices = np.random.choice(dataset.index, size=outlier_count, replace=False)

# Outliers in "Year of Study" (e.g., invalid entries like '10th', 'Masters')
dataset.loc[outlier_indices[:outlier_count // 2], "Year of Study"] = "10th"

# Outliers in "Lead Source" (e.g., invalid or unusual sources)
unusual_sources = ["Unknown", "Other", "Referral"]
dataset.loc[outlier_indices[outlier_count // 2:], "Lead Source"] = np.random.choice(unusual_sources)

# Save to CSV
dataset.to_csv("e_learning_leads(7).csv", index=False)

print("Dataset created and saved as 'e_learning_leads(7).csv'")
