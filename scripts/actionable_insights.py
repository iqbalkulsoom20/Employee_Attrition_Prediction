import pandas as pd

# Step 1: Load the cleaned dataset
data = pd.read_csv("cleaned_data.csv")
print("Step 1: Dataset loaded successfully")

# Step 2: Analyze most influential features (from LIME explanation results)
insights = """
Actionable Insights:
1. Job Satisfaction: Focus on improving job satisfaction through feedback surveys and tailored programs addressing employee needs.
2. Overtime: Minimize excessive overtime by implementing better workforce planning and additional staffing where necessary.
3. Monthly Income: Offer competitive salary structures or performance incentives to retain employees.
4. Work-Life Balance: Promote initiatives that encourage work-life balance, such as flexible working schedules or hybrid work environments.
5. Training Opportunities: Provide more frequent and high-quality training programs to support career growth and employee satisfaction.
"""
print(insights)

# Step 3: Save insights to a text file
with open("actionable_insights.txt", "w") as f:
    f.write(insights)
print("Insights saved as 'actionable_insights.txt'")
