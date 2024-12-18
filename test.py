import pandas as pd

# Create a test DataFrame with commentID and body columns
data = {
    "commentID": [1, 2, 3, 4, 5],
    "body": [
        "This is a test comment.",
        "I really enjoyed this article!",
        "I hate how this made me feel.",
        "Such a great read, highly recommend!",
        "This is the worst experience I've had."
    ]
}

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_file_path = 'C:\\Users\\aabha\\Downloads\\testNLC.csv'
df.to_csv(csv_file_path, index=False)

csv_file_path
