import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk

# Model definition (has to match the one from training)
class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=20):
        super(MatrixFactorization, self).__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)

    def forward(self, user, movie):
        return (self.user_factors(user) * self.movie_factors(movie)).sum(1)

# Dataset class for DataLoader (also same as in training)
class MovieLensDataset(Dataset):
    def __init__(self, ratings):
        self.users = torch.tensor(ratings.userId.values, dtype=torch.int64)
        self.movies = torch.tensor(ratings.movieId.values, dtype=torch.int64)
        self.ratings = torch.tensor(ratings.rating.values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

# Load data
ratings = pd.read_csv('data/ratings.csv')

# Assume the re-indexing of user and movie IDs was done during training, do the same here
user_ids = ratings["userId"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
ratings["userId"] = ratings["userId"].map(user2user_encoded)

movie_ids = ratings["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
ratings["movieId"] = ratings["movieId"].map(movie2movie_encoded)

# Initialize the model
n_users = ratings.userId.nunique()
n_movies = ratings.movieId.nunique()
model = MatrixFactorization(n_users, n_movies)

# Load the saved model weights
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Create a DataLoader instance for the test set
test_dataset = MovieLensDataset(ratings)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Predictions and accuracy calculation
true_ratings = []
predicted_ratings = []

# Add progress bar for prediction
loop = tqdm(test_loader, total=len(test_loader), leave=True)
with torch.no_grad():
    for user_batch, movie_batch, y_batch in loop:
        y_pred = model(user_batch, movie_batch)
        true_ratings.extend(y_batch.numpy().tolist())
        predicted_ratings.extend(y_pred.numpy().tolist())

        # Update the progress bar
        loop.set_description("Predicting")
        loop.set_postfix()

# RMSE calculation
rmse = sqrt(mean_squared_error(true_ratings, predicted_ratings))
print(f"Test RMSE: {rmse}")

# Accuracy calculation (you can define a threshold for accuracy, for simplicity, I'm considering a threshold of 0.5)
accurate_predictions = sum([1 for true, pred in zip(true_ratings, predicted_ratings) if abs(true - pred) < 0.5])
accuracy = accurate_predictions / len(true_ratings)
print(f"Accuracy (with a threshold of 0.5): {accuracy * 100:.2f}%")

# Visualization using Seaborn
plt.figure(figsize=(10,6))
sns.scatterplot(x=true_ratings, y=predicted_ratings, alpha=0.5)
plt.title("True Ratings vs. Predicted Ratings")
plt.xlabel("True Ratings")
plt.ylabel("Predicted Ratings")
plt.show()

def show_accuracy(accuracy):
    # Create the main window
    root = tk.Tk()
    root.title("Model Accuracy")

    # Add a label to display a message
    label = ttk.Label(root, text="Model Accuracy", font=("Helvetica", 24))
    label.pack(pady=10)

    # Add a label to display the accuracy with bigger font size
    accuracy_label = ttk.Label(root, text=f"{accuracy * 100:.2f}%", font=("Helvetica", 48))
    accuracy_label.pack(pady=10)

    # Add a label to explain the accuracy
    explanation_label = ttk.Label(root, text="This accuracy represents the percentage of predictions where the error is less than 0.5.", wraplength=400)
    explanation_label.pack(pady=10)

    # Add an exit button
    exit_button = ttk.Button(root, text="Exit", command=root.quit)
    exit_button.pack(pady=10)

    # Run the GUI loop
    root.mainloop()

# You would call this function with the calculated accuracy at the end of your testing code
accuracy = 0.95 # Example accuracy value
show_accuracy(accuracy)
