import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from math import sqrt

# Load data
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')

# Re-index user IDs
user_ids = ratings["userId"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
ratings["userId"] = ratings["userId"].map(user2user_encoded)

# Re-index movie IDs
movie_ids = ratings["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
ratings["movieId"] = ratings["movieId"].map(movie2movie_encoded)

# Define the model
class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=20):
        super(MatrixFactorization, self).__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)

    def forward(self, user, movie):
        return (self.user_factors(user) * self.movie_factors(movie)).sum(1)

# Dataset class for DataLoader
class MovieLensDataset(Dataset):
    def __init__(self, ratings):
        self.users = torch.tensor(ratings.userId.values, dtype=torch.int64)
        self.movies = torch.tensor(ratings.movieId.values, dtype=torch.int64)
        self.ratings = torch.tensor(ratings.rating.values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

# Splitting data
train, val = train_test_split(ratings, test_size=0.2)

# Create instances for DataLoader
train_dataset = MovieLensDataset(train)
val_dataset = MovieLensDataset(val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Number of unique users and movies
n_users = ratings.userId.nunique()
n_movies = ratings.movieId.nunique()

# Initialize the model
model = MatrixFactorization(n_users, n_movies)

# Loss and optimizer
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
n_epochs = 5
for epoch in range(n_epochs):
    # Add progress bar for each epoch
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    for batch_idx, (user_batch, movie_batch, y_batch) in loop:
        y_pred = model(user_batch, movie_batch)
        loss = loss_func(y_pred, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        loop.set_description(f"Epoch [{epoch+1}/{n_epochs}]")
        loop.set_postfix(loss=loss.item())

print("Training complete!")

# Validation
val_users = torch.tensor(val.userId.values, dtype=torch.int64)
val_movies = torch.tensor(val.movieId.values, dtype=torch.int64)
val_ratings = torch.tensor(val.rating.values, dtype=torch.float32)

with torch.no_grad():
    val_predictions = model(val_users, val_movies)
rmse = sqrt(mean_squared_error(val_ratings.numpy(), val_predictions.numpy()))
print(f"Validation RMSE: {rmse}")

torch.save(model.state_dict(), 'model.pth')
