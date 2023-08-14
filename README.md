# Movie_rec-system-pytorch
A content-based movie recommendation system using PyTorch
# Movie Recommendation System

A content-based movie recommendation system developed by Dakshay Mehta. This system uses the MovieLens dataset to train a matrix factorization model in PyTorch.

## Dataset

- [MovieLens Dataset](link_to_dataset)

## Model

The recommendation model is based on matrix factorization. Given the user and movie, the model predicts the rating.

## Evaluation

The model's performance is evaluated using the RMSE (Root Mean Squared Error) on a validation dataset.

## API

The trained model is served as an API using Flask. To get a predicted rating:

POST /predict
{
    "user_id": <user_id>,
    "movie_id": <movie_id>
}

## Installation & Usage

(Include steps to set up and run your project)

## Author

Dakshay Mehta


