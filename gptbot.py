from typing import List, Tuple
import time
import discord
import random
import os
from dotenv.main import load_dotenv
from discord.ext import commands
import pandas as pd
from surprise import accuracy, Dataset, SVD, Reader
import openai 

# pull environment variables from the .env file if they cannot be found in your OS environment
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPEN_AI_KEY")
model_id = "gpt-3.5-turbo" 

# intents form a list of actions that your bot may want to take on your server
intents = discord.Intents.default()
intents.message_content = True

# global variables to keep track of movie lens data
next_user_id = 0
discord_user_mapping = {}
movie_title_mapping = {}

# function to read the u.user table, populate discord_user_mapping with discord users
def load_users():
    global next_user_id
    df_users = pd.read_csv("./ml-100k/u.user", sep="|", names=["userID", "age", "gender", "username", "discordID"])
    discord_users = df_users[df_users["gender"] == "D"]
    for index, row in discord_users.iterrows():
        discord_user_mapping[int(row['discordID'])] = row['userID']
    next_user_id = max(df_users["userID"].tolist(), default=0) + 1

# load the user information before starting the bot
load_users()

# function to populate a dictionary that maps movieID to movie title
def load_movies():
    df_movies = pd.read_csv("./ml-100k/u.item", usecols=[0, 1], sep="|", names=["movieID", "title"], encoding='ISO-8859-1')
    for index, row in df_movies.iterrows():
        movie_title_mapping[int(row['movieID'])] = row['title']

# load movie information before starting the bot
load_movies()

# command prefix is added to the start of each command name to create the command
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'{bot.user} is now running!')

# Automatically register users if they are not already registered
async def auto_register_user(ctx: commands.Context):
    global next_user_id
    discord_user_mapping[ctx.author.id] = next_user_id
    with open('./ml-100k/u.user', 'a') as file:
        file.write(f"{next_user_id}|18|D|{ctx.author.name}|{ctx.author.id}\n")
    next_user_id += 1
    await ctx.send(f"user {ctx.author.name} has been auto-registered with id {discord_user_mapping[ctx.author.id]}")

# Add a command to register a discord user to the movie lens dataset
@bot.command("add_user")
async def add_user(ctx: commands.Context):
    if ctx.author.id in discord_user_mapping:
        return await ctx.send(f"user {ctx.author.name} is already registered with id {discord_user_mapping[ctx.author.id]}")
    await auto_register_user(ctx)

# Add a command for a registered discord user to post a rating for a movie
@bot.command(name="rate", description="Post a rating for a movie with given movie_id or title and rating")
async def rate(ctx: commands.Context, *, movie_and_rating: str):
    try:
        # Attempt to split into a movie ID and rating
        movie_id, rating = map(int, movie_and_rating.split())
    except ValueError:
        *title_words, rating = movie_and_rating.rsplit(' ', 1)
        rating = int(rating)
        movie_title = ' '.join(title_words).lower()
        movie_id = None
        
        for id, title in movie_title_mapping.items():
            if movie_title == title.lower():
                movie_id = id
                break

        if not movie_id:
            for id, title in movie_title_mapping.items():
                if movie_title in title.lower():
                    movie_id = id
                    break
        
        if movie_id is None:
            return await ctx.send("Could not find a matching movie title. Please try again with a different title or ID.")
    
    if ctx.author.id not in discord_user_mapping:
        await auto_register_user(ctx)

    user_id = discord_user_mapping[ctx.author.id]

    if movie_id not in movie_title_mapping:
        return await ctx.send("Invalid movie ID or title")

    if rating < 1 or rating > 5:
        return await ctx.send("Ratings must be between 1 and 5 inclusive")

    df_ratings = pd.read_csv("./ml-100k/u.data", sep="\t", names=["userID", "itemID", "rating", "timestamp"])
    existing_rating = df_ratings[(df_ratings['userID'] == user_id) & (df_ratings['itemID'] == movie_id)]

    if not existing_rating.empty:
        df_ratings.loc[(df_ratings['userID'] == user_id) & (df_ratings['itemID'] == movie_id), 'rating'] = rating
        df_ratings.to_csv("./ml-100k/u.data", sep="\t", header=False, index=False)
        return await ctx.send(f"Updated your rating for {movie_title_mapping[movie_id]} to {rating}.")
    else:
        with open('./ml-100k/u.data', 'a') as file:
            file.write(f"{user_id}\t{movie_id}\t{rating}\t0\n")

    await ctx.send(f"Your rating of {rating} has been registered for {movie_title_mapping[movie_id]}")

# Add a command to search the movie list for matching movies and return their ids
@bot.command("search")
async def search(ctx: commands.Context, *, search_text):
    matches = []
    for id, title in movie_title_mapping.items():
        if search_text.lower() in title.lower():
            matches.append((id, title))

    if len(matches) == 0:
        return await ctx.send("Could not find any titles that match your search")

    if len(matches) > 10:
        await ctx.send(f"Found {len(matches)} matches, showing the first 10.")
    matches = matches[0:10]
    output = "\n".join([f"{id}: {title}" for id, title in matches])

    return await ctx.send(output[0:2000])

# Add a command to use SVD to generate an estimated rating for user and movie
@bot.command("rec")
async def rec(ctx: commands.Context, movie_id: int):
    if ctx.author.id not in discord_user_mapping:
        return await ctx.send("You must register using !add_user before getting recommendations")

    user_id = discord_user_mapping[ctx.author.id]

    if movie_id not in movie_title_mapping:
        return await ctx.send("Invalid movie ID")

    df_ratings = pd.read_csv("./ml-100k/u.data", sep="\t", names=["userID", "itemID", "rating", "timestamp"])
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(df_ratings[["userID", "itemID", "rating"]], reader)
    trainset = dataset.build_full_trainset()

    algo = SVD(n_factors=5, n_epochs=200, biased=False)
    algo.fit(trainset)
    prediction = algo.predict(uid=user_id, iid=movie_id)

    if prediction.details["was_impossible"]:
        return await ctx.send("Not enough data to make a prediction")

    await ctx.send(f"Estimated rating is {prediction.est} for {movie_title_mapping[movie_id]}")

# Add a command to list the top 10 recommendations for the user
@bot.command("top10")
async def top10(ctx: commands.Context):
    if ctx.author.id not in discord_user_mapping:
        return await ctx.send("You must register using !add_user before getting recommendations")
    user_id = discord_user_mapping[ctx.author.id]

    df_ratings = pd.read_csv("./ml-100k/u.data", sep="\t", names=["userID", "itemID", "rating", "timestamp"])
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(df_ratings[["userID", "itemID", "rating"]], reader)
    trainset = dataset.build_full_trainset()

    algo = SVD(n_factors=5, n_epochs=20, biased=False)
    algo.fit(trainset)

    recommendations = []
    for movie_id in movie_title_mapping.keys():
        prediction = algo.predict(uid=user_id, iid=movie_id)
        recommendations.append((movie_id, prediction.est))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    top10 = recommendations[:10]
    output = "\n".join([f"{movie_title_mapping[movie_id]}: {rating}" for movie_id, rating in top10])

    await ctx.send(f"Top 10 recommended movies:\n{output}")

# Add a command to summarize all ratings submitted by a user
@bot.command("summary")
async def summary(ctx: commands.Context):
    if ctx.author.id not in discord_user_mapping:
        return await ctx.send("You must register using !add_user before getting recommendations")
    user_id = discord_user_mapping[ctx.author.id]

    df_ratings = pd.read_csv("./ml-100k/u.data", sep="\t", names=["userID", "itemID", "rating", "timestamp"])
    user_ratings = df_ratings[df_ratings['userID'] == user_id]
    if user_ratings.empty:
        return await ctx.send("You have not rated any movies yet.")
    
    output = ""
    for _, row in user_ratings.iterrows():
        output += f"{movie_title_mapping[row['itemID']]}: {row['rating']}\n"
    
    await ctx.send(f"Your ratings:\n{output}")

message_history = {}

# Add a command to generate a movie summary using GPT
@bot.command("synopsis")
async def synopsis(ctx: commands.Context, *, movie_title: str):
    query = f"Provide a brief summary for the movie titled '{movie_title}'."
    try:
        response = openai.ChatCompletion.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ],
            temperature=0.5,
            max_tokens=150
        )

        content = response.choices[0].message['content']
        await ctx.send(f"Synopsis for '{movie_title}':\n{content}")

    except Exception as e:
        await ctx.send("There was an error processing your request.")
        print(f"Error in synopsis command: {e}")

@bot.command("gpt")
async def gpt(ctx: commands.Context, *, query):
    if ctx.author.id not in message_history:
        message_history[ctx.author.id] = [{"role": "system", "content": "You are a helpful assistant, please keep responses below 2000 characters"}]

    message_history[ctx.author.id].append({"role": "user", "content": query})

    try:
        completion = openai.ChatCompletion.create(
            model=model_id,
            messages=message_history[ctx.author.id],
            temperature=0.5,
            timeout=30,
            max_tokens=400
        )
        finish_reason = completion.choices[0].finish_reason
        if finish_reason != "stop":
            return await ctx.send("Error Processing Your Request, Try Again Later")
        content = completion.choices[0].message['content']

        while len(message_history[ctx.author.id]) > 7:
            message_history[ctx.author.id].pop(1)

        message_history[ctx.author.id].append({"role": "assistant", "content": content})
        return await ctx.send(content)

    except Exception as e:
        await ctx.send("There was an error processing your request.")
        print(f"Error in gpt command: {e}")

bot.run(os.getenv("TOKEN"))  # stick your token in here instead if you can't figure out .env
