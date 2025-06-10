import discord
import requests
import random
import json
import os
from discord.ext import commands, tasks
from openai import AsyncOpenAI
from collections import defaultdict
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageOps
from io import BytesIO
import asyncio
import aiohttp
import configparser
import time

# ------------------------------------------------------------------
# Load configuration from config.ini
# ------------------------------------------------------------------
config = configparser.ConfigParser()
config.read("config.ini")

DISCORD_BOT_TOKEN = config.get("discord", "bot_token")
ALLOWED_CHANNEL_ID = int(config.get("discord", "allowed_channel_id"))
TRIVIA_DRAW_ROLE_ID = int(config.get("discord", "trivia_draw_role_id"))
OPENAI_API_KEY = config.get("openai", "api_key")
waiting_image_url = config.get("bot", "waiting_image_url")

# ------------------------------------------------------------------
# Set up OpenAI API (no change needed for AsyncOpenAI)
# ------------------------------------------------------------------
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ------------------------------------------------------------------
# In-memory context stores
# ------------------------------------------------------------------
user_context = defaultdict(list)
user_last_activity = {}
user_last_image = {}

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ------------------------------------------------------------------
# Points System Helpers
# ------------------------------------------------------------------
POINTS_FILE = "points.json"

def load_points():
    if not os.path.exists(POINTS_FILE):
        return {}
    try:
        with open(POINTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def save_points(points_dict):
    with open(POINTS_FILE, "w", encoding="utf-8") as f:
        json.dump(points_dict, f, indent=2, ensure_ascii=False)

def get_level(points):
    level = 1
    threshold = 500
    while points >= threshold:
        level += 1
        threshold *= 2
    return level

def get_next_level_threshold(points):
    threshold = 500
    while points >= threshold:
        threshold *= 2
    return threshold

# ---------------------------
# Global Trivia Variables (updated for points)
# ---------------------------
current_trivia = None  # {'question': str, 'answer': str, 'start_time': float, 'current_points': int, 'incorrect_count': int}
trivia_active = False
players_with_correct_answers = set()

# ------------------------------------------------------------------
# Optional JSON Helpers for storing winners
# ------------------------------------------------------------------
def load_trivia_winners():
    """
    Loads a list of winners from trivia_winners.json.
    If the file doesn't exist or is invalid, returns an empty list.
    """
    if not os.path.exists("trivia_winners.json"):
        return []
    try:
        with open("trivia_winners.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

def save_trivia_winners(winners_list):
    """
    Saves the provided list of winners to trivia_winners.json.
    """
    with open("trivia_winners.json", "w", encoding="utf-8") as f:
        json.dump(winners_list, f, indent=2, ensure_ascii=False)

# ---------------------------
# Utility Functions & Tasks
# ---------------------------
def is_allowed_channel(channel_id):
    return channel_id == ALLOWED_CHANNEL_ID

def update_last_activity(user_id):
    user_last_activity[user_id] = datetime.now()

@tasks.loop(minutes=1)
async def auto_reset_inactive_users():
    now = datetime.now()
    inactive_users = [
        user_id for user_id, last_activity in user_last_activity.items()
        if now - last_activity > timedelta(minutes=5)
    ]
    for user_id in inactive_users:
        if user_id in user_context:
            del user_context[user_id]
        del user_last_activity[user_id]
        if user_id in user_last_image:
            del user_last_image[user_id]

@bot.event
async def on_ready():
    print(f"Bot is ready and logged in as {bot.user}")
    auto_reset_inactive_users.start()

# ------------------------------------------------------------------
# Dynamic Pokemon Trivia Commands (One Correct Answer Only)
# ------------------------------------------------------------------
import re

@bot.command()
async def trivia(ctx):
    """
    Fetch a random Pokémon from the PokéAPI, build multiple hints (2 or 3),
    and present them together. Hints are based on:
      - (Optional) Partial flavor text snippet (with name hidden)
      - Color
      - Type(s)
      - Habitat
      - Ability to evolve (or not)
    We do NOT use base stats or moves.
    If the Pokédex snippet contains the Pokémon's name, we hide it.
    """
    if not is_allowed_channel(ctx.channel.id):
        return

    global current_trivia, trivia_active, players_with_correct_answers

    # If a trivia question is already active, repeat the same question
    if trivia_active and current_trivia is not None:
        await ctx.send(
            f"There is still an active trivia question!\n\n"
            f"{current_trivia['question']}\n\n"
            f"Use `!answer <Pokémon Name>` to answer!"
        )
        return

    # Clear any previous correct answers for this new question
    players_with_correct_answers.clear()

    # 1) Pick a random Pokémon
    pokemon_id = random.randint(1, 1010)  # Adjust as needed
    url_species = f"https://pokeapi.co/api/v2/pokemon-species/{pokemon_id}"
    url_pokemon = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url_species) as resp:
            species_data = await resp.json()
        async with session.get(url_pokemon) as resp:
            pokemon_data = await resp.json()

    # We'll need the name to hide it from the snippet
    pokemon_name = species_data["name"].lower()

    # 2) Gather potential hints
    hints = []

    # (A) PARTIAL Flavor Text (Optional, truncated), with name redaction
    english_flavors = [
        f["flavor_text"].replace("\n", " ").replace("\f", " ")
        for f in species_data.get("flavor_text_entries", [])
        if f["language"]["name"] == "en"
    ]
    if english_flavors:
        # Pick a random flavor text
        full_flavor = random.choice(english_flavors)

        # Hide the Pokémon name if it appears (case-insensitive).
        # We'll replace any match of the name (even partial substring) with "*****".
        # More sophisticated approaches can handle word boundaries, plurals, etc.
        redacted_flavor = re.sub(
            re.escape(pokemon_name),
            "*****",
            full_flavor,
            flags=re.IGNORECASE
        )

        # Now truncate the snippet to avoid giving away too much
        snippet_length = 50  # how many characters to show
        if len(redacted_flavor) > snippet_length:
            partial_flavor = redacted_flavor[:snippet_length].rstrip() + "..."
        else:
            partial_flavor = redacted_flavor

        hints.append(f"Pokédex snippet: \"{partial_flavor}\"")

    # (B) Color
    color_info = species_data.get("color")
    if color_info and color_info.get("name"):
        color_name = color_info["name"]
        hints.append(f"This Pokémon is primarily **{color_name}** in color.")

    # (C) Type(s)
    type_entries = pokemon_data.get("types", [])
    if type_entries:
        type_names = [t["type"]["name"] for t in type_entries]
        if len(type_names) == 1:
            hints.append(f"It is a **{type_names[0]}**-type Pokémon.")
        else:
            joined_types = " and ".join(type_names)
            hints.append(f"It has **{joined_types}** typing.")

    # (D) Habitat
    habitat_info = species_data.get("habitat")
    if habitat_info and habitat_info.get("name"):
        hints.append(f"Habitat is **{habitat_info['name']}**")

    # (E) Ability to evolve or not
    evolves_from = species_data.get("evolves_from_species")
    if evolves_from is None:
        # Possibly a base form or single-stage Pokémon; let's see if it can evolve further.
        evolution_chain_url = species_data["evolution_chain"]["url"]
        async with aiohttp.ClientSession() as session:
            async with session.get(evolution_chain_url) as resp:
                chain_data = await resp.json()

        def can_evolve(chain):
            """Returns True if there's a next evolution from 'chain' for this Pokémon's name."""
            if chain["species"]["name"].lower() == pokemon_name:
                return len(chain["evolves_to"]) > 0
            for evo in chain["evolves_to"]:
                if can_evolve(evo):
                    return True
            return False

        if can_evolve(chain_data["chain"]):
            hints.append("It **can evolve further** into another Pokémon.")
        else:
            hints.append("It **does not evolve further**.")
    else:
        # It's an evolved form
        evolves_from_name = evolves_from["name"]
        hints.append(f"This Pokémon evolved from **{evolves_from_name}**.")

    # 3) Randomly pick 2 or 3 hints
    num_hints = random.randint(4, 5)
    random.shuffle(hints)
    selected_hints = hints[:num_hints] if len(hints) >= num_hints else hints

    if not selected_hints:
        # Fallback if no hints are available (very unlikely)
        question_text = "Which Pokémon is it? (No hints found!)"
    else:
        # Combine them into a multi-line question
        lines = []
        for i, h in enumerate(selected_hints, start=1):
            lines.append(f"**Hint #{i}:** {h}")
        question_text = "Which Pokémon matches these hints?\n" + "\n".join(lines)

    # 4) Save final question & answer in global context
    current_trivia = {
        "question": question_text,
        "answer": pokemon_name,  # We already lowered it above
        "start_time": time.time(),
        "current_points": 100,
        "incorrect_count": 0
    }

    # Mark the trivia as active
    trivia_active = True

    # Send the hints as the question
    await ctx.send(
        f"**Pokémon Trivia Time!**\n{question_text}\n\n"
        f"Use `!answer <Pokémon Name>` to answer!"
    )

@bot.command()
async def answer(ctx, *, user_answer: str = None):
    """
    Command to answer the currently active trivia question.
    Once someone answers correctly, we stop accepting answers (clear current_trivia).
    """
    if not is_allowed_channel(ctx.channel.id):
        return

    global current_trivia, players_with_correct_answers, trivia_active

    # If there's no active trivia or it's already answered
    if not current_trivia or not trivia_active:
        await ctx.send("There is no active trivia question or it's already been answered! Use `!trivia` to start another.")
        return

    if not user_answer:
        await ctx.send("Please provide a Pokémon name after `!answer`.")
        return

    user_id = str(ctx.author.id)
    points_dict = load_points()

    # If already answered correctly this round
    if ctx.author.id in players_with_correct_answers:
        await ctx.send("You've already answered correctly for this question. Only one entry per member!")
        return

    # Calculate time since question posted
    now = time.time()
    elapsed = now - current_trivia["start_time"]

    # Points for correct answer: 100 - (5 points per 15s elapsed)
    time_penalty = int(elapsed // 15) * 5
    points = max(0, 100 - time_penalty)

    # Initialize prize pool if not present
    if "prize_pool" not in current_trivia:
        current_trivia["prize_pool"] = 0

    # Check answer
    if user_answer.lower().strip() == current_trivia["answer"]:
        players_with_correct_answers.add(ctx.author.id)
        # Award points (including prize pool)
        prev_points = points_dict.get(user_id, 0)
        total_award = points + current_trivia.get("prize_pool", 0)
        new_points = prev_points + total_award
        points_dict[user_id] = new_points
        save_points(points_dict)

        # Level up check
        prev_level = get_level(prev_points)
        new_level = get_level(new_points)
        if new_level > prev_level:
            await ctx.send(f"🎉 Congratulations {ctx.author.mention}, you reached **Level {new_level}** with {new_points} points! 🎉")

        if current_trivia["prize_pool"] > 0:
            await ctx.send(
                f"**Correct!** {ctx.author.mention}, you earned {points} points + {current_trivia['prize_pool']} bonus points from the prize pool. (Total: {new_points})"
            )
        else:
            await ctx.send(f"**Correct!** {ctx.author.mention}, you earned {points} points. (Total: {new_points})")
        current_trivia = None
        trivia_active = False
    else:
        # Incorrect answer: subtract 10 points, but not below 0, and add to prize pool
        prev_points = points_dict.get(user_id, 0)
        deduction = min(10, prev_points)
        new_points = max(0, prev_points - 10)
        points_dict[user_id] = new_points
        save_points(points_dict)
        current_trivia["incorrect_count"] += 1
        current_trivia["prize_pool"] = current_trivia.get("prize_pool", 0) + deduction
        await ctx.send(
            f"**Incorrect**, {ctx.author.mention}! 10 points have been deducted and added to the prize pool for this question. "
            f"(Total: {new_points}) Try again."
        )

# ------------------------------------------------------------------
# Example of Other Commands (Ask, Strategy, etc.)
# ------------------------------------------------------------------
@bot.command()
async def ask(ctx, *, query: str = None):
    if not is_allowed_channel(ctx.channel.id):
        return
    if not query:
        await ctx.send("Please provide a question or query after `!ask`.")
        return

    user_id = str(ctx.author.id)
    context = user_context[user_id]
    if not context:
        context.append({
            "role": "system",
            "content": (
                f"You are a Pokemon themed Discord bot named '{bot.user.name}'. "
                f"You were invented by BadWolf. You will answer queries, generate images, etc."
                f"You are currently hold a contest where members can use !trivia for a question and !answer to reply, if answered correct you are holding their names in a list for a future draw with prize yet to be determined."
                f"Members are allowed one entry in the draw only but are encouraged to help other members answer correctly as well."
                f"Non-Members (People who aren't subscribed to PokeScans) are still allowed to play but not allowed to win prizes, they will be skipped if drawn"
            )
        })

    context.append({"role": "user", "content": query})
    update_last_activity(user_id)

    try:
        max_token_limit = 4000
        while len(str(context)) > max_token_limit:
            context.pop(1)

        # Use the latest OpenAI chat completions endpoint
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=context
        )
        reply = response.choices[0].message.content.strip()
        context.append({"role": "assistant", "content": reply})

        if len(reply) > 2000:
            chunks = []
            while len(reply) > 2000:
                split_index = reply[:2000].rfind(" ")
                chunks.append(reply[:split_index])
                reply = reply[split_index:].strip()
            chunks.append(reply)
            for chunk in chunks:
                await ctx.send(chunk)
        else:
            await ctx.send(reply)

    except Exception as e:
        await ctx.send(f"An error occurred: {e}")
        
@bot.command()
async def draw(ctx):
    """
    Randomly pick a winner from trivia_winners.json.
    Only members with the specified role ID can use this.
    """
    # 1) Check if the user has the allowed role
    user_role_ids = [role.id for role in ctx.author.roles]
    if TRIVIA_DRAW_ROLE_ID not in user_role_ids:
        await ctx.send("You do not have permission to use this command.")
        return

    # 2) Load winners from file
    winners_list = load_trivia_winners()  # e.g. [{"name": "SomeUser", "id": "1234"}, ...]

    if not winners_list:
        await ctx.send("No winners found in trivia_winners.json.")
        return

    # 3) Pick a random winner
    selected_winner = random.choice(winners_list)

    # 4) Construct a mention for the selected winner
    #    We attempt to mention them using their user ID.
    winner_mention = f"<@{selected_winner['id']}>"

    # 5) Announce it!
    await ctx.send(
        f"**The randomly selected trivia winner is** {winner_mention} "
        f"(Discord Name: {selected_winner['name']})!"
    )


@bot.command()
async def strategy(ctx, *, pokemon: str = None):
    if not pokemon:
        await ctx.send("Please specify a Pokémon to get battle strategies and counters. For example: `!strategy Charizard`.")
        return
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Pokémon battle strategist with expertise in Pokémon Go mechanics. "
                        "Given a Pokémon, provide its best battle strategies, including move sets, types, and abilities. "
                        "Then suggest the top 6 Pokémon counters with reasons why they are effective."
                    )
                },
                {"role": "user", "content": f"Provide strategies and counters for battling {pokemon}."}
            ]
        )
        reply = response.choices[0].message.content
        await ctx.send(f"Battle strategies and counters for **{pokemon}**:\n{reply}")
    except Exception as e:
        await ctx.send(f"An error occurred while generating the strategy: {e}")

@bot.command()
async def team(ctx, *, theme: str = None):
    if not theme:
        await ctx.send("Please specify a theme for your team. For example: `!team water type`.")
        return
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Pokémon expert. Generate a creative Pokémon team of 6 "
                        "based on the given theme. Include reasons why each fits the theme."
                    )
                },
                {"role": "user", "content": f"Create a Pokémon team based on the theme: {theme}."}
            ]
        )
        reply = response.choices[0].message.content
        await ctx.send(f"Here’s a Pokémon team for the theme **{theme}**:\n{reply}")
    except Exception as e:
        await ctx.send(f"An error occurred while generating the team: {e}")

@bot.command()
async def image(ctx, *, prompt: str = None):
    if not is_allowed_channel(ctx.channel.id):
        return
    if not prompt:
        await ctx.send("Please provide a description of the image you want to generate.")
        return

    user_id = str(ctx.author.id)
    context = user_context[user_id]

    if not context:
        context.append({
            "role": "system",
            "content": (
                "You are a professional AI image generator. Generate images based on the user's prompts. "
                "Each new prompt builds on the previous unless reset."
            )
        })

    context.append({"role": "user", "content": prompt})
    waiting_message = await ctx.send("Generating your image, please wait... 🔄")

    try:
        combined_prompt = " ".join(msg["content"] for msg in context if msg["role"] == "user")
        # Use the latest OpenAI image generation endpoint
        response = await openai_client.images.generate(
            model="dall-e-3",
            prompt=combined_prompt,
            n=1,
            size="1024x1024",
            response_format="url"
        )
        image_url = response.data[0].url
        user_last_image[user_id] = image_url
        await waiting_message.edit(content=f"Here is your image: [Image Link]({image_url})")
    except Exception as e:
        if "content_policy_violation" in str(e):
            await waiting_message.edit(content=(
                "Your request seems to have violated content policies. "
                "Please try rephrasing your request."
            ))
        else:
            await waiting_message.edit(content=f"An error occurred while generating the image: {e}")

@bot.command()
async def reset(ctx):
    if not is_allowed_channel(ctx.channel.id):
        return
    user_id = str(ctx.author.id)
    user_context.pop(user_id, None)
    user_last_activity.pop(user_id, None)
    user_last_image.pop(user_id, None)
    await ctx.send("Your conversation history and image history have been reset.")

@bot.command()
async def edit(ctx, *, prompt: str = None):
    if not is_allowed_channel(ctx.channel.id):
        return
    if not prompt:
        await ctx.send("Please provide a description of how you want to edit the image.")
        return

    user_id = str(ctx.author.id)
    if user_id not in user_last_image:
        await ctx.send("No previously generated image found to edit.")
        return

    image_path = "processed_image.png"  # Adjust as needed
    try:
        with open(image_path, "rb") as f:
            processed_image_data = f.read()

        with BytesIO() as mask_stream:
            mask = Image.new("L", (1792, 1024), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([400, 600, 700, 800], fill=255)
            mask.save(mask_stream, format="PNG")
            mask_stream.seek(0)
            mask_data = mask_stream.read()

        mask.save("debug_mask.png")

        waiting_message = await ctx.send("Editing your image, please wait... 🔄")

        response = await openai_client.images.edit(
            model="gpt-image-1",
            image=processed_image_data,
            mask=mask_data,
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        edited_image_url = response.data[0].url
        await waiting_message.edit(content=f"Here is your edited image: [Image Link]({edited_image_url})")

    except Exception as e:
        await ctx.send(f"An error occurred while editing the image: {e}")

@bot.command()
async def translate(ctx, language: str, *, text: str = None):
    if not language or not text:
        await ctx.send("Please provide both the target language and the text to translate. Example: `!translate French Hello!`")
        return
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional translator. Translate the following text into the specified language."
                },
                {
                    "role": "user",
                    "content": f"Translate this text into {language}: {text}"
                }
            ]
        )
        translation = response.choices[0].message.content.strip()
        await ctx.send(f"**Translated to {language}:**\n{translation}")
    except Exception as e:
        await ctx.send(f"An error occurred while translating: {e}")

# ------------------------------------------------------------------
# Points Command
# ------------------------------------------------------------------
@bot.command()
async def points(ctx):
    user_id = str(ctx.author.id)
    points_dict = load_points()
    user_points = points_dict.get(user_id, 0)
    user_level = get_level(user_points)
    next_level = get_next_level_threshold(user_points)
    await ctx.send(
        f"{ctx.author.mention}, you have **{user_points}** points. "
        f"Level: **{user_level}**. "
        f"Next level at {next_level} points."
    )

# Finally, run the bot
bot.run(DISCORD_BOT_TOKEN)
