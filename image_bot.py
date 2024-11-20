# This example requires the 'message_content' intent.

import discord
import csv
from datetime import datetime, timezone, timedelta
import os
import shutil
import cv2 as cv
import numpy as np

# LIMIT = 2000
IMAGE_HEIGHT = 256*2
IMAGE_WIDTH = 192*2
RGB = True
SAVE_INTERVAL = 100

intents = discord.Intents.default()
intents.message_content = True
token = open('discord_token.txt', 'r').read()

client = discord.Client(intents=intents)

# the family server for the bot to use
family_guild = None

# channels for the bot to use
sniped_channel = None
sniped_but_not_channel = None
bot_channel = None

bot_prefix = '$'
scoreboard = []

name_dict = {
    "lemon.yellow": "Abigail",
    "al.lyn": "Allyn",
    "tarzanvader": "Andrew",
    "annathomas.": "Anna",
    "pirate_raider": "Ben",
    "colleen_b.": "Colleen",
    "captainmaster": "Emily",
    "programedevelyn": "Evelyn",
    "iwisheggslikedme": "Garrett",
    "jjc8266": "Jacob",
    "roberj08": "Justin",
    "ktcollins8384": "Katie",
    "kjfan01": "Kiana",
    "kimtheklutz": "Kimmie",
    "marvelgirl21": "Lillian",
    "thatdude123": "Mike",
    "charcoal.newt": "Natasa",
    "1ricebowl1x": "Nathan",
    "nicki0909": "Nikki",
    "reimooney": "Reilly",
    "rielareal": "Reila",
    "spysh66": "Spencer",
    "stephisfalling": "Steph",
    "strawberry.poison": "Taytum",
    "ravenwings03": "Zoe",
    "suillut": "Thomas"
}

name_counts = {}
names = []

time_sections = [datetime(2023,11,1,0,0,0,0,timezone.utc) + timedelta(days=x) for x in range(0,370,5)]

exclude_list = ['Justin', 'Andrew', 'Colleen', 'Kimmie', 'Reila', 'Taytum', 'Thomas']

@client.event
async def on_ready():
    global family_guild
    global sniped_channel
    global sniped_but_not_channel
    global bot_channel
    global name_dict
    global name_counts
    global names

    # grab the family server    
    family = client.guilds[0]
    for ch in family.text_channels:
        # grab the sniped channel
        if ch.name == "sniped":
            sniped_channel = ch
        # grab the sniped but not channel
        if ch.name == "sniped-but-not":
            sniped_but_not_channel = ch
        if ch.name == "bot-channel":
            bot_channel = ch

    # quick verify that the channels and servers were grabed
    print(f'We have logged in as {client.user}')
    print("family id: " + str(family.id))
    print("sniped id: " + str(sniped_channel.id))
    print("sniped but not id: " + str(sniped_but_not_channel.id))
    print("bot channel id: " + str(bot_channel.id))

    # set up labels
    names = set(name_dict.values())
    names = set.difference(names, exclude_list)
    names = sorted(names)
    print(names)

    # save labels to file
    with open("labels.csv", 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        for name in names:
            filewriter.writerow([name])

    await save_newts()


def save_scoreboard(scoreboard, segment):
    fname = "snipe_data_" 
    if RGB:
        fname += "rgb_"
    else:
        fname += "gray_"
    fname += str(segment) + ".npz"
    arr = np.array(scoreboard, dtype=np.uint8)
    print(len(scoreboard))
    print(fname)
    np.savez(fname, arr)

async def save_newts():
    # clear the scorebaord
    global scoreboard
    scoreboard = []

    # initalize counts
    global name_counts
    for name in name_dict.values():
        if not name in exclude_list:
            name_counts[name] = 0


    channels = [sniped_channel, sniped_but_not_channel]
    for i in range(1, len(time_sections)):
        scoreboard = []
        for c in channels:
            async for message in c.history(limit=None, before=time_sections[i], after=time_sections[i-1]):
                # if the message has an image then score it
                if len(message.attachments) > 0:
                    # add the scores for that message to the scorebaord
                    scoreboard += await tally_message_score(message=message, was_aware=c==sniped_but_not_channel, attachments=message.attachments)
        save_scoreboard(scoreboard, i-1)
        print(f"Saved newt for time period: {time_sections[i-1]} to {time_sections[i]}")

    fname = "snipe_data.npz"
    arr = np.array(scoreboard, dtype=np.uint8)
    np.savez_compressed(fname, arr)

    print(name_counts)
    return fname

# take a message, look at the author and the @mentions, and 
# record score entrys to represent who sniped who
async def tally_message_score(message, was_aware, attachments):
    sniper = name_dict.get(str(message.author))
    score_entrys = []

    # each snipe is its own score entry. for example, if someone snipes
    # two people in one picture, then two score entries will be created.
    for mention in message.mentions:
        snipee = name_dict.get(str(mention.name))
        if (not ((sniper in exclude_list) or (snipee in exclude_list))):
            dt_obj = message.created_at
            sniper_index = names.index(sniper)
            metadata = [sniper_index, was_aware, dt_obj.month, dt_obj.day, dt_obj.hour, dt_obj.minute, dt_obj.weekday()]

            label = names.index(snipee)

            # save image
            for file in attachments:
                if file.content_type.startswith("image/"):
                    # await file.save(fname)

                    # save to temp file
                    await file.save("temp.png")

                    # load image
                    img = cv.imread("temp.png")
                    if RGB:
                        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    else:
                        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                    # resize image
                    img = cv.resize(img, dsize=(IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv.INTER_CUBIC)

                    # with the way flatten works, this will be in row major order
                    # with each pixel data in RGB order (if in color)
                    entry = [label] + metadata + list(img.flatten())
                    score_entrys.append(entry)

            # update counts
            name_counts[snipee] += 1
    
    # print(score_entrys)
    return score_entrys
