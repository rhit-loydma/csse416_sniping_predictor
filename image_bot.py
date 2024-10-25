# This example requires the 'message_content' intent.

import discord
import csv
from datetime import datetime
import os
import shutil
import cv2 as cv
import numpy as np

LIMIT = 100
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 192

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

    # # create folders to save data
    # shutil.rmtree('Images/')
    # names = [""]
    # names += name_dict.values()
    # for name in names:
    #     path = "Images/" + name
    #     if not (name in exclude_list or os.path.exists(path)):
    #         os.mkdir(path)
    # print("finished creating folders to save images")

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

@client.event
async def on_message(message):
    global scoreboard

    # if the message is from the bot, do nothing
    if message.author == client.user:
        return
    
    # simple test case
    if message.content.startswith(bot_prefix + 'help'):
        help = "use /'$count_newts/' to tally the newt snipe scores."
        await message.channel.send(help)

    # command to count scores
    if message.content.startswith(bot_prefix + 'save_newts'):
        await bot_channel.send("Creating dataset from newts in sniped channels...")
        count, fname = await save_newts()
        await bot_channel.send(f"counted {count} newts. Logging to file")
        await bot_channel.send(f"Logging finished. Uploading File")
        await bot_channel.send(file=discord.File(fname))
        

async def save_newts():
    # clear the scorebaord
    global scoreboard
    scoreboard = []
    count = 0

    # initalize counts
    for name in name_dict.values():
        if not name in exclude_list:
            name_counts[name] = 0

    # loop through all messages in sniped
    async for message in sniped_channel.history(limit=LIMIT):
        # if the message has an image then score it
        if len(message.attachments) > 0:
            # add the scores for that message to the scorebaord
            scoreboard += await tally_message_score(message=message, was_aware=False, attachments=message.attachments)
            count = count + 1

    # loop through all messages in sniped-but-not
    async for message in sniped_but_not_channel.history(limit=LIMIT-count):
        # if the message has an image then score it
        if len(message.attachments) > 0:
            # add the scores for that message to the scorebaord
            scoreboard += await tally_message_score(message=message, was_aware=True, attachments=message.attachments)
            count = count + 1

    # grab the time and format it
    # fname = datetime.now().strftime('%a %d %b %Y, %I-%M%p-%S')
    # fname = fname + '.csv'
    # fname = "snipe_data.csv"

    # write scoreboard to a file and then upload the file
    # log_scoreboard_to_file(scoreboard, str(fname))

    fname = "snipe_data.npz"
    arr = np.array(scoreboard)
    np.savez_compressed(fname, arr)

    print(name_counts)
    return count, fname

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
            # fname = 'Images/' + snipee + "/" + snipee + "_" + str(name_counts[snipee]) + ".png"
            # # create dict for csv row
            # dt_obj = message.created_at
            # score_entry = {"filename" : fname,
            #             "sniper" : sniper,
            #             "snipee" : snipee,
            #             "was-aware": was_aware,
            #             "year": dt_obj.year,
            #             "month": dt_obj.month,
            #             "day": dt_obj.day,
            #             "hour": dt_obj.hour,
            #             "minute": dt_obj.minute,
            #             "second": dt_obj.second,
            #             "week_day": dt_obj.weekday()}
            # score_entrys.append(score_entry)

            label = names.index(snipee)

            # save image
            for file in attachments:
                if file.content_type.startswith("image/"):
                    # await file.save(fname)

                    # save to temp file
                    await file.save("temp.png")

                    # load image
                    img = cv.imread("temp.png")
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

                    # resize image
                    img = cv.resize(img, dsize=(IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv.INTER_CUBIC)

                    # change to list
                    # with the way flatten works, this will be in row major order
                    # with each pixel data in RGB order
                    entry = [label] + list(img.flatten())
                    score_entrys.append(entry)

            # update counts
            name_counts[snipee] += 1
    
    # print(score_entrys)
    return score_entrys

def log_scoreboard_to_file(scoreboard, filename=None):
    print("logging to file")
    # create a csv with the datetime as a file name
    with open(filename, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        # write the csv header and then dump each score entry to the file
        # filewriter.writerow(['filename', 'sniper', 'snipee', 'was-aware','year','month','day','hour','minute','second','week_day'])
        header = ["label"]
        for c in ["R", "G", "B"]:
            for y in range(IMAGE_HEIGHT):
                for x in range(IMAGE_WIDTH):
                    header.append("pixel" + str(y) + "_" + str(x) + "_" + str(c))
        filewriter.writerow(header)

        for score_entry in scoreboard:
            filewriter.writerow(score_entry)

    print("done logging to file")
        
    

def score_entry_to_csv_row(score_entry):
    row = []
    row.append(score_entry['filename'])
    row.append(score_entry['sniper'])
    row.append(score_entry['snipee'])
    row.append(score_entry['was-aware'])
    row.append(score_entry['year'])
    row.append(score_entry['month'])
    row.append(score_entry['day'])
    row.append(score_entry['hour'])
    row.append(score_entry['minute'])
    row.append(score_entry['second'])
    row.append(score_entry['week_day'])
    return row

client.run(token)