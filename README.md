# csse416_sniping_predictor
This repo contains our code to train the model on our snipe dataset. This data has been collected over the past year during a game our friends group plays where we take photos of each other without the "snipee" noticing. The photos are uploaded to a Discord server, where the snipee is tagged. We can use Discord's API to extract the data for our model, including metadata such as sniper (one-hot encoded), month, day, hour, minute, weekday, and if the snipee was aware or not. For the sake of privacy, we did not include the discord token that enables our [discord bot](image_bot.py) to scrape the data. However, the data can be found on the gauss server under group 4, named snipe_data_rgb_full_cropped.npz. 

## Link to Demo
[Hugging Face Space](https://huggingface.co/spaces/halseysh/Snipe-Prediction)

## Training the Model
To train the model, run the [ViT_Best.ipynb](ViT_Best.ipynb) jupyter notbeook with the snipe_data_rgb_full_cropped.npz data in the same directory. To run this code, you will need the torch, pytorch, transform, scikit-learn, numpy, pandas, opencv, and matplotlib libraries. <br/>

We've also included [ViT_with_metadata.ipynb](ViT_with_metadata.ipynb), though it did not perform as well. Our cropping/data cleaning tool is [crop_discard_ui.py](crop_discard_ui.py). 

## Example Outputs
![Katie](example_outputs/katie.jpg)
![Spencer](example_outputs/spencer.jpg)
![Reilly](example_outputs/reilly.jpg)
![Evelyn](example_outputs/evelyn.jpg)
![Predicting Allyn as Katie](example_outputs/allyn.jpg)
