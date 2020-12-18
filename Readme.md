# SinGAN

## Training the model
1. Run the following command in the root directory of the Project Folder

	`python main_train.py --gpu 0 --train_mode generation --input_name Images/Generation/mtrushmore.jpg`
2. The reconstructed images for each layer are stored in folders names 0, 1, 2, 3, 4, 5.
3. The generated images are stored in `TrainedModels/mtrushmoremore/'experiment'` folder (for each layer)
4. It is to be noted that every time the above command is executed on the terminal, a time stamped folder is created with the results, copy of the model checkpoint, copy of the files and

## Running other Post-Processing Techniques
1. Run the following command in the root directory of the Project Folder

	`python main_train.py --gpu 0 --train_mode animation --input_name Images/animation/lightning1.png`
2. In order to generate the gif files, we would need to call the `evaluate_model.py` file.  This file has a function that generates gifs from images.
`python evaluate_model.py --gpu 0 --model_dir TrainedModels/lightning1/PATH_OF_SPECIFIC_TIMESTAMPED_FOLDER`
3. Once step (2) is done, there will be a folder called **Evaluation** that would contain the gif files.

## Running the Zero-GP loss function
By default the `training_generation.py` file uses the **WGAN-GP** loss function. If we wish to replace it with the **Zero-GP** loss function, the simplest way would be to copy the all the contents from `training_generation_zerogp.py` to `training_generation.py`.
The code should be launched the same way as before: `python main_train.py --gpu 0 --train_mode generation --input_name Images/Generation/mtrushmore.jpg`

## Checking the effect of learning rate scale  and training train_depth
The code base contains ArgumentParser which is used to mention the desired arguments. In `config.py`, `--train_depth` is given as 3. This number can be tweaked based on experiment being run. Likewise, in `main_train.py`, `--lr_scale` can be changed.
The parameters that have been tweaked are:
````sh
--train_depth
--num_layer
--lr_scale
--train_stages
````
## Calcualting SIFID scores

1. Create 2 directories- one for the 'Real Images' and one for the 'Fake Images'. The corresponding images in both the folders should have the same name
 	(you can get the fake samples from the numbered folders specific to each stage. There would be 5 in each folder.)
2. Run this command `python3 ConSinGAN/sifid_score.py --path2real <real images path> --path2fake <fake images path>

## Link for the video presentation
[Video Presentation](https://youtu.be/sCyih8AzsSQ)
