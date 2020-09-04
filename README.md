# Latent-GAN
The files are split into Tracers and Velocity Fields above. Each folder contains the finalised models trained as part of this study.

To test the models, find the _predictAllAE.py_ or _predictAllGAN.py_ files in the respective models folder (Note: You will need to update the folder location to match your own data).

Trained models can be downloaded from: https://drive.google.com/file/d/1N8mvZOXjRzhjv0g9R62Ku5TvGfxAY7zt/view?usp=sharing

## Breakdown of files

**mainAE.py**
* Main file used for AutoEncoder training. See the respective files for Tracer / Velocity Field.

**mainGAN.py**
* Main file used for GAN training. See the respective files for Tracer / Velocity Field.

**getData.py** - Note that all data arrays are returned normalised
* _get_tracer_: retrieves a tracer array from a _.vtu_ file
* _get_tracer_from_latent_: retrieves a tracer array from the latent space representation from a _.csv_ file
* _get_prediction_tracer_: retrieves a predicted tracer array from a _.vtu_ file

* _get_velocity_: retrieves a tracer array from a _.vtu_ file
* _get_prediction_velocity_: retrieves a predicted tracer array from a _.vtu_ file

**Dataset.py**
* _TracerDataset_: Custom implementation of the Dataset Class from PyTorch. Dataset uses _get_tracer_ above. 
* _ToTensor_: Used to convert ndarrays to tensors
* _TracerLatentDataset_: Custom implementation of the Dataset Class from PyTorch. Dataset uses _get_prediction_tracer_ above. 

**Norm.py**
* _normalise_: Given an array/tensor along with its max and min value, normalises the array from between 0 and 1
* _denormalise_: Given the normalised array/tensor along with the original max and min value, denormalises the data back to between the real values i.e. the inverse of the previous function

**Variables.py** 
* Used to keep track of the variables used

**createVTU.py**
* _create_tracer_VTU_AE_:_ Creates a _.vtu_ file adding the AE reconstruction as a data array.
* _create_tracer_VTU_GAN:_ Same as above. The name of the created file represents the input timestep _t_. The prediction data array represents the prediction at _t+1_ and the tracer represents the ground truth at _t+1_.
* _create_velocity_field_VTU_AE_: Same as above, but for Velocity Fields.
* _create_velocity_field_VTU_GAN_: Same as above, but for Velocity Fields.

**convertToStructuredMesh.py**
* _get_structured_velocity_: Retrieves the velocity field data array from a _.vtu_ file and then remaps this to a structured mesh
* _convert_to_structured_: Given a velocity field data array, maps this to a structured mesh

**exportLatentSpace.py (Legacy)**
* Runs all timesteps through Encoder to a Latent Space representation and stores as _.csv_ files
