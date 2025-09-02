import keras
import keras_nlp
import keras_cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load Causal Language Model
causal_lm = keras_nlp.models.GemmaCausalLM.from_preset(
    "gemma_2b_instruct_en",
    dtype="float16",  # Make sure you have GPU and mixed precision enabled
)

prompt = """<start_of_turn>user
Write python code to print the first 100 prime numbers.
<end_of_turn>
<start_of_turn>model
"""

# Generate text
text_output = causal_lm.generate(prompt, max_length=256)
print("Generated Text:\n")
print(text_output[0])

# Text-to-image
text_to_image = keras_cv.models.StableDiffusion.from_preset(
    "stable_diffusion_3_medium",
    dtype="float16",
)

image_prompt = "Astronaut riding a horse on the moon, photorealistic"
generated_image = text_to_image.generate(image_prompt)[0]  # (H, W, 3) numpy array

# Convert to PIL image and display
image = Image.fromarray((generated_image * 255).astype(np.uint8))
image.show()
